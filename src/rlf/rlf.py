import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
from utils.img_data_arrays import ImageDataArrays
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import loguniform
from image_downloading.hardcastle_catalogue import Source, HardcastleCatalogue
from utils.logging import get_logger
import logging
from tqdm import tqdm
from utils.fitfunctions import sigmoid

logger = get_logger( __name__, logging.DEBUG )

# from Hardcastle et al. 2022, https://github.com/mhardcastle/agn-selection/blob/main/plots.py
def ccol(i):
    colours=[[0,0,0],[0,73,73],[0,146,146],[255,109,182],[255,182,219],[73,0,146],[0,109,219],[182,109,255],[109,182,255],[182,219,255],[146,0,0],[146,73,0],[219,209,0],[36,255,36],[255,255,109]]
    i-=1
    return [v/255.0 for v in colours[i]]

colors=[ccol(2),ccol(3),ccol(11),ccol(4),ccol(5),ccol(6),ccol(8),ccol(7),ccol(9),ccol(10)]

def z_from_v( v, a, b ):
    """
    Redshift from comoving volume, where a and b are respective points in log-log space
    """
    return np.interp( v, a, b )

class RLF:
    """
    A class to calculate the radio luminosity function (RLF) of a sample of AGN using the method of Page & Carrera
    2000.
    """

    def __init__(self, cosmo = None):
        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(pth.PROGRAM_CONFIG)
        # we are using sources generated in a loguniform way
        lu_config = config['loguniform_distribution']

        # Dir names
        self.generated_subdir = lu_config['generated_subdir']
        self.dataset_subdir = lu_config['vm_dataset_subdir']

        # Cosmological Parameters
        self.h = float(lu_config['h']) # hubble constant = h * 100 km/s/Mpc
        self.Tcmb0 = float(lu_config['Tcmb0']) # temp of the CMB at z=0 in K
        self.Om0 = float(lu_config['Om0']) # matter density parameter at z=0

        # RLF parameters
        self.dz = float(lu_config['dz']) # redshift bin width
        self.lum_bins_count = int(lu_config['LUM_BINS']) # number of luminosity bins between min and max luminosity
        self.n_interp_pts = int(lu_config['N_INTERP_PTS']) # number of points to use in interpolation approximation of
        self.n_mc_pts = int(lu_config['N_MC_PTS']) # number of points to use in the monte-carlo integral for each redshift-luminosity bin
        self.spectral_index = float(lu_config['SPECTRAL_INDEX'])
        self.z_max = float(lu_config['Z_MAX'])
        self.z_min = float(lu_config['Z_MIN'])
        self.l_max = float(lu_config['L_MAX'])
        self.l_min = float(lu_config['L_MIN'])

        # initialize cosmology so we can define interpolation grids
        if cosmo == None:
            self.cosmo = astropy.cosmology.FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)
        else:
            self.cosmo = cosmo
        self.v_min, self.v_max = self.cosmo.comoving_volume( [self.z_min, self.z_max] ).to( u.Mpc**3 ).value
        logger.debug( f"volume range: {self.v_min}-{self.v_max}" )
        self.redshift_grid = np.geomspace( self.z_min, self.z_max, self.n_interp_pts )
        self.volume_grid = self.cosmo.comoving_volume( self.redshift_grid ).to( u.Mpc**3 ).value

        # define RLF z/l bins
        if lu_config.getboolean( 'HARDCASTLE_Z_BINS' ):
            self.z_bins = np.array( [ 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2 ] ) #hardcastle bins
        else:
            self.z_bins = np.arange( self.z_min, self.z_max, self.dz )

        self.l_bins = np.logspace( self.l_min, self.l_max, self.lum_bins_count)
        self.n_lum_bins = self.lum_bins_count - 1
        self.n_z_bins = self.z_bins.shape[ 0 ] - 1

        # params for interp
        self.zvparams = [ self.volume_grid, self.redshift_grid ]

        # init rlf values as zero
        self.phi = np.zeros((self.n_z_bins, self.n_lum_bins))


    def get_completeness(self, integ_fluxes, completeness_path=None):
        # todo: Read completeness function parameters from file
        #return 1.004 / ( 1 + np.exp(-6.995 * ( np.log10( integ_fluxes * 1000 ) - 0.483)) + -0.001)
        if completeness_path is None:
            completeness_path = pth.NP_ARRAY_PARENT / 'completeness_args_sigmoid.txt'
        if completeness_path.exists():
            completeness_args = np.loadtxt( completeness_path )
            #return sigmoid( integ_fluxes * 1000, *completeness_args )
        
        #for now use hardcastle completeness
        return np.where( integ_fluxes > 1.1e-3, 1, 0 )
    
    def get_completeness_from_coord( self, v: float | np.ndarray, l: float | np.ndarray, completeness_path=None ):
        #logger.debug( f'C[s(v,l)]: v={v.shape if isinstance( v, np.ndarray ) else v}, l={l.shape}, s={self.flux_from_coordinate( v, l, cosmo, zvparams )}')
        return self.get_completeness( self.flux_from_coordinate( v, l ), completeness_path )
    
    def flux_from_coordinate( self, v: float | np.ndarray, l: float | np.ndarray, z: float | np.ndarray | None = None ):
        """
        Generate luminosities + redshifts -> fluxes. Flux values here are in Jy, luminosities in W/Hz
        
        :param v: Volume(s)
        :type v: float | np.ndarray
        :param l: Luminosity/Luminosities
        :type l: float | np.ndarray
        :param z: Redshift override, v parameter ignored in this case
        :type z: float | np.ndarray | None
        """
        if z is None:
            z = z_from_v( v, *self.zvparams )

        d_l = self.cosmo.luminosity_distance(z).to(u.m).value
        s = 1e26 * l / (4 * np.pi * d_l**2) * (1+z)**(1+self.spectral_index)
        return s
    
    def one_dim_simpsons_rule( self, f, a, b, n ):
        """
        Perform integration using simpson's rule on a 1 dimentional function as integral from a to b of f( x ) dx

        :param f: Function to integrate
        :param a: Lower limit
        :param b: Upper limit
        :param n: Order of simpson's rule, must be divisible by 2
        """
        logger.debug( f'Simpsons rule order {n}' )

        h = ( b - a ) / n
        i = np.arange( 0, n+1, 1, dtype=int )


        # a and b may be passed as (1,n) arrays, so this allows for broadcasting
        if isinstance( a, np.ndarray ) and isinstance( a, np.ndarray ):
            i = i[ :, np.newaxis ]

        logger.debug( f'h={h.shape if isinstance( h, np.ndarray ) else h}, i={i.shape}' )

        x = a + i * h
        y = f( x )
        logger.debug( f'x={x.shape}, y={y.shape}' )
        return 1/3 * h * ( y[ 0 ] 
               + 4 * np.sum( y[ 1:-1:2 ], axis=0 )
               + 2 * np.sum( y[ 2:-1:2 ], axis=0 )
               + y[ -1 ] )
    
    def completeness_simpson_lum_integral( self, v: float, l_mins: np.ndarray, l_maxs: np.ndarray ):
        # integral from l_min to l_max of C[L] d(log10 L)
        # = integral from l_min to l_max of ln[10] C[L] / L dL
        # = ln[10] ( l_max - l_min ) / 6 * ( C[l_min]/l_min + C[l_max]/l_max + 8C[(l_max+l_min)/2]/(l_max+l_min) )

        logger.debug( f'Entering lum simpson integral from {l_mins.shape}-{l_maxs.shape}' )

        c_div_l = lambda l : self.get_completeness_from_coord( v, l ) / l
        return np.log( 10 ) * self.one_dim_simpsons_rule( c_div_l, l_mins, l_maxs, self.n_mc_pts )

    def simpson_integral( self, v_min: float, v_max: float, l_mins: np.ndarray, l_maxs: np.ndarray ):
        logger.debug( f'Entering volume simpson integral from {v_min}-{v_max}' )
        c = lambda v : self.completeness_simpson_lum_integral( v, l_mins, l_maxs )
        return self.one_dim_simpsons_rule( c, v_min, v_max, self.n_mc_pts )

    def monte_carlo_integral( self, v_min: float, v_max: float, l_min: float | np.ndarray, l_max: float | np.ndarray, lum: np.ndarray | float | None = None ):
        """
        Evaluate the Page & Carrera 2000 integral using monte-carlo methods for a given volume bin and set of luminosity bins

        lum is a 2-dim array with shape
        (n_mc_pts, n_integrals)

        where n_integrals is any of 1 (in the case of l_min/max being floats and lum being None or float), l_min/max.shape[ 1 ] (in the case of l_min/max being passed as arrays), or lum.shape[ 0 ] (in the case of lum being passed as an array)
        
        :param v_min: Bin volume minimum
        :param v_max: Bin volume maximum
        :param l_mins: Bin luminosity minimums, shape (1, n_lum_bins) or float
        :param l_maxs: Bin luminosity maximums, shape (1, n_lum_bins) or float
        :param lum: enforced luminosity to pass to completeness function, for use in the 1/Vmax method
        """
        if lum is None:
            if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
                lums = 10**np.random.uniform(np.log10(l_min[ 0, : ]), np.log10(l_max[ 0, : ]), size=(self.n_mc_pts, l_min.shape[ 1 ]))
            else:
                lums = 10**np.random.uniform(np.log10(l_min), np.log10(l_max), size=(self.n_mc_pts, l_min.shape[ 1 ]))

        elif isinstance( lum, np.ndarray ):
            if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
                raise AssertionError( 'Cannot have lum and l_min/max as nparrays' )

            if lum.ndim == 1:
                lums = lum[ np.newaxis, : ]
                lums = np.broadcast_to( lums, (self.n_mc_pts, lums.shape[ 1 ]) )
            elif lum.ndim == 2:
                lums = lum
            else: raise RuntimeError( f'lum arg of ndims {lum.ndim} invalid, must be at most 2' )
        
        else: #lum is float
            if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
                lums = np.broadcast_to( lum, (self.n_mc_pts, l_min.shape[ 1 ] ) )
            else:
                lums = np.broadcast_to( lum, (self.n_mc_pts, 1) )

        # lums now definitely has shape (self.n_mc_pts, n_integrals)

        # -- MONTE CARLO METHOD ---
        # Now generate random points in this bin; so random in volume and in luminosity, and then
        # compare to the completeness function to find the function of RLF in that bin.
        # random_fluxes has shape (self.n_mc_pts, n_lum_bins) for compat w/ np.random.uniform
        random_volumes = np.random.uniform(v_min, v_max, self.n_mc_pts)[ :, np.newaxis ]

        # weight each point by Completeness[ flux ] and add to total monte-carlo integral
        # for now, placeholder, assume flux cutoff at 1 mJy
        bin_integrals = np.sum( self.get_completeness_from_coord( random_volumes, lums ), axis=0) / self.n_mc_pts

        # divide by the luminosity-volume bin area so the result is / MPc^3 / (W/Hz)
        if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
            bin_integrals *= ( v_max - v_min ) * ( np.log10( l_max[ 0, : ] ) - np.log10( l_min[ 0, : ] ) )
        else:
            bin_integrals *= ( v_max - v_min ) * ( np.log10( l_max ) - np.log10( l_min ) )

        # if n_integrals is 1, cut out the last axis so we just return a 1d array of shape (n_lum_bins,)
        if isinstance( bin_integrals, np.ndarray ) and ( bin_integrals.shape[ -1 ] == 1 ):
            bin_integrals = bin_integrals[ :, 0 ]

        return bin_integrals

    def calculate_rlf( self, 
                            fluxes,
                            redshifts = None, 
                            luminosities = None,
                            debug_flux_lum_relation: bool = False,
                            vmax_method: bool = False ):
        """
        Calculate the Radio Luminosity Function, either using the Page & Carrera 2000 method or
        the traditional 1/Vmax method
        """
        logger.info( "start 1/Vmax rlf calculation" )

        # reset phi in case it's not our first time calling this
        self.phi = np.zeros( (self.n_z_bins, self.n_lum_bins) )

        if redshifts is None:
            # assign each source a comoving volume with a uniform dist s.t. dN/dV = const
            volumes = np.random.uniform( self.v_min, self.v_max, fluxes.shape[ 0 ] )

            # conversion from comoving volume to redshift
            redshifts = z_from_v( volumes, *self.zvparams )

        if luminosities is None:
            # use the redshift to calculate luminosity distance and luminosity
            luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
            luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 / ( 1 + redshifts )**(1+self.spectral_index) # W/Hz
        elif debug_flux_lum_relation:
            #ensure luminosities and redshifts are consistent with total flux
            # it's a lot easier to tell by visual inspection so save a scatterplot to debugdiff.png
            luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
            flux_luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 / ( 1 + redshifts )**(1+self.spectral_index) # W/Hz
            logger.info( 'saving scatterplot to compare luminosity from flux to luminosity from catalog...' )
            plt.scatter( flux_luminosities, luminosities, s=0.001 )
            plt.yscale( 'log' )
            plt.xscale( 'log' )
            plt.title( 'Luminosity from flux vs catalog' )
            plt.xlabel( 'luminosity from flux' )
            plt.ylabel( 'luminosity from catalog' )
            plt.grid()
            plt.savefig( 'debugdiff.png' )
            plt.figure()
            logger.info( 'saved debug luminosity diff figure to debugdiff.png' )

        # clip flux values to those above 0 for log plotting
        redshifts = redshifts[ fluxes > 0 ]
        luminosities = luminosities[ fluxes > 0 ]
        fluxes = fluxes[ fluxes > 0 ]

        if vmax_method:
            for i_z in range( self.n_z_bins ):
                z_min, z_max = self.z_bins[ i_z ], self.z_bins[ i_z+1 ]
                v_min, v_max = self.cosmo.comoving_volume([z_min, z_max]).to( u.Mpc**3 ).value
                redshift_mask = (redshifts >= z_min) & (redshifts < z_max)
                logger.debug( f'{luminosities[ redshift_mask ].shape[ 0 ]} sources in z: {z_min}-{z_max}' )

                for i_l in range( self.n_lum_bins ):
                    l_min, l_max = self.l_bins[ i_l ], self.l_bins[ i_l+1 ]

                    luminosity_mask = (luminosities >= l_min) & (luminosities < l_max)

                    luminosities_in_bin = luminosities[ redshift_mask & luminosity_mask ]

                    # for n=0, the RLF should be 0 regardless, and also it breaks the code so just ignore it
                    if not luminosities_in_bin.size:
                        logger.debug( f'no sources in z: {z_min}-{z_max}, l={l_min}-{l_max}' )
                        self.phi[ i_z, i_l ] = 0
                        continue
                    logger.debug( f'{luminosities_in_bin.shape[ 0 ]} sources in z: {z_min}-{z_max}, l={l_min}-{l_max}' )

                    Vmaxs = self.monte_carlo_integral( v_min, v_max, l_min, l_max, luminosities_in_bin )

                    self.phi[ i_z, i_l ] = 1.0 / ( np.log10( l_max ) - np.log10( l_min ) ) * np.sum( 1.0 / Vmaxs )
                logger.info( f'Redshift range {z_min:.2f}-{z_max:.2f} complete' )

        # otherwise, do Page & Carrera 2000 method
        else:
            for i_z in range(self.n_z_bins):
                z_min = self.z_bins[i_z]
                z_max = self.z_bins[i_z+1]

                # find min & max of comoving volume for redshift bin
                v_min, v_max = self.cosmo.comoving_volume([z_min, z_max]).to( u.Mpc**3 ).value

                # get luminosity bins from offset indices
                # and make them (1,n_lum_bins) arrays for broadcasting with (n_sources,1)
                # luminosity bins are defined by their minimum value
                l_mins = self.l_bins[ :-1 ][ np.newaxis, : ]
                l_maxs = self.l_bins[ 1: ][ np.newaxis, : ]

                # now calculate the number of 'real' sources in each bin
                # masks have shape:
                #   redshift_mask: (n_sources, 1)
                #   luminosity_mask: (n_sources, n_lum_bins)
                redshift_mask = (redshifts[ :, np.newaxis ] >= z_min) & (redshifts[ :, np.newaxis ] < z_max)
                luminosity_mask = (luminosities[ :, np.newaxis ] >= l_mins) & (luminosities[ :, np.newaxis ] < l_maxs)

                # shape (n_lum_bins)
                n_sources_in_lum_bins = np.sum( redshift_mask & luminosity_mask, axis=0 )
                logger.debug( f"n_sources_in_lum_bins: {n_sources_in_lum_bins}" )

                bin_integrals = self.monte_carlo_integral( v_min, v_max, l_mins, l_maxs )
                #bin_integrals = self.simpson_integral( v_min, v_max, l_mins, l_maxs )

                logger.debug( f"bin integrals: {bin_integrals}" )

                # if we have a 0 bin integral but N > 0 it must be a monte carlo failure or completeness mismatch
                if np.any( ( bin_integrals == 0 ) & ( n_sources_in_lum_bins > 0 ) ):
                    logger.error( f"Monte Carlo failure - {self.n_mc_pts} points insufficient for \
                        {np.sum( ( bin_integrals == 0 ) & ( n_sources_in_lum_bins > 0  ) )}/{bin_integrals.shape[ 0 ]} bins" )
                bin_integrals[ n_sources_in_lum_bins == 0 ] = 1

                # now we have phi_est as given by Page & Carrera 2000
                self.phi[i_z] = n_sources_in_lum_bins / bin_integrals

                logger.info( f'Redshift range {z_min:.2f}-{z_max:.2f} complete' )

        # plot the resulting graph
        self.plot_rlf( f'1/Vmax RLF - {self.n_mc_pts} pts per source', colors )

    def plot_rlf( self, 
                  title: str,
                  colors: list,
                  ylim: tuple = [ 1e-9, 3e-4 ],
                  xlim: tuple = [ 1e21, 1e29 ],
                  output: str = 'rlf.png' ):
        logger.info( "plotting..." )
        plt.figure( figsize=(10,10) )
        for i_z in range( self.phi.shape[0] ):
            specific_phi = self.phi[i_z]
            mask = specific_phi > 0
            plt.plot( self.l_bins[ :-1][ mask ], specific_phi[ mask ], color=colors[ i_z ],
                     marker='o',
                     label=f'{self.z_bins[ i_z ]:.2f}<z<{self.z_bins[ i_z + 1 ]:.2f}')
        plt.title( title )
        plt.xscale( 'log' )
        plt.xlabel( 'L144 * Hz / W')
        plt.yscale( 'log' )
        plt.ylabel( 'phi * MPc^3 * log10( W / m^2 )' )
        plt.xlim( xlim )
        plt.ylim( ylim )
        plt.legend()
        plt.savefig( output )
        logger.info( "saved figure" )



if __name__ == "__main__":
    rlf_calculator = RLF()
    catalog = HardcastleCatalogue( resolved_only=False )

    logger.debug( "getting catalog data" )
    redshifts = catalog.get_values( Source.Redshift )
    logger.debug( "   redshifts done" )
    fluxes = catalog.get_values( Source.TotalFlux ) / 1000
    logger.debug( "   fluxes done" )
    luminosities = catalog.get_values( Source.Luminosity )
    logger.debug( "   luminosity done" )

    redshift_lum_mask = ~(np.isnan( luminosities ) | np.isnan( redshifts ))
    hardcastle_flux_mask = fluxes > 1.1e-3

    mask = redshift_lum_mask & hardcastle_flux_mask

    redshifts = redshifts[ mask ]
    fluxes = fluxes[ mask ]
    luminosities = luminosities[ mask ]

    logger.debug( f"lum: {np.min( luminosities )}-{np.max( luminosities )}, redsh: {np.min( redshifts )}-{np.max( redshifts )}, flux: {np.min( fluxes )}-{np.max( fluxes )}" )
    rlf_calculator.calculate_rlf( fluxes, redshifts, luminosities, vmax_method=True )
