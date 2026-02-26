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

logger = get_logger( __name__, logging.DEBUG )

class RLF:
    """
    A class to calculate the radio luminosity function (RLF) of a sample of AGN using the method of Page & Carrera
    2000.
    """

    def __init__(self):
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


    def get_completeness(self, integ_fluxes, completeness_path=None):
        # Read completeness function parameters from file
        #logger.info( f"min flux: {np.min( integ_fluxes )} - max flux: {np.max( integ_fluxes )} - cutoff 0.01" )
        return np.where( integ_fluxes > 1.1e-3, 1, 0 )
    
    def get_completeness_from_coord( self, v: float | np.ndarray, l: float | np.ndarray, cosmo: astropy.cosmology.FLRW, volume_grid: np.ndarray | None = None, redshift_grid: np.ndarray | None = None, completeness_path=None ):
        logger.debug( f'C[s(v,l)]: v={v.shape if isinstance( v, np.ndarray ) else v}, l={l.shape}, s={self.flux_from_coordinate( v, l, cosmo, volume_grid, redshift_grid )}')
        return self.get_completeness( self.flux_from_coordinate( v, l, cosmo, volume_grid, redshift_grid ), completeness_path )
    
    def flux_from_coordinate( self, v: float | np.ndarray, l: float | np.ndarray, cosmo: astropy.cosmology.FLRW, volume_grid: np.ndarray | None = None, redshift_grid: np.ndarray | None = None ):
        """
        Generate luminosities + redshifts -> fluxes. Flux values here are in Jy, luminosities in W/Hz
        
        :param v: Volume(s)
        :type v: float | np.ndarray
        :param l: Luminosity/Luminosities
        :type l: float | np.ndarray
        :param cosmo: FLRW cosmology (e.g. astropy.cosmology.FlatLambdaCDM)
        :type cosmo: astropy.cosmology.FLRW
        :param volume_grid: Volume grid for volume -> redshift interpolation, paired with redshift grid
        :type volume_grid: np.ndarray | None
        :param redshift_grid: Redshift grid for volume -> redshift interpolation, paired with volume grid
        :type redshift_grid: np.ndarray | None
        """

        # generate volume/redshift grid if not provided
        if redshift_grid is None or volume_grid is None:
            redshift_grid = np.linspace( self.z_min, self.z_max, self.n_interp_pts )
            volume_grid = cosmo.comoving_volume( redshift_grid ).to( u.Mpc**3 ).value

        z = np.interp(v, volume_grid, redshift_grid)[ :, np.newaxis ]
        d_l = cosmo.luminosity_distance(z).to(u.m).value
        s = l / (4 * np.pi * d_l) / ( 1e-26 * d_l ) * (1+z)**(1+self.spectral_index)
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
    
    def completeness_simpson_lum_integral( self, v: float, l_mins: np.ndarray, l_maxs: np.ndarray, cosmo: astropy.cosmology.FLRW, volume_grid: np.ndarray | None = None, redshift_grid: np.ndarray | None = None ):
        # integral from l_min to l_max of C[L] d(log10 L)
        # = integral from l_min to l_max of ln[10] C[L] / L dL
        # = ln[10] ( l_max - l_min ) / 6 * ( C[l_min]/l_min + C[l_max]/l_max + 8C[(l_max+l_min)/2]/(l_max+l_min) )

        logger.debug( f'Entering lum simpson integral from {l_mins.shape}-{l_maxs.shape}' )

        c_div_l = lambda l : self.get_completeness_from_coord( v, l, cosmo, volume_grid, redshift_grid ) / l
        return np.log( 10 ) * self.one_dim_simpsons_rule( c_div_l, l_mins, l_maxs, self.n_mc_pts )

    def simpson_integral( self, v_min: float, v_max: float, l_mins: np.ndarray, l_maxs: np.ndarray, cosmo: astropy.cosmology.FLRW, volume_grid: np.ndarray | None = None, redshift_grid: np.ndarray | None = None ):
        logger.debug( f'Entering volume simpson integral from {v_min}-{v_max}' )
        c = lambda v : self.completeness_simpson_lum_integral( v, l_mins, l_maxs, cosmo, volume_grid, redshift_grid )
        return self.one_dim_simpsons_rule( c, v_min, v_max, self.n_mc_pts )

    
    def monte_carlo_integral( self, v_min: float, v_max: float, l_mins: np.ndarray, l_maxs: np.ndarray, cosmo: astropy.cosmology.FLRW, volume_grid: np.ndarray | None = None, redshift_grid: np.ndarray | None = None ):
        """
        Evaluate the Page & Carrera 2000 integral using monte-carlo methods for a given volume bin and set of luminosity bins
        
        :param v_min: Bin volume minimum
        :param v_max: Bin volume maximum
        :param l_mins: Bin luminosity minimums, shape (1, n_lum_bins)
        :param l_maxs: Bin luminosity maximums, shape (1, n_lum_bins)
        :param volume_grid: Grid of volumes for interpolation, pair with redshift_grid
        :param redshift_grid: Grid of redshifts for interpolation, pair with volume_grid
        :param cosmo: FLRW Cosmology (e.g. astropy.cosmology.FlatLambdaCDM)
        """
        # generate volume/redshift grid if not provided
        if redshift_grid is None or volume_grid is None:
            redshift_grid = np.linspace( self.z_min, self.z_max, self.n_interp_pts )
            volume_grid = cosmo.comoving_volume( redshift_grid ).to( u.Mpc**3 ).value

        # -- MONTE CARLO METHOD ---
        # Now generate random points in this bin; so random in volume and in luminosity, and then
        # compare to the completeness function to find the function of RLF in that bin.
        # random_fluxes has shape (self.n_mc_pts, n_lum_bins) for compat w/ np.random.uniform
        random_volumes = np.random.uniform(v_min, v_max, self.n_mc_pts)
        random_luminosities = loguniform.rvs(l_mins[ 0, : ], l_maxs[ 0, : ], size=(self.n_mc_pts, l_mins.shape[ 1 ]))
        random_fluxes = self.flux_from_coordinate( random_volumes[ :, np.newaxis ], random_luminosities, cosmo, volume_grid, redshift_grid )

        # weight each point by Completeness[ flux ] and add to total monte-carlo integral
        # for now, placeholder, assume flux cutoff at 1 mJy
        bin_integrals = np.sum(self.get_completeness(random_fluxes), axis=0) / self.n_mc_pts

        # divide by the luminosity-volume bin area so the result is / MPc^3 / (W/Hz)
        bin_integrals *= ( v_max - v_min ) * ( np.log10( l_maxs[ 0, : ] ) - np.log10( l_mins[ 0, : ] ) )

        return bin_integrals


    def calculate_rlf(self, fluxes, redshifts = None, luminosities = None):
        """
        Calculate and plot the estimated differential RLF for the generated data
        """
        logger.info( "start rlf calculation" )


        cosmo = astropy.cosmology.FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)
        Z_MIN = self.z_min
        Z_MAX = self.z_max
        V_MIN, V_MAX = cosmo.comoving_volume( [Z_MIN, Z_MAX] ).value

        logger.debug( f"volume range: {V_MIN}-{V_MAX}" )

        redshift_grid = np.linspace( Z_MIN, Z_MAX, self.n_interp_pts )
        volume_grid = cosmo.comoving_volume( redshift_grid ).to( u.Mpc**3 ).value
        if redshifts is None:
            # assign each source a comoving volume with a uniform dist s.t. dN/dV = const
            volumes = np.random.uniform( V_MIN, V_MAX, fluxes.shape[ 0 ] )

            # conversion from comoving volume to redshift
            redshifts = np.interp(volumes, volume_grid, redshift_grid)

        if luminosities is None:
            # use the redshift to calculate luminosity distance and luminosity
            luminosity_distances = cosmo.luminosity_distance(redshifts).to(u.m).value
            luminosities = fluxes * ( 1e-26 * luminosity_distances ) * (4 * np.pi * luminosity_distances) / ( 1 + redshifts )**(1+self.spectral_index) # W/Hz

        # clip flux values to those above 0 for log plotting
        redshifts = redshifts[ fluxes > 0 ]
        luminosities = luminosities[ fluxes > 0 ]
        fluxes = fluxes[ fluxes > 0 ]

        # select a redshift-luminosity bin, use monte-carlo to populate the bin,
        # work backwards to find fluxes and weight by completeness function to calculate integral as in Page & Carrera 2000.
        # Also calculate the number of sources in the bin while we're at it
        z_bins = np.arange(Z_MIN, Z_MAX, self.dz)
        l_bins = np.logspace(21, 29, self.lum_bins_count)
        n_lum_bins = len( l_bins ) - 1
        phi_est = np.zeros((len(z_bins), n_lum_bins))

        # Can read previously created numpy files to avoid recomputation
        #if (pth.NP_ARRAY_PARENT / subdir / 'rlf.npy').exists():
        #    phi_est = np.load(pth.NP_ARRAY_PARENT / subdir / 'rlf.npy')
        # iteration so we can index phi_est by phi_est[ i_z, i_l ]
        logger.debug( f"z bins to iterate over: {len(z_bins)}" )
        logger.debug( f"l bins to iterate over: {self.lum_bins_count}" )

        for i_z in range(len(z_bins)):
            z_min = z_bins[i_z]
            z_max = z_min + self.dz

            # find min & max of comoving volume for redshift bin
            v_min, v_max = cosmo.comoving_volume([z_min, z_max]).to( u.Mpc**3 ).value

            # get luminosity bins from offset indices
            # and make them (1,n_lum_bins) arrays for broadcasting with (n_sources,1)
            # luminosity bins are defined by their minimum value
            l_mins = l_bins[ :-1 ][ np.newaxis, : ]
            l_maxs = l_bins[ 1: ][ np.newaxis, : ]

            # now calculate the number of 'real' sources in each bin
            # masks have shape:
            #   redshift_mask: (n_sources, 1)
            #   luminosity_mask: (n_sources, n_lum_bins)
            redshift_mask = (redshifts[ :, np.newaxis ] >= z_min) & (redshifts[ :, np.newaxis ] < z_max)
            luminosity_mask = (luminosities[ :, np.newaxis ] >= l_mins) & (luminosities[ :, np.newaxis ] < l_maxs)

            # shape (n_lum_bins)
            n_sources_in_lum_bins = np.sum( redshift_mask & luminosity_mask, axis=0 )
            logger.debug( f"n_sources_in_lum_bins: {n_sources_in_lum_bins}" )

            #bin_integrals = self.monte_carlo_integral( v_min, v_max, l_mins, l_maxs, cosmo, volume_grid, redshift_grid )
            bin_integrals = self.simpson_integral( v_min, v_max, l_mins, l_maxs, cosmo, volume_grid, redshift_grid )

            logger.debug( f"bin integrals: {bin_integrals}" )

            # if we have a 0 bin integral but N > 0 it must be a monte carlo failure or completeness mismatch
            if np.any( ( bin_integrals == 0 ) & ( n_sources_in_lum_bins > 0 ) ):
                logger.error( f"Monte Carlo failure - {self.n_mc_pts} points insufficient for \
                       {np.sum( ( bin_integrals == 0 ) & ( n_sources_in_lum_bins > 0  ) )}/{bin_integrals.shape[ 0 ]} bins" )
            bin_integrals[ n_sources_in_lum_bins == 0 ] = 1

            # now we have phi_est as given by Page & Carrera 2000
            phi_est[i_z] = n_sources_in_lum_bins / bin_integrals

            logger.info( f'Redshift range {z_bins[i_z]:.2f}-{z_bins[i_z]+self.dz:.2f} complete' )

            #np.save(pth.NP_ARRAY_PARENT / subdir / 'rlf.npy', phi_est)

        # plot the resulting graph
        logger.info( "plotting..." )
        plt.figure()
        for i_z in range(phi_est.shape[0]):
            specific_phi_est = phi_est[i_z]
            #mask = specific_phi_est > 0
            plt.plot( l_bins[ :-1], specific_phi_est, color=colormaps['cool'](i_z / phi_est.shape[0]),
                     marker='o',
                     label=f'{z_bins[i_z]:.2f}<z<{z_bins[i_z]+self.dz:.2f}')
            
        plt.xscale( 'log' )
        plt.xlabel( 'L144 * Hz / W')
        plt.yscale( 'log' )
        plt.ylabel( 'phi * MPc^3 * log10( W / m^2 )' )
        plt.legend()
        plt.savefig(f'rlf.png')
        logger.info( "saved figure" )


if __name__ == "__main__":
    rlf_calculator = RLF()
    catalog = HardcastleCatalogue( resolved_only=False )

    logger.debug( "getting catalog data" )
    redshifts = catalog.get_values( Source.Redshift )
    logger.debug( "   redshifts done" )
    fluxes = catalog.get_values( Source.TotalFlux )
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
    rlf_calculator.calculate_rlf( fluxes, redshifts, luminosities )
