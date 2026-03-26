import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
import matplotlib.pyplot as plt
from hardcastle_catalogue import Source, HardcastleCatalogue
from utils.logging import get_logger
import logging
from tqdm import tqdm
from utils.fitfunctions import sigmoid
from pathlib import Path

# from Hardcastle et al. 2022, https://github.com/mhardcastle/agn-selection/blob/main/plots.py
def ccol(i):
    colours=[[0,0,0],[0,73,73],[0,146,146],[255,109,182],[255,182,219],[73,0,146],[0,109,219],[182,109,255],
             [109,182,255],[182,219,255],[146,0,0],[146,73,0],[219,209,0],[36,255,36],[255,255,109]]
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
        # Start logging
        self.logger = get_logger("RLF", logging.DEBUG)

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
        self.spectral_index = float(lu_config['SPECTRAL_INDEX']) # spectral index to use for the k-correction, typically -0.7 for AGN
        self.z_max = float(lu_config['Z_MAX']) # maximum Z (redshift) to consider in RLF calculation
        self.z_min = float(lu_config['Z_MIN']) # minimum Z (redshift) to consider in RLF calculation
        self.l_max = float(lu_config['L_MAX']) # maximum luminosity to plot, to the power 10 (max lum = 10**L_MAX)
        self.l_min = float(lu_config['L_MIN']) # minimum luminosity to plot, to the power 10 (min lum = 10**L_MIN)

        # initialize cosmology so we can define interpolation grids
        if cosmo is None:
            self.cosmo = astropy.cosmology.FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)
        else:
            self.cosmo = cosmo
        self.v_min, self.v_max = self.cosmo.comoving_volume( [self.z_min, self.z_max] ).to( u.Mpc**3 ).value
        self.logger.debug( f"volume range: {self.v_min}-{self.v_max}" )
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
        self.counts = np.zeros((self.n_z_bins, self.n_lum_bins))

    # ---------- COMPLETENESS ----------

    def get_completeness(self,
                         integ_fluxes : np.ndarray,
                         completeness_path : Path = None,
                         step_completeness : bool = False,
                         threshold : float = 1.1e-3):
        """
        Returns a value for the completeness correction for use in the RLF integral estimation. Can either return a
        fitted sigmoid completeness read from a file, or a step completeness function (i.e., 1 if above a threshold, 0
        otherwise)

        :param integ_fluxes: The array of integrated fluxes to apply completeness corrections to.
        :param completeness_path: The path to the completeness parameters file.
        :param step_completeness: Whether or not to use a step completeness.
        :param threshold: The threshold for the stpe completeness.
        :return: Completeness corrections in the same shape as integ_fluxes
        """
        # Implement a step completeness
        if step_completeness:
            return np.where(integ_fluxes > threshold, 1, 0)

        # Reading completeness parameters from a file
        if completeness_path is None:
            completeness_path = pth.NP_ARRAY_PARENT / 'completeness_args_sigmoid.txt'
        if completeness_path.exists():
            completeness_args = np.loadtxt( completeness_path )
            return sigmoid( integ_fluxes * 1000, *completeness_args )
        else:
            self.logger.warning( f'Completeness path {completeness_path} does not exist, returning step completeness with threshold {threshold}' )
            return np.where(integ_fluxes > threshold, 1, 0)

    def get_completeness_from_coord( self, v: float | np.ndarray, l: float | np.ndarray, *args ):
        """
        Functions as a proxy for the get_completeness functon, allowing it to be ran in volume-luminosity space without
        requiring at-use computation of the integrated flux.

        :param v: The comoving volume at the coordinate in volume-luminosity space.
        :param l: The luminosity at the coordinate in volume-luminosity space.
        :return: Completeness corrections at the specific coordinate in volume-luminosity space.
        """
        #logger.debug( f'C[s(v,l)]: v={v.shape if isinstance( v, np.ndarray ) else v}, l={l.shape}, s={self.flux_from_coordinate( v, l, cosmo, zvparams )}')
        return self.get_completeness( self.flux_from_coordinate( v, l ), *args )
    
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

        # Find the luminosity distance & convert into flux with a k-correction
        d_l = self.cosmo.luminosity_distance(z).to(u.m).value
        s = 1e26 * l / (4 * np.pi * d_l**2) * self.k_corr_factor( z )
        return s

    # ---------- INTEGRALS ----------

    def one_dim_simpsons_rule( self, f, a, b, n ):
        """
        Perform integration using simpson's rule on a 1 dimensional function as integral from a to b of f( x ) dx. For
        more information, see https://en.wikipedia.org/wiki/Simpson%27s_rule

        :param f: Function to integrate
        :param a: Lower limit
        :param b: Upper limit
        :param n: Order of simpson's rule, must be divisible by 2
        """
        self.logger.debug( f'Simpsons rule order {n}' )

        h = ( b - a ) / n
        i = np.arange( 0, n+1, 1, dtype=int )


        # a and b may be passed as (1,n) arrays, so this allows for broadcasting
        if isinstance( a, np.ndarray ) and isinstance( a, np.ndarray ):
            i = i[ :, np.newaxis ]

        self.logger.debug( f'h={h.shape if isinstance( h, np.ndarray ) else h}, i={i.shape}' )

        x = a + i * h
        y = f( x )
        self.logger.debug( f'x={x.shape}, y={y.shape}' )
        return 1/3 * h * ( y[ 0 ] 
               + 4 * np.sum( y[ 1:-1:2 ], axis=0 )
               + 2 * np.sum( y[ 2:-1:2 ], axis=0 )
               + y[ -1 ] )
    
    def completeness_simpson_lum_integral( self, v: float, l_mins: np.ndarray, l_maxs: np.ndarray ):
        # integral from l_min to l_max of C[L] d(log10 L)
        # = integral from l_min to l_max of ln[10] C[L] / L dL
        # = ln[10] ( l_max - l_min ) / 6 * ( C[l_min]/l_min + C[l_max]/l_max + 8C[(l_max+l_min)/2]/(l_max+l_min) )

        self.logger.debug( f'Entering lum simpson integral from {l_mins.shape}-{l_maxs.shape}' )

        c_div_l = lambda l : self.get_completeness_from_coord( v, l ) / l
        return np.log( 10 ) * self.one_dim_simpsons_rule( c_div_l, l_mins, l_maxs, self.n_mc_pts )

    def simpson_integral( self, v_min: float, v_max: float, l_mins: np.ndarray, l_maxs: np.ndarray ):
        self.logger.debug( f'Entering volume simpson integral from {v_min}-{v_max}' )
        c = lambda v : self.completeness_simpson_lum_integral( v, l_mins, l_maxs )
        return self.one_dim_simpsons_rule( c, v_min, v_max, self.n_mc_pts )

    def monte_carlo_integral( self, v_min: float, v_max: float, l_min: float | np.ndarray, l_max: float | np.ndarray, lum: np.ndarray | float | None = None ):
        """
        Evaluate the Page & Carrera 2000 integral using monte-carlo methods for a given volume bin and set of luminosity bins

        :param v_min: Bin volume minimum
        :param v_max: Bin volume maximum
        :param l_mins: Bin luminosity minimums, shape (1, n_lum_bins) or float. To use more than one lum bin requires lum = None
        :param l_maxs: Bin luminosity maximums, shape (1, n_lum_bins) or float. To use more than one lum bin requires lum = None
        :param lum: enforced luminosity to pass to completeness function, for use in the 1/Vmax method. Float for one integral of that luminosity, or array of shape (1, n_integrals) or (n_integrals), or none to use uniform logluminosities (Page & Carrera 2000 method).
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
            elif lum.ndim == 2:
                lums = lum
            else: raise RuntimeError( f'lum arg of ndims {lum.ndim} invalid, must be at most 2' )
       
        # lums now definitely has shape (self.n_mc_pts, n_integrals)

        # -- MONTE CARLO METHOD ---
        # Now generate random points in volume space and either use given luminosities or random luminosities within the bin(s)
        # evaluate the completeness at each point to determine the integral C[S[v,L]] dV dlog10L from v=(v_min, v_max) and l=(l_min, l_max)
        # random_volumes has shape (self.n_mc_pts, 1) while lums has shape (1, n_integrals)
        random_volumes = np.random.uniform(v_min, v_max, self.n_mc_pts)[ :, np.newaxis ]
        bin_integrals = np.sum( self.get_completeness_from_coord( random_volumes, lums ), axis=0) / self.n_mc_pts

        # divide by the log luminosity-volume bin area so the result is / MPc^3 / log10(W/Hz)
        if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
            bin_integrals *= ( v_max - v_min ) * ( np.log10( l_max[ 0, : ] / l_min[ 0, : ] ) )
        else:
            bin_integrals *= ( v_max - v_min ) * ( np.log10( l_max / l_min ) )

        # if n_integrals is 1, cut out the last axis so we just return a scalar
        if isinstance( bin_integrals, np.ndarray ) and ( bin_integrals.shape[ -1 ] == 1 ):
            bin_integrals = bin_integrals[ 0 ]

        if not isinstance( bin_integrals, np.ndarray ) and bin_integrals == 0:
            self.logger.info( 'bin_integral is 0 when it probably shouldn\'t be' )
            self.logger.debug( f'lums={lums}, shape={lums.shape}, volumes={random_volumes}, shape={random_volumes.shape}' )
        elif np.any( bin_integrals == 0 ):
            self.logger.info( f'{bin_integrals[ bin_integrals == 0 ].shape[ 0 ]} bin integrals are 0 when they probably shouldn\'t be' )
            self.logger.debug( f'bin_integrals={( bin_integrals )}' )
            self.logger.debug( f'lums={lums}, shape={lums.shape}, min_vol={np.min( random_volumes )}, min_z={z_from_v( np.min( random_volumes ), *self.zvparams )}, max_fluxes={self.flux_from_coordinate( np.min( random_volumes ), lums )}' )

        return bin_integrals

    def k_corr_factor( self, redshift, mag_space: bool = False, spectral_index = None ):
        """
        Returns the k-correction factor for one or more objects at given redshifts

        mag_space: bool = False
        - whether or not to give the k correction in magnitude space (-2.5 * log10( k_corr_lum_space )), default lum space
        spectral_index = None
        - override for spectral index to use instead of self.spectral_index, broadcastable with redshift
        """
        if spectral_index is None:
            spectral_index = self.spectral_index
        k_corr_lum_space = ( 1 + redshift ) ** ( 1 + spectral_index )
        if not mag_space:
            return k_corr_lum_space
        else:
            return -2.5 * np.log10( k_corr_lum_space )

    # ---------- RADIO LUMINOSITY FUNCTION ----------
    def calculate_rlf( self,
                            fluxes,
                            redshifts = None, 
                            luminosities = None,
                            debug_flux_lum_relation: bool = False,
                            plot_rlf: bool = True,
                            vmax_method: bool = False ):
        """
        Calculate the Radio Luminosity Function, either using the Page & Carrera 2000 method or
        the traditional 1/Vmax method
        """
        self.logger.info( "start 1/Vmax rlf calculation" )

        # reset phi in case it's not our first time calling this
        self.phi = np.zeros( (self.n_z_bins, self.n_lum_bins) )
        self.counts = np.zeros( (self.n_z_bins, self.n_lum_bins) )

        if redshifts is None:
            # assign each source a comoving volume with a uniform dist s.t. dN/dV = const
            volumes = np.random.uniform( self.v_min, self.v_max, fluxes.shape[ 0 ] )

            # conversion from comoving volume to redshift
            redshifts = z_from_v( volumes, *self.zvparams )

        # use the redshift to calculate luminosity distance and luminosity
        # because of errors on the margin, disregard passed luminosity
        luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
        luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 / self.k_corr_factor( redshifts ) # W/Hz

        if debug_flux_lum_relation:
            #ensure luminosities and redshifts are consistent with total flux
            # it's a lot easier to tell by visual inspection so save a scatterplot to debugdiff.png
            luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
            flux_luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 / self.k_corr_factor( redshifts ) # W/Hz
            self.logger.info( 'saving scatterplot to compare luminosity from flux to luminosity from catalog...' )
            residuals = flux_luminosities / luminosities
            plt.scatter( flux_luminosities, residuals, s=0.001 )
            #plt.yscale( 'log' )
            epsilon = 1e-10
            plt.ylim( 1-epsilon, 1+epsilon )
            plt.xscale( 'log' )
            plt.title( 'Luminosity from flux vs catalog' )
            plt.xlabel( 'luminosity from flux' )
            plt.ylabel( 'luminosity from catalog / luminosity from flux' )
            plt.grid()
            plt.savefig( 'debugdiff.png' )
            plt.figure()
            self.logger.info( 'saved debug luminosity diff figure to debugdiff.png' )

        # clip flux values to those above 0 for log plotting
        redshifts = redshifts[ fluxes > 0 ]
        luminosities = luminosities[ fluxes > 0 ]
        fluxes = fluxes[ fluxes > 0 ]

        if vmax_method:
            for i_z in range( self.n_z_bins ):
                z_min, z_max = self.z_bins[ i_z ], self.z_bins[ i_z+1 ]
                v_min, v_max = self.cosmo.comoving_volume([z_min, z_max]).to( u.Mpc**3 ).value
                redshift_mask = (redshifts >= z_min) & (redshifts < z_max)
                self.logger.debug( f'{luminosities[ redshift_mask ].shape[ 0 ]} sources in z: {z_min}-{z_max}' )

                for i_l in range( self.n_lum_bins ):
                    l_min, l_max = self.l_bins[ i_l ], self.l_bins[ i_l+1 ]

                    luminosity_mask = (luminosities >= l_min) & (luminosities < l_max)

                    luminosities_in_bin = luminosities[ redshift_mask & luminosity_mask ]

                    # for n=0, the RLF should be 0 regardless, and also it breaks the code so just ignore it
                    if not luminosities_in_bin.size:
                        self.logger.debug( f'no sources in z: {z_min}-{z_max}, l={l_min}-{l_max}' )
                        continue
                    self.logger.debug( f'{luminosities_in_bin.shape[ 0 ]} sources in z: {z_min}-{z_max}, l={l_min}-{l_max}' )

                    Vmaxs = self.monte_carlo_integral( v_min, v_max, l_min, l_max, luminosities_in_bin )

                    self.phi[ i_z, i_l ] = np.sum( 1.0 / Vmaxs ) #log bin size included in Vmaxs from monte_carlo_integral
                    self.counts[ i_z, i_l ] = luminosities_in_bin.shape[ 0 ]
                self.logger.info( f'Redshift range {z_min:.2f}-{z_max:.2f} complete' )

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
                self.logger.debug( f"n_sources_in_lum_bins: {n_sources_in_lum_bins}" )

                bin_integrals = self.monte_carlo_integral( v_min, v_max, l_mins, l_maxs )
                #bin_integrals = self.simpson_integral( v_min, v_max, l_mins, l_maxs )

                self.logger.debug( f"bin integrals: {bin_integrals}" )

                # if we have a 0 bin integral but N > 0 it must be a monte carlo failure or completeness mismatch
                if np.any( ( bin_integrals == 0 ) & ( n_sources_in_lum_bins > 0 ) ):
                    self.logger.error( f"Monte Carlo failure - {self.n_mc_pts} points insufficient for \
                        {np.sum( ( bin_integrals == 0 ) & ( n_sources_in_lum_bins > 0  ) )}/{bin_integrals.shape[ 0 ]} bins" )
                bin_integrals[ n_sources_in_lum_bins == 0 ] = 1

                # now we have phi_est as given by Page & Carrera 2000
                self.phi[i_z] = n_sources_in_lum_bins / bin_integrals
                self.counts[ i_z ] = n_sources_in_lum_bins

                self.logger.info( f'Redshift range {z_min:.2f}-{z_max:.2f} complete' )

        # sky coverage completeness factor
        self.phi /= 5700 / 41253 # 5700 lotss dr2 area from hardcastle et al. 2023, 41253 deg^2 is solid angle of a sphere

        if plot_rlf:
            # plot the resulting graph
            title = ''
            output = ''
            if vmax_method:
                title = f'1/Vmax RLF - {self.n_mc_pts} pts per bin'
                output = 'rlf_vmax.png'
            else:
                title = f'Page & Carrera RLF - {self.n_mc_pts} pts per bin'
                output = 'rlf_page_and_carrera.png'
            self.plot_rlf( title, colors, output=output )

    def plot_rlf( self, 
                  title: str,
                  colors: list,
                  ax = None,
                  ylim: tuple = [ 1e-9, 3e-4 ],
                  xlim: tuple = [ 1e21, 1e29 ],
                  output: str = 'rlf.png' ):
        self_contained = ax is None
        if self_contained:
            self.logger.info( "plotting self contained rlf" )
            fig, ax = plt.subplots( figsize=(10,10) )

        bin_centres = ( self.l_bins[ :-1 ] + self.l_bins[ 1: ] ) / 2
        for i_z in range( self.phi.shape[0] ):
            specific_phi = self.phi[i_z]
            specific_counts = self.counts[i_z]
            mask = specific_counts >= 7
            ax.plot( bin_centres[ mask ], specific_phi[ mask ], color=colors[ i_z ],
                     marker='o',
                     label=f'{self.z_bins[ i_z ]:.2f}<z<{self.z_bins[ i_z + 1 ]:.2f}')
            ax.errorbar( bin_centres[ mask ], specific_phi[ mask ], yerr=specific_phi[ mask ] / np.sqrt( specific_counts[ mask ] ), color=colors[ i_z ], fmt='none' )
        ax.set_title( title )
        ax.set_xscale( 'log' )
        ax.set_xlabel( 'L144 * Hz / W' )
        ax.set_yscale( 'log' )
        ax.set_ylabel( 'phi * MPc^3 * log10( W / m^2 )' )
        ax.set_xlim( xlim )
        ax.set_ylim( ylim )
        ax.legend()
        if self_contained:
            plt.savefig( output )
            self.logger.info( f"saved figure to {output}" )

def mag_to_flux_w3( mag, default_spectral_index = -1 ):
    # https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html
    f_corr_table = [ 1.1344, 1.0088, 0.9393, 0.9169, 0.9373, 1.0000, 1.1081, 1.2687 ]
    spectral_index_table = [ 3, 2, 1, 0, -1, -2, -3, -4 ]
    f_corr = np.interp( default_spectral_index, spectral_index_table, f_corr_table )
    Fstar_v0 = 29.045
    return ( Fstar_v0 / f_corr ) * 10**(-mag / 2.5)

def mag_to_flux_w2( mag, default_spectral_index = -1 ):
    # https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html
    f_corr_table = [ 1.0206, 1.0066, 0.9976, 0.9935, 0.9943, 1.0000, 1.0107, 1.0265 ]
    spectral_index_table = [ 3, 2, 1, 0, -1, -2, -3, -4 ]
    f_corr = np.interp( default_spectral_index, spectral_index_table, f_corr_table )
    Fstar_v0 = 170.663
    return ( Fstar_v0 / f_corr ) * 10**(-mag / 2.5)

if __name__ == "__main__":
    rlf_calculator = RLF()
    vmax_rlf = RLF()
    catalog = HardcastleCatalogue( resolved_only=True )

    rlf_calculator.logger.debug( "getting catalog data" )
    redshifts = catalog.get_values( Source.Redshift )
    fluxes = catalog.get_values( Source.TotalFlux ) / 1000
    luminosities = catalog.get_values( Source.Luminosity )
    wise_3_mag = catalog.get_values( Source.WISE3Mag )
    wise_2_mag = catalog.get_values( Source.WISE2Mag )
    rlf_calculator.logger.debug( "done" )

    mask = ( fluxes > 0 ) & ( wise_3_mag > 0 ) & ( wise_2_mag > 0 ) & ~np.isnan( fluxes ) & ~np.isnan( wise_3_mag ) & ~np.isnan( wise_2_mag ) & ~np.isnan( redshifts ) & ~np.isnan( luminosities ) & ( fluxes > 1.1e-3 ) & ( redshifts > 0.01 )
    redshifts = redshifts[ mask ]
    luminosities = luminosities[ mask ]
    wise_3_mag = wise_3_mag[ mask ]
    wise_2_mag = wise_2_mag[ mask ]
    fluxes = fluxes[ mask ]

    #logger.debug( f'wise_3_mag: mean={np.average( wise_3_mag )}, std={np.std( wise_3_mag )}, max={np.max( wise_3_mag )}, min={np.min( wise_3_mag ) }, count={wise_3_mag.shape[ 0 ]}' )
    #logger.debug( f'wise_2_mag: mean={np.average( wise_2_mag )}, std={np.std( wise_2_mag )}, max={np.max( wise_2_mag )}, min={np.min( wise_2_mag ) }, count={wise_2_mag.shape[ 0 ]}' )

    # use wise bands 2/3 to calculate spectral indices for the k-correction
    wise_3_flux = mag_to_flux_w3( wise_3_mag )
    wise_2_flux = mag_to_flux_w2( wise_2_mag )
    wise_3_freq = 3e8 / 12e-6
    wise_2_freq = 3e8 / 4.6e-6
    spectral_inds = -np.log( wise_3_flux / wise_2_flux ) / np.log( wise_3_freq / wise_2_freq )
    #logger.debug( f'spectral_inds: mean={np.average( spectral_inds )}, std={np.std( spectral_inds )}, max={np.max( spectral_inds )}, min={np.min( spectral_inds ) }, count={spectral_inds.shape[ 0 ]}' )

    wise_3_absmag = wise_3_mag - 5 * ( np.log10( rlf_calculator.cosmo.luminosity_distance( redshifts ).to(u.parsec).value ) - 1 ) - rlf_calculator.k_corr_factor( redshifts, mag_space=True, spectral_index=spectral_inds )

    # plot the relationship between L144 and Abs W3 (Fig. 2, H25)
    rqq_xpt = -27.923076923076923 #mag
    rqq_ypt = 25.563106796116504 #log10( lum )
    if True:
        wise_3_linspace = np.linspace( -34, -18, 1000 )
        wise_3_linspace_below_27 = np.linspace( -34, -27, 1000 )
        sfg_lum_cutoff = 10**( 14 - wise_3_linspace / 2.5 )
        rqq_lum_cutoff = 10**( -( wise_3_linspace_below_27 - rqq_xpt ) / 3.4844629455909923 + rqq_ypt )
        plt.figure( figsize=(8,8) )
        plt.hexbin( wise_3_absmag[ wise_3_absmag < -19 ], luminosities[ wise_3_absmag < -19 ], gridsize=50, yscale='log' )
        plt.plot( wise_3_linspace, sfg_lum_cutoff, color='r' )
        plt.plot( wise_3_linspace_below_27, rqq_lum_cutoff, color='m' )
        plt.xlabel( 'wise 3 absolute magnitude' )
        plt.ylabel( 'L144' )
        plt.title( 'L144 vs W3 AbsMag Relation' )
        plt.yscale( 'log' )
        plt.xlim( -18, -34 )
        plt.ylim( 4e20, 1e29 )
        plt.savefig( 'lum_vs_w3.png' )
        plt.show()
        rlf_calculator.logger.debug( "saved lum_vs_w3.png" )


    sfg_mask = ( luminosities < 10**( 14 - wise_3_absmag / 2.5 ) ) & ( luminosities < 10**(24.8) )
    rqq_mask = ( luminosities < 10**( -( wise_3_absmag - rqq_xpt ) / 3.4844629455909923 + rqq_ypt ) ) & ( wise_3_absmag < -27 )
    agn_mask = ~sfg_mask & ~rqq_mask

    rlf_calculator.logger.info( f'# agn: {redshifts[ agn_mask ].shape[ 0 ]} - # sfg: {redshifts[ sfg_mask ].shape[ 0 ]} - # rqq: {redshifts[ rqq_mask ].shape[ 0 ]} - total: {redshifts.shape[ 0 ]}' )

    redshifts = redshifts[ agn_mask ]
    fluxes = fluxes[ agn_mask ]
    luminosities = luminosities[ agn_mask ]

    fig, axes = plt.subplots( ncols=2, figsize=(20, 10) )
    ax_vmax, ax_pnc = axes

    rlf_calculator.logger.debug( f"lum: {np.min( luminosities )}-{np.max( luminosities )}, redsh: {np.min( redshifts )}-{np.max( redshifts )}, flux: {np.min( fluxes )}-{np.max( fluxes )}" )
    rlf_calculator.calculate_rlf( fluxes, redshifts, luminosities, False, plot_rlf=False, vmax_method=False )
    vmax_rlf.calculate_rlf( fluxes, redshifts, luminosities, False, plot_rlf=False, vmax_method=True )

    rlf_calculator.plot_rlf( f'Page & Carrera RLF - {rlf_calculator.n_mc_pts} pts per bin', colors, ax_pnc )
    vmax_rlf.plot_rlf( f'1/Vmax RLF - {vmax_rlf.n_mc_pts} pts per bin', colors, ax_vmax )

    plt.savefig( 'rlfs.png' )
    plt.show()

    rlf_calculator.logger.info( 'done' )
    
