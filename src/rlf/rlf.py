import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
import astropy.stats
import matplotlib.pyplot as plt
from hardcastle_catalogue import Source, HardcastleCatalogue
from utils.logging import get_logger
import logging
from tqdm import tqdm
from utils.functions import sigmoid, mag_to_flux_w2, mag_to_flux_w3, k_corr_factor
from pathlib import Path
from scipy.optimize import curve_fit
import utils.functions as functions

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

shimwell_data = np.array( [
    [ 0.20,   0.00000 ],
    [ 0.22,   0.015625 ],
    [ 0.24,   0.015625 ],
    [ 0.27,   0.03125 ],
    [ 0.30,   0.06250 ],
    [ 0.34,   0.12500 ],
    [ 0.38,   0.18750 ],
    [ 0.42,   0.28125 ],
    [ 0.46,   0.34375 ],
    [ 0.52,   0.46875 ],
    [ 0.58,   0.53125 ],
    [ 0.64,   0.62500 ],
    [ 0.72,   0.71875 ],
    [ 0.80,   0.78125 ],
    [ 0.88,   0.81250 ],
    [ 0.98,   0.87500 ],
    [ 1.10,   0.90625 ],
    [ 1.19,   0.96875 ],
    [ 1.25,   1.00000 ],
] ).transpose()


class RLF:
    """
    A class to calculate the radio luminosity function (RLF) of a sample of AGN using the method of Page & Carrera
    2000.
    """

    def __init__(self, fluxes, redshifts, luminosities, resolved, cosmo, bias: float = 0, flux_cut_jy: float = 1.1e-3, debug_flux_lum_relation: bool = False, vmax_method: bool = False, use_shimwell: bool = True, completeness_path: str = None):
        # Start logging
        self.logger = get_logger("RLF", logging.DEBUG)

        # init parameters
        self.debug_flux_lum_relation = debug_flux_lum_relation
        self.vmax_method = vmax_method
        self.bias = bias
        self.flux_cut_jy = flux_cut_jy
        self.fluxes = fluxes
        self.redshifts = redshifts
        self.luminosities = luminosities
        self.resolved = resolved
        self.use_shimwell = use_shimwell

        if completeness_path is None:
            completeness_path = pth.NP_ARRAY_PARENT / 'completeness_args_sigmoid.txt'
        if completeness_path.exists():
            self.completeness_args = np.loadtxt( completeness_path )
        else:
            self.completeness_args = None
            self.logger.error( f'Could not find completeness args at path {completeness_path}' )

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(pth.PROGRAM_CONFIG)
        default_config = config['DEFAULT']

        # RLF parameters
        self.dz = float(default_config['dz']) # redshift bin width
        self.lum_bins_count = int(default_config['LUM_BINS']) # number of luminosity bins between min and max luminosity
        self.n_interp_pts = int(default_config['N_INTERP_PTS']) # number of points to use in interpolation approximation of
        self.n_mc_pts = int(default_config['N_MC_PTS']) # number of points to use in the monte-carlo integral for each redshift-luminosity bin
        self.spectral_index = float(default_config['SPECTRAL_INDEX']) # spectral index to use for the k-correction, typically -0.7 for AGN
        self.z_max = float(default_config['Z_MAX']) # maximum Z (redshift) to consider in RLF calculation
        self.z_min = float(default_config['Z_MIN']) # minimum Z (redshift) to consider in RLF calculation
        self.l_max = float(default_config['L_MAX']) # maximum luminosity to plot, to the power 10 (max lum = 10**L_MAX)
        self.l_min = float(default_config['L_MIN']) # minimum luminosity to plot, to the power 10 (min lum = 10**L_MIN)

        # initialize cosmology so we can define interpolation grids
        self.cosmo = cosmo
        self.v_min, self.v_max = self.cosmo.comoving_volume( [self.z_min, self.z_max] ).to( u.Mpc**3 ).value
        self.logger.debug( f"volume range: {self.v_min}-{self.v_max}" )
        self.redshift_grid = np.geomspace( self.z_min, self.z_max, self.n_interp_pts )
        self.volume_grid = self.cosmo.comoving_volume( self.redshift_grid ).to( u.Mpc**3 ).value

        # define RLF z/l bins
        if default_config.getboolean( 'HARDCASTLE_Z_BINS' ):
            self.z_bins = np.array( [ 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2 ] ) #hardcastle bins
        elif default_config.getboolean( 'DEJONG_Z_BINS' ):
            self.z_bins = np.array( [ 0.01, 0.3 ] )
        else:
            self.z_bins = np.arange( self.z_min, self.z_max, self.dz )

        self.l_bins = np.logspace( self.l_min, self.l_max, self.lum_bins_count)
        self.n_lum_bins = self.lum_bins_count - 1
        self.n_z_bins = self.z_bins.shape[ 0 ] - 1

        # params for interp
        self.zvparams = [ self.volume_grid, self.redshift_grid ]

        # init rlf values as zero
        self.phi = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.phi_err = np.zeros( (self.n_z_bins, self.n_lum_bins) ) 
        self.counts = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.rlf_fit_params = np.zeros( (self.n_z_bins, 4, 2) ) 

    # ---------- COMPLETENESS ----------

    def get_completeness(self,
                         integ_fluxes : np.ndarray,
                         resolved : np.ndarray):
        """
        Returns a value for the completeness correction for use in the RLF integral estimation. Can either return a
        fitted sigmoid completeness read from a file, or a step completeness function (i.e., 1 if above a threshold, 0
        otherwise)

        :param integ_fluxes: The array of integrated fluxes to apply completeness corrections to.
        :param completeness_path: The path to the completeness parameters file.
        :param step_completeness: Whether or not to use a step completeness.
        :return: Completeness corrections in the same shape as integ_fluxes
        """
        bias = self.bias

        sigmoid_completeness = sigmoid( ( integ_fluxes - bias ) * 1000, *self.completeness_args )
        resolved_completeness = np.where( integ_fluxes > self.flux_cut_jy, sigmoid_completeness, 0 )
        if self.use_shimwell:
            shimwell_completeness = np.interp( integ_fluxes, shimwell_data[ 0 ] / 1000, shimwell_data[ 1 ] )
            unresolved_completeness = np.where( integ_fluxes > self.flux_cut_jy, shimwell_completeness, 0 )
        else:
            unresolved_completeness = np.where( integ_fluxes > self.flux_cut_jy, 1, 0 )

        return np.where( resolved, resolved_completeness, unresolved_completeness )

    def get_completeness_from_coord( self, v: float | np.ndarray, l: float | np.ndarray, resolved : np.ndarray | bool ):
        """
        Functions as a proxy for the get_completeness functon, allowing it to be ran in volume-luminosity space without
        requiring at-use computation of the integrated flux.

        :param v: The comoving volume at the coordinate in volume-luminosity space.
        :param l: The luminosity at the coordinate in volume-luminosity space.
        :return: Completeness corrections at the specific coordinate in volume-luminosity space.
        """
        #logger.debug( f'C[s(v,l)]: v={v.shape if isinstance( v, np.ndarray ) else v}, l={l.shape}, s={self.flux_from_coordinate( v, l, cosmo, zvparams )}')
        return self.get_completeness( self.flux_from_coordinate( v, l ), resolved=resolved )
    
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
        s = 1e26 * l / (4 * np.pi * d_l**2) * k_corr_factor( z, spectral_index = self.spectral_index )
        return s

    # ---------- INTEGRALS ----------

    #def one_dim_simpsons_rule( self, f, a, b, n ):
    #    """
    #    Perform integration using simpson's rule on a 1 dimensional function as integral from a to b of f( x ) dx. For
    #    more information, see https://en.wikipedia.org/wiki/Simpson%27s_rule

    #    :param f: Function to integrate
    #    :param a: Lower limit
    #    :param b: Upper limit
    #    :param n: Order of simpson's rule, must be divisible by 2
    #    """
    #    self.logger.debug( f'Simpsons rule order {n}' )

    #    h = ( b - a ) / n
    #    i = np.arange( 0, n+1, 1, dtype=int )


    #    # a and b may be passed as (1,n) arrays, so this allows for broadcasting
    #    if isinstance( a, np.ndarray ) and isinstance( a, np.ndarray ):
    #        i = i[ :, np.newaxis ]

    #    self.logger.debug( f'h={h.shape if isinstance( h, np.ndarray ) else h}, i={i.shape}' )

    #    x = a + i * h
    #    y = f( x )
    #    self.logger.debug( f'x={x.shape}, y={y.shape}' )
    #    return 1/3 * h * ( y[ 0 ] 
    #           + 4 * np.sum( y[ 1:-1:2 ], axis=0 )
    #           + 2 * np.sum( y[ 2:-1:2 ], axis=0 )
    #           + y[ -1 ] )
    
    #def completeness_simpson_lum_integral( self, v: float, l_mins: np.ndarray, l_maxs: np.ndarray, step_completeness: bool = False  ):
        # integral from l_min to l_max of C[L] d(log10 L)
        # = integral from l_min to l_max of ln[10] C[L] / L dL
        # = ln[10] ( l_max - l_min ) / 6 * ( C[l_min]/l_min + C[l_max]/l_max + 8C[(l_max+l_min)/2]/(l_max+l_min) )

    #    self.logger.debug( f'Entering lum simpson integral from {l_mins.shape}-{l_maxs.shape}' )

    #    c_div_l = lambda l : self.get_completeness_from_coord( v, l, step_completeness=step_completeness ) / l
    #    return np.log( 10 ) * self.one_dim_simpsons_rule( c_div_l, l_mins, l_maxs, self.n_mc_pts )

    #def simpson_integral( self, v_min: float, v_max: float, l_mins: np.ndarray, l_maxs: np.ndarray ):
    #    self.logger.debug( f'Entering volume simpson integral from {v_min}-{v_max}' )
    #    c = lambda v : self.completeness_simpson_lum_integral( v, l_mins, l_maxs )
    #    return self.one_dim_simpsons_rule( c, v_min, v_max, self.n_mc_pts )

    def monte_carlo_integral( self, v_min: float, v_max: float, l_min: float | np.ndarray, l_max: float | np.ndarray, resolved: bool, lum: np.ndarray | float | None = None, vmax_method: bool = False ):
        """
        Evaluate the Page & Carrera 2000 integral using monte-carlo methods for a given volume bin and set of luminosity bins

        :param v_min: Bin volume minimum
        :param v_max: Bin volume maximum
        :param l_mins: Bin luminosity minimums, shape (1, n_lum_bins) or float. To use more than one lum bin requires lum = None
        :param l_maxs: Bin luminosity maximums, shape (1, n_lum_bins) or float. To use more than one lum bin requires lum = None
        :param lum: enforced luminosity to pass to completeness function, for use in the 1/Vmax method. Float for one integral of that luminosity, or array of shape (1, n_integrals) or (n_integrals), or none to use uniform logluminosities (Page & Carrera 2000 method).
        """
        mc_pts = self.n_mc_pts // 10 if vmax_method else self.n_mc_pts
        if lum is None:
            if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
                lums = 10**np.random.uniform(np.log10(l_min[ 0, : ]), np.log10(l_max[ 0, : ]), size=(mc_pts, l_min.shape[ 1 ]))
            else:
                lums = 10**np.random.uniform(np.log10(l_min), np.log10(l_max), size=(mc_pts, l_min.shape[ 1 ]))

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
        random_volumes = np.random.uniform(v_min, v_max, mc_pts)[ :, np.newaxis ]
        bin_integrals = np.sum( self.get_completeness_from_coord( random_volumes, lums, resolved=resolved ), axis=0) / mc_pts

        # divide by the log luminosity-volume bin area so the result is / MPc^3 / log10(W/Hz)
        if isinstance( l_min, np.ndarray ) and isinstance( l_max, np.ndarray ):
            bin_integrals *= ( v_max - v_min ) * ( np.log10( l_max[ 0, : ] / l_min[ 0, : ] ) )
        else:
            bin_integrals *= ( v_max - v_min ) * ( np.log10( l_max / l_min ) )

        # if n_integrals is 1, cut out the last axis so we just return a scalar
        if isinstance( bin_integrals, np.ndarray ) and ( bin_integrals.shape[ -1 ] == 1 ):
            bin_integrals = bin_integrals[ 0 ]

        return bin_integrals


    # ---------- RADIO LUMINOSITY FUNCTION ----------
    def calculate_rlf( self, plot_rlf: bool = True):
        """
        Calculate the Radio Luminosity Function, either using the Page & Carrera 2000 method or
        the traditional 1/Vmax method
        """
        fluxes = self.fluxes
        luminosities = self.luminosities
        redshifts = self.redshifts
        resolved = self.resolved
        vmax_method = self.vmax_method
        self.logger.info( "start " + ( '1/Va' if vmax_method else 'P&C2000' ) + " rlf calculation" )

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
        luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 / k_corr_factor( redshifts, spectral_index=self.spectral_index ) # W/Hz

        if self.debug_flux_lum_relation:
            #ensure luminosities and redshifts are consistent with total flux
            # it's a lot easier to tell by visual inspection so save a scatterplot to debugdiff.png
            luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
            flux_luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 / k_corr_factor( redshifts, spectral_index=self.spectral_index ) # W/Hz
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
        resolved = resolved[ fluxes > 0 ]
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
                    resolved_in_bin = resolved[ redshift_mask & luminosity_mask ]

                    # for n=0, the RLF should be 0 regardless, and also it breaks the code so just ignore it
                    if not luminosities_in_bin.size:
                        self.logger.debug( f'no sources in z: {z_min}-{z_max}, l={l_min}-{l_max}' )
                        continue
                    self.logger.debug( f'{luminosities_in_bin.shape[ 0 ]} sources in z: {z_min}-{z_max}, l={l_min}-{l_max}' )

                    Vmaxs_resolved = self.monte_carlo_integral( v_min, v_max, l_min, l_max, resolved, luminosities_in_bin, vmax_method=True )
                    Vmaxs_unresolved = self.monte_carlo_integral( v_min, v_max, l_min, l_max, resolved, luminosities_in_bin, vmax_method=True )
                    Vmaxs = np.where( resolved_in_bin, Vmaxs_resolved, Vmaxs_unresolved )

                    self.phi[ i_z, i_l ] = np.sum( 1.0 / Vmaxs ) #log bin size included in Vmaxs from monte_carlo_integral
                    self.counts[ i_z, i_l ] = luminosities_in_bin.shape[ 0 ]

                    # get errors from Page & Carrera 2000
                    self.phi_err[ i_z, i_l ] = np.sqrt( np.sum( 1.0 / Vmaxs**2 ) )

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
                n_resolved_in_lum_bins = np.sum( redshift_mask & luminosity_mask & resolved[ :, np.newaxis ], axis=0 )
                n_unresolved_in_lum_bins = np.sum( redshift_mask & luminosity_mask & ~resolved[ :, np.newaxis ], axis=0 )
                self.logger.debug( f"n_sources_in_lum_bins: {n_sources_in_lum_bins}" )
                self.logger.debug( f"n_resolved_in_lum_bins: {n_resolved_in_lum_bins}" )
                self.logger.debug( f"n_unresolved_in_lum_bins: {n_unresolved_in_lum_bins}" )


                bin_integrals_resolved = self.monte_carlo_integral( v_min, v_max, l_mins, l_maxs, resolved=True )
                bin_integrals_unresolved = self.monte_carlo_integral( v_min, v_max, l_mins, l_maxs, resolved=False )
                #bin_integrals = self.simpson_integral( v_min, v_max, l_mins, l_maxs )

                self.logger.debug( f"bin integrals resolved: {bin_integrals_resolved}" )
                self.logger.debug( f"bin integrals unresolved: {bin_integrals_unresolved}" )

                # if we have a 0 bin integral but N > 0 it must be a monte carlo failure or completeness mismatch
                problematic_bins_resolved = ( bin_integrals_resolved == 0 ) & ( n_resolved_in_lum_bins > 0 )
                problematic_bins_unresolved = ( bin_integrals_unresolved == 0 ) & ( n_unresolved_in_lum_bins > 0 )
                if np.any( problematic_bins_resolved | problematic_bins_unresolved ):
                    self.logger.error( f"Monte Carlo failure - {self.n_mc_pts} points insufficient for number of bins" )
                    if np.any( problematic_bins_resolved ):
                        pbr_indices = np.nonzero( problematic_bins_resolved )[ 0 ]
                        if pbr_indices.shape[ 0 ] == 1:
                            index = pbr_indices[ 0 ]
                            self.logger.error( f'Resolved bin {index} had {n_resolved_in_lum_bins[ index ]} sources but a 0 bin integral' )
                            max_flux = self.flux_from_coordinate( v_min, l_maxs[ 0, index ] )
                            min_flux = self.flux_from_coordinate( v_max, l_mins[ 0, index ] )
                            self.logger.error( f'Min flux in bin {min_flux}, max flux in bin {max_flux}, cutoff {self.flux_cut_jy}' )
                        else:
                            self.logger.error( f'{pbr_indices.shape[ 0 ]} resolved bins had sources but a 0 bin integral, indices {pbr_indices}' )
                    if np.any( problematic_bins_unresolved ):
                        pbu_indices = np.nonzero( problematic_bins_unresolved )[ 0 ]
                        if pbu_indices.shape[ 0 ] == 1:
                            index = pbu_indices[ 0 ]
                            self.logger.error( f'Unresolved bin {index} had {n_unresolved_in_lum_bins[ index ]} sources but a 0 bin integral' )
                            max_flux = self.flux_from_coordinate( v_min, l_maxs[ 0, index ] )
                            min_flux = self.flux_from_coordinate( v_max, l_mins[ 0, index ] )
                            self.logger.error( f'Min flux in bin {min_flux}, max flux in bin {max_flux}, cutoff {self.flux_cut_jy}' )
                        else:
                            self.logger.error( f'{pbu_indices.shape[ 0 ]} resolved bins had sources but a 0 bin integral, indices {pbu_indices}' )


                bin_integrals_unresolved[ n_unresolved_in_lum_bins == 0 ] = 1
                bin_integrals_resolved[ n_resolved_in_lum_bins == 0 ] = 1

                # now we have phi_est as given by Page & Carrera 2000
                self.phi[i_z] = n_unresolved_in_lum_bins / bin_integrals_unresolved + n_resolved_in_lum_bins / bin_integrals_resolved
                self.counts[ i_z ] = n_sources_in_lum_bins

                # get errors from poisson statistics
                phi_err_range_resolved = astropy.stats.poisson_conf_interval( n_resolved_in_lum_bins ) / bin_integrals_resolved
                phi_err_range_unresolved = astropy.stats.poisson_conf_interval( n_unresolved_in_lum_bins ) / bin_integrals_unresolved
                phi_err_resolved = np.abs( phi_err_range_resolved[ 1 ] - phi_err_range_resolved[ 0 ] ) / 2
                phi_err_unresolved = np.abs( phi_err_range_unresolved[ 1 ] - phi_err_range_unresolved[ 0 ] ) / 2
               
                phi_err_resolved[ n_resolved_in_lum_bins == 0 ] = 0
                phi_err_unresolved[ n_unresolved_in_lum_bins == 0 ] = 0

                self.logger.debug( f"phi err resolved: {phi_err_resolved}" )
                self.logger.debug( f"phi err unresolved: {phi_err_unresolved}" )
                self.phi_err[ i_z ] = np.sqrt( phi_err_resolved**2 + phi_err_unresolved**2 + (0.05*self.phi[ i_z ])**2 )

                self.logger.info( f'Redshift range {z_min:.2f}-{z_max:.2f} complete' )

        # sky coverage completeness factor
        self.phi /= 5700 / 41253 # 5700 lotss dr2 area from hardcastle et al. 2023, 41253 deg^2 is solid angle of a sphere
        self.phi_err /= 5700 / 41253


        # fit parameters to the RLFs
        self.fit_rlf()

        if plot_rlf:
            # plot the resulting graph
            title = ''
            output = ''
            if vmax_method:
                title = f'1/Va RLF - {self.n_mc_pts // 10} pts per source'
                output = 'rlf_vmax.png'
            else:
                title = f'Page & Carrera RLF - {self.n_mc_pts} pts per bin'
                output = 'rlf_page_and_carrera.png'
            self.plot_rlf( title, colors, output=output )

    def fit_rlf( self ):
        self.logger.info( 'Fitting Parameters to RLFs' )

        # fit a dual power law to each redshift RLF
        bin_centres = ( self.l_bins[ :-1 ] + self.l_bins[ 1: ] ) / 2
        
        # (4,2) being 4 parameters, with [:, 0] being values and [:, 1] being errors
        self.rlf_fit_params = np.zeros( (self.n_z_bins, 4, 2) ) 
        self.chi_sqr = np.zeros( (self.n_z_bins) ) 

        for i_z in range( self.n_z_bins ):
            popt, pcov = curve_fit( functions.rlf_power_law,
                                    bin_centres[ self.phi[ i_z ] > 0 ],
                                    self.phi[ i_z ][ self.phi[ i_z ] > 0 ],
                                    p0=[ 0.5, 1.5, -5.5, 26 ],
                                    bounds=( [0, 1, -10, 20], [1, 4, -1, 30]),
                                    absolute_sigma=True,
                                    sigma=self.phi_err[ i_z ][ self.phi[ i_z ] > 0 ],
                                    maxfev=1000000 )
            perr = np.sqrt( np.diag( pcov ) )
            self.rlf_fit_params[ i_z ] = np.array( [ popt, perr ] ).T
            self.logger.info( f'z={self.z_bins[ i_z ]}-{self.z_bins[ i_z+1 ]}: ' )
            self.logger.info( f'    alpha={popt[ 0 ]:.3f} +/- {perr[ 0 ]:.3f}' )
            self.logger.info( f'    beta={popt[ 1 ]:.3f} +/- {perr[ 1 ]:.3f}' )
            self.logger.info( f'    Log10C={popt[ 2 ]:.2f} +/- {perr[ 2 ]:.2f}' )
            self.logger.info( f'    Log10Lstar={popt[ 3 ]} +/- {perr[ 3 ]}' )

            model_values = functions.rlf_power_law( bin_centres[ self.phi[ i_z ] > 0 ], *popt )
            residuals = self.phi[ i_z ][ self.phi[ i_z ] > 0 ] - model_values
            self.chi_sqr[ i_z ] = np.sum( ( residuals / self.phi_err[ i_z ][ self.phi[ i_z ] > 0 ] )**2 )
            self.chi_sqr[ i_z ] /= residuals.shape[ 0 ] - 4
            self.logger.info( f'Reduced chi squared: {self.chi_sqr[ i_z ]}' )
        self.chi_sqr_tot = np.sum( self.chi_sqr )
        self.logger.info( f'Reduced chi squared total: {self.chi_sqr_tot}' )
            

    def plot_rlf( self, 
                  title: str,
                  colors: list,
                  ax = None,
                  ylim: tuple = [ 1e-9, 3e-4 ],
                  xlim: tuple = [ 1e21, 1e29 ],
                  output: str = 'rlf.png',
                  draw_ylabel: bool = True ):
        self_contained = ax is None
        if self_contained:
            self.logger.info( "plotting self contained rlf" )
            fig, ax = plt.subplots( figsize=(10,10) )

        luminosity_space = np.geomspace( self.l_bins[ 0 ], self.l_bins[ -1 ], num=100 )

        bin_centres = ( self.l_bins[ :-1 ] + self.l_bins[ 1: ] ) / 2
        for i_z in range( self.phi.shape[0] ):
            fit_params = self.rlf_fit_params[ i_z, :, 0 ]
            fitted_rlf = functions.rlf_power_law( luminosity_space, *fit_params )
            ax.plot( luminosity_space, fitted_rlf, color=colors[ i_z ] )

            specific_phi = self.phi[i_z]
            specific_counts = self.counts[i_z]
            mask = specific_counts >= 1
            ax.plot( bin_centres[ mask ], specific_phi[ mask ], color=colors[ i_z ],
                     marker='o', linestyle='None',
                     label=f'{self.z_bins[ i_z ]:.2f}<z<{self.z_bins[ i_z + 1 ]:.2f}')
            ax.errorbar( bin_centres[ mask ], specific_phi[ mask ], yerr=self.phi_err[ i_z ][ mask ], color=colors[ i_z ], fmt='none' )
        ax.set_title( title )
        ax.set_xscale( 'log' )
        ax.set_xlabel( '$L_{144}$  $( \\frac{W}{Hz} )$' )
        ax.set_yscale( 'log' )
        if draw_ylabel:
            ax.set_ylabel( '$\\phi_{est}$  $( Mpc^{-3} (\\log_{10}( \\frac{W}{m^2} ))^{-1} )$' )
        ax.set_xlim( xlim )
        ax.set_ylim( ylim )
        ax.grid()
        ax.tick_params( which='both', right=True, top=True )
        ax.legend()
        if self_contained:
            plt.savefig( output )
            self.logger.info( f"saved figure to {output}" )

def get_catalog_info( cosmo, flux_cut_jy: float, plot_rlagn_selection_contour: bool = False ):
    logger = get_logger("RLF Catalog Info", logging.INFO)

    catalog = HardcastleCatalogue( resolved_only=False )

    logger.info( "getting catalog data" )
    redshifts = catalog.get_values( Source.Redshift )
    fluxes = catalog.get_values( Source.TotalFlux ) / 1000
    luminosities = catalog.get_values( Source.Luminosity )
    wise_3_mag = catalog.get_values( Source.WISE3Mag )
    wise_3_magerr = catalog.get_values( Source.WISE3MagErr )
    wise_2_mag = catalog.get_values( Source.WISE2Mag )
    resolved = catalog.get_values( Source.Resolved )
    logger.info( "done" )

    mask = ( fluxes > 0 ) & ( wise_3_mag > 0 ) & ( wise_2_mag > 0 ) & ~np.isnan( fluxes ) & ~np.isnan( wise_3_mag ) & ~np.isnan( wise_2_mag ) & ~np.isnan( redshifts ) & ~np.isnan( luminosities ) & ( redshifts > 0.01 ) & ( fluxes > flux_cut_jy ) 
    redshifts = redshifts[ mask ]
    luminosities = luminosities[ mask ]
    wise_3_mag = wise_3_mag[ mask ]
    wise_3_magerr = wise_3_magerr[ mask ]
    wise_2_mag = wise_2_mag[ mask ]
    fluxes = fluxes[ mask ]
    resolved = resolved[ mask ]

    logger.debug( f'wise_3_mag: mean={np.average( wise_3_mag )}, std={np.std( wise_3_mag )}, max={np.max( wise_3_mag )}, min={np.min( wise_3_mag ) }, count={wise_3_mag.shape[ 0 ]}' )
    logger.debug( f'wise_2_mag: mean={np.average( wise_2_mag )}, std={np.std( wise_2_mag )}, max={np.max( wise_2_mag )}, min={np.min( wise_2_mag ) }, count={wise_2_mag.shape[ 0 ]}' )

    # use wise bands 2/3 to calculate spectral indices for the k-correction
    wise_3_flux = mag_to_flux_w3( wise_3_mag )
    wise_2_flux = mag_to_flux_w2( wise_2_mag )
    wise_3_freq = 3e8 / 12e-6
    wise_2_freq = 3e8 / 4.6e-6
    spectral_inds = -np.log( wise_3_flux / wise_2_flux ) / np.log( wise_3_freq / wise_2_freq )
    logger.debug( f'spectral_inds: mean={np.average( spectral_inds )}, std={np.std( spectral_inds )}, max={np.max( spectral_inds )}, min={np.min( spectral_inds ) }, count={spectral_inds.shape[ 0 ]}' )

    wise_3_absmag = wise_3_mag - 5 * ( np.log10( cosmo.luminosity_distance( redshifts ).to(u.parsec).value ) - 1 ) + k_corr_factor( redshifts, mag_space=True, spectral_index=spectral_inds )

    # plot the relationship between L144 and Abs W3 (Fig. 2, H25)
    rqq_xpt = -27.923076923076923 #mag
    rqq_ypt = 25.563106796116504 #log10( lum )
    if plot_rlagn_selection_contour:
        wise_3_linspace_sfg = np.linspace( -27, -18, 1000 )
        wise_3_linspace_rqq = np.linspace( -34, -27, 1000 )
        sfg_lum_cutoff = 10**( 14 - wise_3_linspace_sfg / 2.5 )
        rqq_lum_cutoff = 10**( -( wise_3_linspace_rqq - rqq_xpt ) / 3.4844629455909923 + rqq_ypt )
        plt.figure( figsize=(8,8) )
        hist2d, xedges, yedges = np.histogram2d( wise_3_absmag, np.log10( luminosities ), bins=50, range=[[-35, -17], [19+np.log10(4), 30]] )
        xcenters = ( xedges[ 1: ] + xedges[ :-1 ] ) / 2
        ycenters = ( yedges[ 1: ] + yedges[ :-1 ] ) / 2
        plt.contourf( xcenters[ ::-1 ], 10**ycenters, np.sqrt( hist2d[ ::-1, : ].T ) )
        plt.plot( wise_3_linspace_sfg, sfg_lum_cutoff, color='r' )
        plt.plot( wise_3_linspace_rqq, rqq_lum_cutoff, color='m' )
        plt.plot( [-27, -27], [1, rqq_lum_cutoff[ -1 ]], color='m' )
        plt.xlabel( 'wise 3 absolute magnitude' )
        plt.ylabel( 'L144' )
        plt.title( 'H25 RLAGN Selection Plot' )
        plt.yscale( 'log' )
        plt.xlim( -18, -34 )
        plt.ylim( 4e20, 1e29 )
        plt.savefig( 'lum_vs_w3.png' )
        plt.show()
        logger.info( "saved lum_vs_w3.png" )


    sfg_mask = ( luminosities < 10**( 14 - wise_3_absmag / 2.5 ) ) & ( luminosities < 10**(24.8) ) & ~np.isnan( wise_3_magerr )
    rqq_mask = ( luminosities < 10**( -( wise_3_absmag - rqq_xpt ) / 3.4844629455909923 + rqq_ypt ) ) & ( wise_3_absmag < -27 ) & ~np.isnan( wise_3_magerr )
    agn_mask = ~sfg_mask & ~rqq_mask

    logger.info( f'# agn: {redshifts[ agn_mask ].shape[ 0 ]} - # sfg: {redshifts[ sfg_mask ].shape[ 0 ]} - # rqq: {redshifts[ rqq_mask ].shape[ 0 ]} - total: {redshifts.shape[ 0 ]}' )
    logger.info( f'{np.isnan( wise_3_magerr ).sum()} wise 3 values are upper limits' )

    redshifts = redshifts[ agn_mask ]
    fluxes = fluxes[ agn_mask ]
    luminosities = luminosities[ agn_mask ]
    resolved = resolved[ agn_mask ]

    return redshifts, fluxes, luminosities, resolved


if __name__ == "__main__":
    logger = get_logger("RLF Main", logging.INFO)
    plt.rcParams[ 'font.size' ] = 18

    flux_cut_jy = 1.1e-3

    # Read parameters from the config.ini file
    config = configparser.ConfigParser()
    config.read(pth.PROGRAM_CONFIG)
    default_config = config['DEFAULT']

    # Cosmological Parameters
    h = float(default_config['h']) # hubble constant = h * 100 km/s/Mpc
    Tcmb0 = float(default_config['Tcmb0']) # temp of the CMB at z=0 in K
    Om0 = float(default_config['Om0']) # matter density parameter at z=0
    cosmo = astropy.cosmology.FlatLambdaCDM(h * 100 * u.km / u.s / u.Mpc, Tcmb0=Tcmb0 * u.K, Om0=Om0)

    redshifts, fluxes, luminosities, resolved = get_catalog_info( cosmo, 1.1e-3 )
    redshifts1, fluxes1, luminosities1, resolved1 = get_catalog_info( cosmo, 0.5e-3 )

    all_unresolved = np.zeros( resolved.shape[ 0 ], dtype=bool )

    rlf0 = RLF( fluxes, redshifts, luminosities, resolved, cosmo, bias=0, flux_cut_jy=1.1e-3 )
    rlf1 = RLF( fluxes1, redshifts1, luminosities1, resolved1, cosmo, bias=0, flux_cut_jy=0.5e-3 )
    rlfs = [ rlf0, rlf1 ]

    fig, axes = plt.subplots( ncols=2, figsize=(20, 10) )

    logger.debug( f"lum: {np.min( luminosities )}-{np.max( luminosities )}, redsh: {np.min( redshifts )}-{np.max( redshifts )}, flux: {np.min( fluxes )}-{np.max( fluxes )}" )

    rlf_titles = [ f'Shimwell et al. 2022 After Flux Cut',
                   f'Flux Cut Only',
                   f'Composite RLF -0.1 mJy Resolved Completeness',
                   f'Composite RLF +0.1 mJy Resolved Completeness',
                   f'H25 RLF using Page & Carrera 2000',
                   f'H25 RLF using $1/V_a$',
                   f'H23 Catalog RLF, Flux Cut 0.5mJy',
                   f'H23 Catalog RLF, Flux Cut 1.1mJy' ]
    rlf_titles = [ rlf_titles[ -1 ], rlf_titles[ -2 ] ]
    draw_ylabels = [ True, False ]

    for rlf, ax, title, draw_ylabel in zip( rlfs, axes, rlf_titles, draw_ylabels ):
        rlf.calculate_rlf( plot_rlf=False )
        rlf.plot_rlf( title, colors, ax, draw_ylabel=draw_ylabel )

    plt.savefig( 'rlfs.png' )
    plt.show()

    logger.info( 'done' )
    
