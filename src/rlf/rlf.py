import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
from utils.img_data_arrays import ImageDataArrays
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import loguniform
from image_downloading.hardcastle_catalogue import Property, HardcastleCatalogue
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

        redshift_grid = np.geomspace( Z_MIN, Z_MAX, self.n_interp_pts )
        volume_grid = cosmo.comoving_volume( redshift_grid )
        if redshifts is None:
            # assign each source a comoving volume with a uniform dist s.t. dN/dV = const
            volumes = np.random.uniform( V_MIN, V_MAX, fluxes.shape[ 0 ] )

            # conversion from comoving volume to redshift
            redshifts = np.interp(volumes, volume_grid.value, redshift_grid)

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
            v_min, v_max = cosmo.comoving_volume([z_min, z_max])

            # -- MONTE CARLO METHOD ---
            # Now generate random points in this bin; so random in volume and in luminosity, and then
            # compare to the completeness function to find the function of RLF in that bin.

            # we need to generate random comoving volumes and calculate the respective redshifts in MC method
            # getting redshift from comoving volume would involve reversing an integral, which can be quite
            # complicated. Official documentation offers linear interpolation as a solution
            random_volumes = np.random.uniform(v_min.value, v_max.value, self.n_mc_pts)
            random_redshifts = np.interp(random_volumes, volume_grid.value, redshift_grid)[ :, np.newaxis ]

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

            # then generate random luminosities -> fluxes
            # flux values here are in Jy, luminosities in W/Hz
            # random_fluxes has shape (self.n_mc_pts, n_lum_bins) for compat w/ np.random.uniform
            random_luminosities = loguniform.rvs(l_mins[ 0, : ], l_maxs[ 0, : ], size=(self.n_mc_pts, n_lum_bins))
            random_luminosity_distances = (cosmo.luminosity_distance(random_redshifts).to(u.m).value)
            random_fluxes = random_luminosities / (4 * np.pi * random_luminosity_distances) / ( 1e-26 * random_luminosity_distances ) * (1+random_redshifts)**(1+self.spectral_index)

            # weight each point by Completeness[ flux ] and add to total monte-carlo integral
            # for now, placeholder, assume flux cutoff at 1 mJy
            bin_integrals = np.sum(self.get_completeness(random_fluxes), axis=0) / self.n_mc_pts

            # divide by the luminosity-volume bin area so the result is / MPc^3 / (W/Hz)
            bin_integrals *= ( v_max - v_min ).to( u.Mpc**3 ).value * ( np.log10( l_maxs[ 0, : ] ) - np.log10( l_mins[ 0, : ] ) )

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
    catalog = HardcastleCatalogue()
    hardcastle_dict_list = catalog.get_multiple_values( Property.Redshift.value, Property.TotalFlux.value, Property.Luminosity.value ) 
    hardcastle_data = [ [ dict[ Property.Redshift.value ], dict[ Property.TotalFlux.value ], dict[ Property.Luminosity.value ] ] for dict in hardcastle_dict_list ]

    hardcastle_data = np.array( hardcastle_data )
    redshifts = hardcastle_data[ :, 0 ]
    fluxes = hardcastle_data[ :, 1 ]
    luminosities = hardcastle_data[ :, 2 ]

    redshift_lum_mask = ~(np.isnan( luminosities ) | np.isnan( redshifts ))
    hardcastle_flux_mask = fluxes > 1.1e-3

    mask = redshift_lum_mask & hardcastle_flux_mask

    redshifts = redshifts[ mask ]
    fluxes = fluxes[ mask ]
    luminosities = luminosities[ mask ]

    logger.debug( f"lum: {np.min( luminosities )}-{np.max( luminosities )}, redsh: {np.min( redshifts )}-{np.max( redshifts )}, flux: {np.min( fluxes )}-{np.max( fluxes )}" )
    rlf_calculator.calculate_rlf( fluxes, redshifts, luminosities )
