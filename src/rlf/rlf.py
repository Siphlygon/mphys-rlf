import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
from utils.img_data_arrays import ImageDataArrays
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import loguniform

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


    def get_completeness(self, integ_fluxes, completeness_path=None):
        # Read completeness function parameters from file
        #print( f"min flux: {np.min( integ_fluxes )} - max flux: {np.max( integ_fluxes )} - cutoff 0.01" )
        return np.where( integ_fluxes > 1e-3, 1, 0 )

    def calculate_rlf(self):
        """
        Calculate and plot the estimated differential RLF for the generated data
        """
        print( "start rlf calculation" )
        for subdir in [self.dataset_subdir, self.generated_subdir]:
            print( f"subdir {subdir}" )
            # assign each AGN a random redshift
            data = ImageDataArrays(subdir)
            redshifts = np.random.uniform(0.01, 1, len(data.images))

            # use the redshift to calculate luminosity distance and luminosity
            cosmo = astropy.cosmology.FlatLambdaCDM(self.h * 100 * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)
            luminosity_distances = cosmo.luminosity_distance(redshifts).to(u.m).value
            luminosities = data.model_fluxes * ( 1e-26 * luminosity_distances ) * (4 * np.pi * luminosity_distances) / ( 1 + redshifts )**(1+self.spectral_index) # W/Hz

            # clip luminosity values to those above 0 for log plotting
            redshifts = redshifts[ luminosities > 0 ]
            luminosities = luminosities[ luminosities > 0 ]

            print( f"luminosities {np.min( luminosities )} to {np.max( luminosities )}" )

            # select a redshift-luminosity bin, use monte-carlo to populate the bin,
            # work backwards to find fluxes and weight by completeness function to calculate integral as in Page & Carrera 2000.
            # Also calculate the number of sources in the bin while we're at it
            z_bins = np.arange(0.01, 1, self.dz) # only consider sources with z < 1 for now, we can extend this later if needed
            l_bins = np.logspace(21, 29, self.lum_bins_count)
            phi_est = np.zeros((len(z_bins), self.lum_bins_count-1))

            # Can read previously created numpy files to avoid recomputation
            #if (pth.NP_ARRAY_PARENT / subdir / 'rlf.npy').exists():
            #    phi_est = np.load(pth.NP_ARRAY_PARENT / subdir / 'rlf.npy')
            # iteration so we can index phi_est by phi_est[ i_z, i_l ]
            print( f"z bins to iterate over: {len(z_bins)}" )
            print( f"l bins to iterate over: {self.lum_bins_count}" )

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
                redshift_grid = np.geomspace(z_min, z_max, self.n_interp_pts)
                volume_grid = cosmo.comoving_volume(redshift_grid)
                random_redshifts = np.interp(random_volumes, volume_grid.value, redshift_grid)

                for i_l in range(self.lum_bins_count-1):
                    # Bins are defined by their minimum value
                    l_min = l_bins[i_l]

                    # Find max value in bin by adding width to min_value
                    # Width unspecified for L bins, so we calc based on max
                    l_max = l_bins[i_l+1]

                    print( f"luminosity range {l_min} - {l_max}" )

                    # then generate random luminosities -> fluxes
                    # flux values here are in Jy, luminosities in W/Hz
                    random_luminosities = np.random.uniform(l_min, l_max, self.n_mc_pts)
                    random_luminosity_distances = cosmo.luminosity_distance(random_redshifts).to(u.m).value
                    random_fluxes = random_luminosities / (4 * np.pi * random_luminosity_distances) / ( 1e-26 * random_luminosity_distances ) * (1+random_redshifts)**(1+self.spectral_index)

                    # weight each point by Completeness[ flux ] and add to total monte-carlo integral
                    # for now, placeholder, assume flux cutoff at 1 mJy
                    bin_integral = np.sum(self.get_completeness(random_fluxes)) / self.n_mc_pts

                    # divide by the luminosity-volume bin area so the result is / MPc^3 / (W/Hz)
                    bin_integral *= ( v_max - v_min ).to( u.Mpc**3 ).value * ( np.log10( l_max ) - np.log10( l_min ) )

                    # now calculate the number of 'real' sources in this bin
                    redshift_mask = (redshifts >= z_min) & (redshifts < z_max)
                    luminosity_mask = (luminosities >= l_min) & (luminosities < l_max)
                    N = np.sum( redshift_mask & luminosity_mask )

                    # divide by the number of total sources to get a density
                    #N_tot = np.size( data.model_fluxes )

                    # now we have phi_est as given by Page & Carrera 2000
                    #phi_est[i_z, i_l] = N / N_tot / bin_integral.value
                    if N == 0 or bin_integral == 0:
                        phi_est[i_z, i_l] = 0
                    else:
                        phi_est[i_z, i_l] = N / bin_integral
                print( f'Redshift range {z_bins[i_z]:.2f}-{z_bins[i_z]+self.dz:.2f} complete' )

            #np.save(pth.NP_ARRAY_PARENT / subdir / 'rlf.npy', phi_est)

            # plot the resulting graph
            print( "plotting..." )
            plt.figure()
            for i_z in range(phi_est.shape[0]):
                specific_phi_est = phi_est[i_z]
                mask = specific_phi_est > 0
                plt.plot( l_bins[ :-1][ mask ], specific_phi_est[ mask ], color=colormaps['cool'](i_z / phi_est.shape[0]),
                         marker='o',
                         label=f'{z_bins[i_z]:.2f}<z<{z_bins[i_z]+self.dz:.2f}')
                
            plt.xscale( 'log' )
            plt.xlabel( 'L144 * Hz / W')
            plt.yscale( 'log' )
            plt.ylabel( 'phi * MPc^3 * log10( W / m^2 )' )
            plt.legend()
            plt.savefig(f'{subdir}_rlf.png')
            print( "saved figure" )


if __name__ == "__main__":
    rlf_calculator = RLF()
    rlf_calculator.calculate_rlf()
