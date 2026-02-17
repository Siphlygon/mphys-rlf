import configparser
import utils.paths as pth
import numpy as np
import astropy.cosmology
import astropy.units as u
from utils.img_data_arrays import ImageDataArrays
import matplotlib.pyplot as plt
from matplotlib import colormaps

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
        self.lum_bins_count = int(lu_config['lum_bins']) # number of luminosity bins between min and max luminosity
        self.n_interp_pts = int(lu_config['n_interp_pts']) # number of points to use in interpolation approximation of
        self.n_mc_pts = int(lu_config['n_mc_pts']) # number of points to use in the monte-carlo integral for each redshift-luminosity bin


    def get_completeness(self, integ_fluxes, completeness_path=pth.COMPLETENESS_FUNC_PARAMS):
        # Read completeness function parameters from file
        return 1


    def calculate_rlf(self):
        """
        Calculate and plot the estimated differential RLF for the generated data
        """
        for subdir in [self.dataset_subdir, self.generated_subdir]:
            # assign each AGN a random redshift
            data = ImageDataArrays(subdir)
            redshifts = np.random.normal(0.5, 0.3, len(data.images))

            # use the redshift to calculate luminosity distance and luminosity
            cosmo = astropy.cosmology.FlatLambdaCDM(self.h * u.km / u.s / u.Mpc, Tcmb0=self.Tcmb0 * u.K, Om0=self.Om0)
            luminosity_distances = cosmo.luminosity_distance(redshifts).to(u.Mpc).value
            luminosities = data.model_fluxes * (4 * np.pi * luminosity_distances ** 2)

            # select a redshift-luminosity bin, use monte-carlo to populate the bin,
            # work backwards to find fluxes and weight by completeness function to calculate integral as in Page & Carrera 2000.
            # Also calculate the number of sources in the bin while we're at it
            z_bins = np.arange(0, 1, self.dz) # only consider sources with z < 1 for now, we can extend this later if needed
            l_bins = np.linspace(np.min(luminosities), np.max(luminosities), self.lum_bins_count)
            phi_est = np.zeros((len(z_bins), self.lum_bins_count))

            # Can read previously created numpy files to avoid recomputation
            if (pth.NP_ARRAY_PARENT / subdir / 'rlf.npy').exists():
                phi_est = np.load(pth.NP_ARRAY_PARENT / subdir / 'rlf.npy')
            else:
                # iteration so we can index phi_est by phi_est[ i_z, i_l ]
                for i_z in range(len(z_bins)):
                    for i_l in range(self.lum_bins_count):
                        # Bins are defined by their minimum value
                        z_min = z_bins[i_z]
                        l_min = l_bins[i_l]

                        # Find max value in bin by adding width to min_value
                        # Width unspecified for L bins, so we calc based on max
                        z_max = z_min + self.dz
                        l_max = l_min + np.max(luminosities) / self.lum_bins_count

                        # find min & max of comoving volume for redshift bin
                        v_min, v_max = cosmo.comoving_volume([z_min, z_max])

                        # -- MONTE CARLO METHOD ---
                        # Now generate random points in this bin; so random in volume and in luminosity, and then
                        # compare to the completeness function to find the function of RLF in that bin.

                        # we need to generate random comoving volumes and calculate the respective redshifts in MC method
                        # getting redshift from comoving volume would involve reversing an integral, which can be quite
                        # complicated. Official documentation offers linear interpolation as a solution
                        random_volumes = np.random.uniform(v_min.value, v_max.value, self.n_mc_pts)
                        redshift_grid = np.logspace(z_min, z_max, self.n_interp_pts)
                        volume_grid = cosmo.comoving_volume(redshift_grid)
                        random_redshifts = np.interp(random_volumes, volume_grid.value, redshift_grid)

                        # then generate random luminosities -> fluxes
                        # flux values here are in Jy, luminosities in Jy * Mpc**2
                        random_luminosities = np.random.uniform(l_min, l_max, self.n_mc_pts)
                        random_luminosity_distances = cosmo.luminosity_distance(random_redshifts).to(u.Mpc).value
                        random_fluxes = random_luminosities / (4 * np.pi * random_luminosity_distances ** 2)

                        # weight each point by Completeness[ flux ] and add to total monte-carlo integral
                        # for now, placeholder, assume flux cutoff at 1 mJy
                        bin_integral = np.sum(self.get_completeness(random_fluxes)) / self.n_mc_pts

                        # divide by the redshift bin width so the result is /MPc^3
                        bin_integral /= z_max - z_min

                        # now calculate the number of 'real' sources in this bin
                        redshift_mask = (redshifts >= z_min) & (redshifts < z_max)
                        luminosity_mask = (luminosities >= l_min) & (luminosities < l_max)
                        N = np.size(data.model_fluxes[redshift_mask & luminosity_mask])

                        # now we have phi_est as given by Page & Carrera 2000
                        phi_est[i_z, i_l] = N / bin_integral

                np.save(pth.NP_ARRAY_PARENT / subdir / 'rlf.npy', phi_est)

            # plot the resulting graph
            for i_z in range(phi_est.shape[0]):
                specific_phi_est = phi_est[i_z]
                plt.plot(l_bins, specific_phi_est, color=colormaps['hsv'](i_z / phi_est.shape[0]),
                         label=f'z={z_bins[i_z]:.2f}')
                
            plt.xscale( 'log' )
            plt.xlabel( 'log[ Luminosity / Jy / MPc^2 ]')
            plt.yscale( 'log' )
            plt.ylabel( 'log[ phi / MPc^3 ]')
            plt.legend()
            plt.savefig(f'{subdir}_rlf.png')


if __name__ == "__main__":
    rlf_calculator = RLF()
    rlf_calculator.calculate_rlf()
