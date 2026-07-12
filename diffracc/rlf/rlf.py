import configparser
from pathlib import Path

import astropy.cosmology
import astropy.stats
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from ..utils import functions as func
from ..utils import paths
from ..utils.logger import LoggingLevels, get_logger
from .rlf_constants import colors, shimwell_data, z_from_v


class RLF:
    """
    A class to calculate the radio luminosity function (RLF) of a sample of AGN using either the Page & Carrera 2000
    method or the traditional 1/Vmax method. The RLF is calculated in bins of redshift and luminosity, and can be
    corrected for completeness using a fitted sigmoid function or a step function.
    """

    def __init__(self,
                 fluxes: np.ndarray,
                 redshifts: np.ndarray,
                 luminosities: np.ndarray,
                 resolved: np.ndarray,
                 cosmo : astropy.cosmology.FlatLambdaCDM,
                 bias: float = 0,
                 flux_cut_jy: float = 1.1e-3,
                 debug_flux_lum_relation: bool = False,
                 vmax_method: bool = False,
                 use_shimwell: bool = True,
                 completeness_path: str | Path | None = None,
                 use_pde: bool = False):
        """
        Initialise the RLF class with the necessary parameters for calculating the radio luminosity function (RLF) of a
        sample of AGN.

        Parameters
        ----------
        fluxes : np.ndarray
            The array of integrated fluxes of the AGN in Jy
        redshifts : np.ndarray
            The array of redshifts of the AGN
        luminosities : np.ndarray
            The array of luminosities of the AGN
        resolved : np.ndarray
            The array indicating whether each AGN is resolved
        cosmo : astropy.cosmology.FlatLambdaCDM
            The cosmology object for distance calculations
        bias : float, optional
            The bias factor for the RLF calculation, by default 0
        flux_cut_jy : float, optional
            The flux cut in Jy, by default 1.1e-3
        debug_flux_lum_relation : bool, optional
            Whether to debug the flux-luminosity relation, by default False
        vmax_method : bool, optional
            Whether to use the 1/Vmax method, by default False
        use_shimwell : bool, optional
            Whether to use the Shimwell et al. (2003) method, by default True
        completeness_path : str | None, optional
            The path to the completeness arguments file, by default None
        use_pde : bool, optional
            Whether to use the PDE method, by default False
        """
        # Start logging
        self.logger = get_logger("RLF", LoggingLevels.DEBUG.value)

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
        self.use_pde = use_pde

        if completeness_path is None:
            completeness_path = paths.NP_ARRAY_PARENT / 'completeness_args_sigmoid.txt'
        if isinstance(completeness_path, str):
            completeness_path = Path(completeness_path)
        if completeness_path.exists():
            self.completeness_args = np.loadtxt(completeness_path)
        else:
            # it looks like completeness is actually necessary, so raise exception
            self.logger.error(f'Could not find completeness args at path {completeness_path}')
            raise FileNotFoundError(f'Could not find completeness args at path {completeness_path}')

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)
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
        self.v_min, self.v_max = self.cosmo.comoving_volume([self.z_min, self.z_max]).to(u.Mpc**3).value
        self.logger.debug(f"volume range: {self.v_min}-{self.v_max}")
        self.redshift_grid = np.geomspace(self.z_min, self.z_max, self.n_interp_pts)
        self.volume_grid = self.cosmo.comoving_volume(self.redshift_grid).to(u.Mpc**3).value

        # define RLF z/l bins
        if default_config.getboolean('HARDCASTLE_Z_BINS'):
            self.z_bins = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]) #hardcastle bins
        elif default_config.getboolean('DEJONG_Z_BINS'):
            self.z_bins = np.array([0.01, 0.3])
        else:
            self.z_bins = np.arange(self.z_min, self.z_max, self.dz)
        #self.z_bins = np.array([0.01, 0.3, 0.5, 1.0])

        self.l_bins = np.logspace(self.l_min, self.l_max, self.lum_bins_count)
        self.n_lum_bins = self.lum_bins_count - 1
        self.n_z_bins = self.z_bins.shape[0] - 1

        # params for interp
        self.zvparams = [self.volume_grid, self.redshift_grid]

        # init rlf values as zero
        self.phi = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.phi_err = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.counts = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.rlf_fit_params = np.zeros((self.n_z_bins, 4, 2))


    # ---------- COMPLETENESS ----------
    def get_completeness(self,
                         integ_fluxes: np.ndarray,
                         resolved: np.ndarray) -> np.ndarray:
        """
        Returns a value for the completeness correction for use in the RLF integral estimation. Can either return a
        fitted sigmoid completeness read from a file, or a step completeness function (i.e., 1 if above a threshold, 0
        otherwise)

        Parameters
        ----------
        integ_fluxes : np.ndarray
            The integrated fluxes of the sources in Jy
        resolved : np.ndarray
            The array indicating whether each source is resolved

        Returns
        -------
        np.ndarray
            The completeness correction for each source
        """
        completeness_args = self.completeness_args
        #self.logger.debug(f'x0: {completeness_args[0]} - S0: {10**completeness_args[0]} - bias: {self.bias}')
        completeness_args[0] = np.log10(10**completeness_args[0] + self.bias)
        #self.logger.debug(f'x\'0: {completeness_args[0]} - S\'0: {10**completeness_args[0]}')

        sigmoid_completeness = func.sigmoid(np.log10(integ_fluxes * 1000), *completeness_args)
        resolved_completeness = np.where(integ_fluxes > self.flux_cut_jy, sigmoid_completeness, 0)
        
        # Use Shimwell et al. (2023) completeness for unresolvedd sources if use_shimwell is True, otherwise use a step
        # function at the flux_cut_jy threshold.
        if self.use_shimwell:
            shimwell_completeness = np.interp(integ_fluxes, shimwell_data[0] / 1000, shimwell_data[1])
            unresolved_completeness = np.where(integ_fluxes > self.flux_cut_jy, shimwell_completeness, 0)
        else:
            unresolved_completeness = np.where(integ_fluxes > self.flux_cut_jy, 1, 0)

        completeness_args[0] = np.log10(10**completeness_args[0] - self.bias)

        return np.where(resolved, resolved_completeness, unresolved_completeness)


    def get_completeness_from_coord(self,
                                    v: float | np.ndarray,
                                    l: float | np.ndarray,
                                    resolved: np.ndarray | bool) -> np.ndarray:
        """
        Functions as a proxy for the get_completeness functon, allowing it to be ran in volume-luminosity space without
        requiring at-use computation of the integrated flux.

        Parameters
        ----------
        v : float | np.ndarray
            The volume(s) to compute the completeness for
        l : float | np.ndarray
            The luminosity/luminosities to compute the completeness for
        resolved : np.ndarray | bool
            The array indicating whether each source is resolved, or a single boolean value for all sources
            
        Returns
        -------
        np.ndarray
            The completeness correction for each source
        """
        #logger.debug(f'C[s(v,l)]: v={v.shape if isinstance(v, np.ndarray) else v}, l={l.shape}, s={self.flux_from_coordinate(v, l, cosmo, zvparams)}')
        return self.get_completeness(self.flux_from_coordinate(v, l), resolved=resolved)


    def flux_from_coordinate(self,
                             v: float | np.ndarray,
                             l: float | np.ndarray,
                             z: float | np.ndarray | None = None) -> np.ndarray:
        """
        Generate luminosities + redshifts -> fluxes. Flux values here are in Jy, luminosities in W/Hz
        
        Parameters
        ----------
        v : float | np.ndarray
            The volume(s) to compute the flux for
        l : float | np.ndarray
            The luminosity/luminosities to compute the flux for
        z : float | np.ndarray | None, optional
            The redshift(s) to compute the flux for, by default None. If None, the redshift is computed from the volume
            
        Returns
        -------
        np.ndarray
            The fluxes corresponding to the input volumes and luminosities
        """
        if z is None:
            z = z_from_v(v, *self.zvparams)

        # Find the luminosity distance & convert into flux with a k-correction
        d_l = self.cosmo.luminosity_distance(z).to(u.m).value
        s = 1e26 * l / (4 * np.pi * d_l**2) * func.k_corr_factor(z, spectral_index = self.spectral_index)
        return s


    # ---------- INTEGRALS ----------

    def monte_carlo_integral(self,
                             v_min: float,
                             v_max: float,
                             l_min: float | np.ndarray,
                             l_max: float | np.ndarray,
                             resolved: np.ndarray | bool,
                             lum: np.ndarray | float | None = None,
                             vmax_method: bool = False) -> np.ndarray:
        """
        Evaluate the integral of the completeness function over a volume-luminosity bin using a Monte Carlo method. This
        method generates random points in volume-luminosity space and evaluates the completeness at each point to
        determine the integral C[S[v,L]] dV dlog10L from v=(v_min, v_max) and l=(l_min, l_max).

        Parameters
        ----------
        v_min : float
            The minimum volume of the bin
        v_max : float
            The maximum volume of the bin
        l_min : float | np.ndarray
            The minimum luminosity of the bin, can be a single value or an array of values
        l_max : float | np.ndarray
            The maximum luminosity of the bin, can be a single value or an array of values
        resolved : np.ndarray | bool
            The array indicating whether each source is resolved, or a single boolean value for all sources
        lum : np.ndarray | float | None, optional
            The luminosity/luminosities to compute the completeness for, by default None. If None, random luminosities
            within the bin(s) are generated
        vmax_method : bool, optional
            Whether to use the 1/Vmax method, by default False. If True, the number of Monte Carlo points is reduced by
            a factor of 10 to speed up the calculation.
            
        Returns
        -------
        np.ndarray
            The integral of the completeness function over the volume-luminosity bin, divided by the log
            luminosity-volume bin area so the result is in units of / MPc^3 / log10(W/Hz)
        """
        # V_max has many more calculations so we can reduce the number of Monte Carlo points to speed up the function
        mc_pts = self.n_mc_pts // 10 if vmax_method else self.n_mc_pts

        # If lum is None, generate random luminosities within the bin(s).
        if lum is None:
            if isinstance(l_min, np.ndarray) and isinstance(l_max, np.ndarray):
                lums = 10**np.random.uniform(np.log10(l_min[0, :]), np.log10(l_max[0, :]),
                                             size=(mc_pts, l_min.shape[1]))
            else:
                lums = 10**np.random.uniform(np.log10(l_min), np.log10(l_max),
                                             size=(mc_pts, l_min.shape[1]))

        # If lum is provided, ensure it is a 2D array with shape (n_mc_pts, n_integrals). If it is a 1D array, reshape
        # it to (1, n_integrals).
        elif isinstance(lum, np.ndarray):
            if isinstance(l_min, np.ndarray) and isinstance(l_max, np.ndarray):
                raise AssertionError('Cannot have lum and l_min/max as nparrays')

            if lum.ndim == 1:
                lums = lum[np.newaxis, :]
            elif lum.ndim == 2:
                lums = lum
            else:
                raise RuntimeError(f'lum arg of ndims {lum.ndim} invalid, must be at most 2')
    
        # lums now definitely has shape (self.n_mc_pts, n_integrals)

        # -- MONTE CARLO METHOD ---
        # Now generate random points in volume space and either use given luminosities or random luminosities within the
        # bin(s) evaluate the completeness at each point to determine the integral C[S[v,L]] dV dlog10L from
        # v=(v_min, v_max) and l=(l_min, l_max) random_volumes has shape (self.n_mc_pts, 1) while lums has shape
        # (1, n_integrals)
        # note: resolved has shape (n_integrals) or is a bool
        random_volumes = np.random.uniform(v_min, v_max, mc_pts)[:, np.newaxis]
        bin_integrals = np.sum(self.get_completeness_from_coord(random_volumes, lums, resolved=resolved), axis=0) / mc_pts

        # divide by the log luminosity-volume bin area so the result is / MPc^3 / log10(W/Hz)
        if isinstance(l_min, np.ndarray) and isinstance(l_max, np.ndarray):
            bin_integrals *= (v_max - v_min) * (np.log10(l_max[0, :] / l_min[0, :]))
        else:
            bin_integrals *= (v_max - v_min) * (np.log10(l_max / l_min))

        # if n_integrals is 1, cut out the last axis so we just return a scalar
        if isinstance(bin_integrals, np.ndarray) and (bin_integrals.shape[-1] == 1):
            bin_integrals = bin_integrals[0]

        return bin_integrals


    # ---------- RADIO LUMINOSITY FUNCTION ----------
    def calculate_rlf(self, plot_rlf: bool = True):
        """
        Calculate the Radio Luminosity Function, either using the Page & Carrera 2000 method or the traditional 1/Vmax
        method
        """
        fluxes = self.fluxes
        luminosities = self.luminosities
        redshifts = self.redshifts
        resolved = self.resolved
        vmax_method = self.vmax_method
        self.logger.info("start " + ('1/Va' if vmax_method else 'P&C2000') + " rlf calculation")

        # reset phi in case it's not our first time calling this
        self.phi = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.counts = np.zeros((self.n_z_bins, self.n_lum_bins))

        if redshifts is None:
            # assign each source a comoving volume with a uniform dist s.t. dN/dV = const
            volumes = np.random.uniform(self.v_min, self.v_max, fluxes.shape[0])

            # conversion from comoving volume to redshift
            redshifts = z_from_v(volumes, *self.zvparams)

        # use the redshift to calculate luminosity distance and luminosity
        # because of errors on the margin, disregard passed luminosity
        luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
        luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 \
            / func.k_corr_factor(redshifts, spectral_index=self.spectral_index) # W/Hz

        if self.debug_flux_lum_relation:
            #ensure luminosities and redshifts are consistent with total flux
            # it's a lot easier to tell by visual inspection so save a scatterplot to debugdiff.png
            luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
            flux_luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 \
                / func.k_corr_factor(redshifts, spectral_index=self.spectral_index) # W/Hz
            self.logger.info('saving scatterplot to compare luminosity from flux to luminosity from catalog...')
            residuals = flux_luminosities / luminosities
            plt.scatter(flux_luminosities, residuals, s=0.001)
            #plt.yscale('log')
            epsilon = 1e-10
            plt.ylim(1-epsilon, 1+epsilon)
            plt.xscale('log')
            plt.title('Luminosity from flux vs catalog')
            plt.xlabel('luminosity from flux')
            plt.ylabel('luminosity from catalog / luminosity from flux')
            plt.grid()
            plt.savefig('debugdiff.png')
            plt.figure()
            self.logger.info('saved debug luminosity diff figure to debugdiff.png')

        # clip flux values to those above 0 for log plotting
        redshifts = redshifts[fluxes > 0]
        luminosities = luminosities[fluxes > 0]
        resolved = resolved[fluxes > 0]
        fluxes = fluxes[fluxes > 0]

        if vmax_method:
            for i_z in range(self.n_z_bins):
                z_min, z_max = self.z_bins[i_z], self.z_bins[i_z+1]
                v_min, v_max = self.cosmo.comoving_volume([z_min, z_max]).to(u.Mpc**3).value
                redshift_mask = (redshifts >= z_min) & (redshifts < z_max)
                self.logger.debug(f'{luminosities[redshift_mask].shape[0]} sources in z: {z_min}-{z_max}')

                for i_l in range(self.n_lum_bins):
                    l_min, l_max = self.l_bins[i_l], self.l_bins[i_l+1]

                    luminosity_mask = (luminosities >= l_min) & (luminosities < l_max)

                    luminosities_in_bin = luminosities[redshift_mask & luminosity_mask]
                    resolved_in_bin = resolved[redshift_mask & luminosity_mask]

                    # for n=0, the RLF should be 0 regardless, and also it breaks the code so just ignore it
                    if not luminosities_in_bin.size:
                        self.logger.debug(f'no sources in z: {z_min}-{z_max}, l={l_min}-{l_max}')
                        continue
                    self.logger.debug(f'{luminosities_in_bin.shape[0]} sources in z:'
                                      f' {z_min}-{z_max}, l={l_min}-{l_max}')

                    Vmaxs_resolved = self.monte_carlo_integral(v_min, v_max, l_min, l_max, resolved=True,
                                                               lum=luminosities_in_bin, vmax_method=True)
                    Vmaxs_unresolved = self.monte_carlo_integral(v_min, v_max, l_min, l_max, resolved=False,
                                                                 lum=luminosities_in_bin, vmax_method=True)
                    Vmaxs = np.where(resolved_in_bin, Vmaxs_resolved, Vmaxs_unresolved)

                    self.phi[i_z, i_l] = np.sum(1.0 / Vmaxs) #log bin size included in Vmaxs from monte_carlo_integral
                    self.counts[i_z, i_l] = luminosities_in_bin.shape[0]

                    # get errors from Page & Carrera 2000
                    self.phi_err[i_z, i_l] = np.sqrt(np.sum(1.0 / Vmaxs**2))

                self.logger.info(f'Redshift range {z_min:.2f}-{z_max:.2f} complete')

        # otherwise, do Page & Carrera 2000 method
        else:
            for i_z in range(self.n_z_bins):
                z_min = self.z_bins[i_z]
                z_max = self.z_bins[i_z+1]

                # find min & max of comoving volume for redshift bin
                v_min, v_max = self.cosmo.comoving_volume([z_min, z_max]).to(u.Mpc**3).value

                # get luminosity bins from offset indices
                # and make them (1,n_lum_bins) arrays for broadcasting with (n_sources,1)
                # luminosity bins are defined by their minimum value
                l_mins = self.l_bins[:-1][np.newaxis, :]
                l_maxs = self.l_bins[1:][np.newaxis, :]

                # now calculate the number of 'real' sources in each bin
                # masks have shape:
                #   redshift_mask: (n_sources, 1)
                #   luminosity_mask: (n_sources, n_lum_bins)
                redshift_mask = (redshifts[:, np.newaxis] >= z_min) & (redshifts[:, np.newaxis] < z_max)
                luminosity_mask = (luminosities[:, np.newaxis] >= l_mins) & (luminosities[:, np.newaxis] < l_maxs)

                # shape (n_lum_bins)
                n_sources_in_lum_bins = np.sum(redshift_mask & luminosity_mask, axis=0)
                n_resolved_in_lum_bins = np.sum(redshift_mask & luminosity_mask & resolved[:, np.newaxis], axis=0)
                n_unresolved_in_lum_bins = np.sum(redshift_mask & luminosity_mask & ~resolved[:, np.newaxis], axis=0)
                self.logger.debug(f"n_sources_in_lum_bins: {n_sources_in_lum_bins}")
                self.logger.debug(f"n_resolved_in_lum_bins: {n_resolved_in_lum_bins}")
                self.logger.debug(f"n_unresolved_in_lum_bins: {n_unresolved_in_lum_bins}")


                bin_integrals_resolved = self.monte_carlo_integral(v_min, v_max, l_mins, l_maxs, resolved=True)
                bin_integrals_unresolved = self.monte_carlo_integral(v_min, v_max, l_mins, l_maxs, resolved=False)
                #bin_integrals = self.simpson_integral(v_min, v_max, l_mins, l_maxs)

                self.logger.debug(f"bin integrals resolved: {bin_integrals_resolved}")
                self.logger.debug(f"bin integrals unresolved: {bin_integrals_unresolved}")

                # if we have a 0 bin integral but N > 0 it must be a monte carlo failure or completeness mismatch
                problematic_bins_resolved = (bin_integrals_resolved == 0) & (n_resolved_in_lum_bins > 0)
                problematic_bins_unresolved = (bin_integrals_unresolved == 0) & (n_unresolved_in_lum_bins > 0)
                if np.any(problematic_bins_resolved | problematic_bins_unresolved):
                    self.logger.error(f"Monte Carlo failure - {self.n_mc_pts} points insufficient for number of bins")
                    if np.any(problematic_bins_resolved):
                        pbr_indices = np.nonzero(problematic_bins_resolved)[0]
                        if pbr_indices.shape[0] == 1:
                            index = pbr_indices[0]
                            self.logger.error(f'Resolved bin {index} had {n_resolved_in_lum_bins[index]} sources but a 0 bin integral')
                            max_flux = self.flux_from_coordinate(v_min, l_maxs[0, index])
                            min_flux = self.flux_from_coordinate(v_max, l_mins[0, index])
                            self.logger.error(f'Min flux in bin {min_flux}, max flux in bin {max_flux}, cutoff {self.flux_cut_jy}')
                        else:
                            self.logger.error(f'{pbr_indices.shape[0]} resolved bins had sources but a 0 bin integral, indices {pbr_indices}')
                    if np.any(problematic_bins_unresolved):
                        pbu_indices = np.nonzero(problematic_bins_unresolved)[0]
                        if pbu_indices.shape[0] == 1:
                            index = pbu_indices[0]
                            self.logger.error(f'Unresolved bin {index} had {n_unresolved_in_lum_bins[index]} sources but a 0 bin integral')
                            max_flux = self.flux_from_coordinate(v_min, l_maxs[0, index])
                            min_flux = self.flux_from_coordinate(v_max, l_mins[0, index])
                            self.logger.error(f'Min flux in bin {min_flux}, max flux in bin {max_flux}, cutoff {self.flux_cut_jy}')
                        else:
                            self.logger.error(f'{pbu_indices.shape[0]} resolved bins had sources but a 0 bin integral, indices {pbu_indices}')


                bin_integrals_unresolved[n_unresolved_in_lum_bins == 0] = 1
                bin_integrals_resolved[n_resolved_in_lum_bins == 0] = 1

                # now we have phi_est as given by Page & Carrera 2000
                self.phi[i_z] = n_unresolved_in_lum_bins / bin_integrals_unresolved + n_resolved_in_lum_bins / bin_integrals_resolved
                self.counts[i_z] = n_sources_in_lum_bins

                # get errors from poisson statistics
                phi_err_range_resolved = astropy.stats.poisson_conf_interval(n_resolved_in_lum_bins) / bin_integrals_resolved
                phi_err_range_unresolved = astropy.stats.poisson_conf_interval(n_unresolved_in_lum_bins) / bin_integrals_unresolved
                phi_err_resolved = np.abs(phi_err_range_resolved[1] - phi_err_range_resolved[0]) / 2
                phi_err_unresolved = np.abs(phi_err_range_unresolved[1] - phi_err_range_unresolved[0]) / 2
               
                phi_err_resolved[n_resolved_in_lum_bins == 0] = 0
                phi_err_unresolved[n_unresolved_in_lum_bins == 0] = 0

                self.logger.debug(f"phi err resolved: {phi_err_resolved}")
                self.logger.debug(f"phi err unresolved: {phi_err_unresolved}")
                self.phi_err[i_z] = np.sqrt(phi_err_resolved**2 + phi_err_unresolved**2 + (0.05*self.phi[i_z])**2)

                self.logger.info(f'Redshift range {z_min:.2f}-{z_max:.2f} complete')

        # sky coverage completeness factor
        self.phi /= 5700 / 41253 # 5700 lotss dr2 area from hardcastle et al. 2023, 41253 deg^2 is solid angle of a sphere
        self.phi_err /= 5700 / 41253


        # fit parameters to the RLFs
        self.fit_rlf_individually()

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
            self.plot_rlf(title, colors, output=output)


    def fit_rlf_individually(self):
        """
        Fit a dual power law to the RLFs in each redshift bin individually, using the function rlf_power_law. The
        parameters are fitted using scipy's curve_fit function, with initial guesses and bounds provided. The fitted
        parameters and their errors are stored in self.rlf_fit_params, and the results are logged.
        """
        self.logger.info('Fitting Parameters to RLFs')

        # fit a dual power law to each redshift RLF
        bin_centres = (self.l_bins[:-1] + self.l_bins[1:]) / 2

        # (4,2) being 4 parameters, with [:, 0] being values and [:, 1] being errors
        self.rlf_fit_params = np.zeros((self.n_z_bins, 4, 2)) 
        self.chi_sqr = np.zeros((self.n_z_bins)) 

        for i_z in range(self.n_z_bins):
            fit_bin_centres = bin_centres[self.phi[i_z] > 0]
            fit_phi = self.phi[i_z][self.phi[i_z] > 0]
            fit_phi_err = self.phi_err[i_z][self.phi[i_z] > 0]
            if self.vmax_method:
                fit_bin_centres = fit_bin_centres[1:]
                fit_phi = fit_phi[1:]
                fit_phi_err = fit_phi_err[1:]
            popt, pcov = curve_fit(func.rlf_power_law,
                                    fit_bin_centres,
                                    fit_phi,
                                    p0=[0.5, 1.5, -5.5, 26],
                                    bounds=([0, 1, -10, 20], [1, 4, -1, 30]),
                                    absolute_sigma=True,
                                    sigma=fit_phi_err,
                                    maxfev=1000000)
            perr = np.sqrt(np.diag(pcov))
            self.rlf_fit_params[i_z] = np.array([popt, perr]).T
            self.logger.info(f'z={self.z_bins[i_z]}-{self.z_bins[i_z+1]}: ')
            self.logger.info(f'    alpha={popt[0]:.3f} +/- {perr[0]:.3f}')
            self.logger.info(f'    beta={popt[1]:.3f} +/- {perr[1]:.3f}')
            self.logger.info(f'    Log10C={popt[2]:.3f} +/- {perr[2]:.3f}')
            self.logger.info(f'    Log10Lstar={popt[3]:.3f} +/- {perr[3]:.3f}')

            model_values = func.rlf_power_law(bin_centres[self.phi[i_z] > 0], *popt)
            residuals = self.phi[i_z][self.phi[i_z] > 0] - model_values
            self.chi_sqr[i_z] = np.sum((residuals / self.phi_err[i_z][self.phi[i_z] > 0])**2)
            self.chi_sqr[i_z] /= residuals.shape[0] - 4
            self.logger.info(f'Reduced chi squared: {self.chi_sqr[i_z]}')
        self.chi_sqr_tot = np.sum(self.chi_sqr)
        self.logger.info(f'Reduced chi squared total: {self.chi_sqr_tot}')


    def fit_rlf(self):
        """
        Fit a dual power law to the RLFs across all redshift bins, using the function rlf_power_law_evolution. The
        parameters are fitted using scipy's curve_fit function, with initial guesses and bounds provided. The fitted
        parameters and their errors are stored in self.rlf_fit_params, and the results are logged.
        """
        self.logger.info('Fitting Parameters to RLFs')

        # fit a dual power law to each redshift RLF
        l_bin_centres = (self.l_bins[:-1] + self.l_bins[1:]) / 2
        z_bin_centres = (self.z_bins[:-1] + self.z_bins[1:]) / 2

        ydata = self.phi.ravel()
        L, Z = np.meshgrid(l_bin_centres, z_bin_centres)
        L = L.ravel()[ydata > 0]
        Z = Z.ravel()[ydata > 0]
        xdata = np.vstack((L, Z))
        yerr = self.phi_err.ravel()

        yerr = yerr[ydata > 0]
        ydata = ydata[ydata > 0]

        # (4,2) being 4 parameters, with [:, 0] being values and [:, 1] being errors

        p0_yuan=[0.5, 1.5, -5.5, 24.59, 1, 1, 1, 1]
        bounds_yuan=([0, 1, -10, 20, -5, -5, 0.01, 0], [1, 2.5, -1, 30, 5, 5, 1, 5])
        p0_yuan2018=      [ 0.31, -5.92, 0.86, -4.85, 24.68, 0.44, 0.31, 4.73]
        bounds_yuan2018=([-30,    -10,   0,    -10,   20,    0,    0,   -10], 
                          [10,     10,   5,    -1,    28,    10,   6,    10])

        #p0_powerlaw = [0.5, 1.5, -5.5, 26, 0]
        p0_powerlaw = [0.5, 1.5, -5.5, 26, 0, 0]
        #bounds_powerlaw = ([0, 1, -10, 20, -100], [1, 4, -1, 30, 100])
        bounds_powerlaw = ([0, 1, -10, 20, -100, -100], [1, 4, -1, 30, 100, 100])
        

        #fn_powerlaw = functions.rlf_pde if self.use_pde else functions.rlf_ple
        fn_powerlaw = func.rlf_power_law_evolution

        self.rlf_fit_params = np.zeros((len(p0_powerlaw), 2)) 

        popt, pcov = curve_fit(fn_powerlaw,
                                xdata,
                                ydata,
                                p0=p0_powerlaw,
                                bounds=bounds_powerlaw,
                                absolute_sigma=True,
                                sigma=yerr,
                                maxfev=1000000)
        perr = np.sqrt(np.diag(pcov))
        self.rlf_fit_params = np.array([popt, perr]).T

        param_names_yuan2018 = ['p1', 'p2', 'zc', 'Log10Phi', 'Log10Lstar', 'beta', 'gamma', 'k1']
        param_names_yuan = ['alpha', 'beta', 'Log10C', 'Log10Lstar', 'm', 'z0', 'zsigma', 'k1']
        param_names_powerlaw = ['alpha', 'beta', 'Log10C', 'Log10Lstar', 'alphaD', 'alphaL']
        #if self.use_pde:
        #    param_names_powerlaw.append('alphaD')
        #else:
        #    param_names_powerlaw.append('alphaL')


        for param_name, popt_i, perr_i in zip(param_names_powerlaw, popt, perr):
            self.logger.info(f'{param_name}={popt_i:.3f} +/- {perr_i:.3f}')


    def plot_rlf(self, 
                  title: str,
                  colors: list,
                  ax = None,
                  ylim: tuple | None = [1e-9, 3e-4],
                  xlim: tuple | None = [1e21, 1e29],
                  output: str = 'rlf.png',
                  draw_ylabel: bool = True):
        self_contained = ax is None
        if self_contained:
            self.logger.info("plotting self contained rlf")
            fig, ax = plt.subplots(figsize=(10,10))

        luminosity_space = np.geomspace(self.l_bins[0], self.l_bins[-1], num=100)

        bin_centres = (self.l_bins[:-1] + self.l_bins[1:]) / 2
        z_bin_centres = (self.z_bins[:-1] + self.z_bins[1:]) / 2
        for i_z in range(self.phi.shape[0]):

            if self.rlf_fit_params.ndim == 3:
                fit_params = self.rlf_fit_params[i_z, :, 0]
                fitted_rlf = func.rlf_power_law(luminosity_space, *fit_params)
            else:
                redshift = bin_centres[i_z]
                redshift_space = np.repeat(redshift, luminosity_space.shape[0])
                fit_params = self.rlf_fit_params[:, 0]
                #fn_powerlaw = func.rlf_pde if self.use_pde else func.rlf_ple
                fn_powerlaw = func.rlf_power_law_evolution
                fitted_rlf = fn_powerlaw((luminosity_space, z_bin_centres[i_z]), *fit_params)
                #self.logger.debug(fitted_rlf)

            ax.plot(luminosity_space, fitted_rlf, color=colors[i_z])

            specific_phi = self.phi[i_z]
            specific_counts = self.counts[i_z]
            mask = specific_counts >= 1
            ax.plot(bin_centres[mask], specific_phi[mask], color=colors[i_z],
                     marker='o', linestyle='None',
                     label=f'{self.z_bins[i_z]:.2f}<z<{self.z_bins[i_z + 1]:.2f}')
            ax.errorbar(bin_centres[mask], specific_phi[mask], yerr=self.phi_err[i_z][mask], color=colors[i_z], fmt='none')
        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_xlabel('$L_{144}$  $(\\frac{W}{Hz})$')
        ax.set_yscale('log')
        if draw_ylabel:
            ax.set_ylabel('$\\phi_{est}$  $(Mpc^{-3} (\\log_{10}(\\frac{W}{m^2}))^{-1})$')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid()
        ax.tick_params(which='both', right=True, top=True)
        ax.legend()
        if self_contained:
            plt.savefig(output)
            self.logger.info(f"saved figure to {output}")

