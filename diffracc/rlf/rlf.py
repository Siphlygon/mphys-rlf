"""
A module for calculating the radio luminosity function (RLF) of a sample of AGN using either the Page & Carrera 2000
method or the traditional 1/Vmax method. The RLF is calculated in bins of redshift and luminosity, and can be corrected
for completeness using a fitted completeness function or a step function. Also fits the resulting RLFs with a double
power law function and stores the fit parameters for each redshift bin.
"""

import configparser
from pathlib import Path
from typing import Callable

import astropy.cosmology
import astropy.stats
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ..completeness.completeness_io import read_completeness_fit
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
        self.logger = get_logger("RLF", LoggingLevels.INFO.value)

        # init parameters
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
            completeness_path = paths.NP_ARRAY_PARENT / 'completeness_args.json'
        if isinstance(completeness_path, str):
            completeness_path = Path(completeness_path)

        # Completeness is not optional for the RLF, so a missing or unreadable fit is fatal rather than defaulted. The
        # fit carries its own function and x-axis and evaluates itself; this class deliberately never decides whether
        # to log10 a flux before handing it over.
        try:
            self.completeness_fit = read_completeness_fit(completeness_path)
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f'Could not load completeness fit from {completeness_path}: {e}')
            raise
        self.logger.info(f'Loaded {self.completeness_fit.function_name} completeness fit from {completeness_path} '
                         f'(x_space={self.completeness_fit.x_space}, popt={self.completeness_fit.popt})')

        # Read parameters from the config.ini file
        config = configparser.ConfigParser()
        config.read(paths.PROGRAM_CONFIG)
        default_config = config['DEFAULT']

        # RLF parameters
        # redshift bin width
        self.dz = float(default_config['dz'])
        # number of luminosity bins between min and max luminosity
        self.lum_bins_count = int(default_config['LUM_BINS'])
        # number of points to use in interpolation approximation of
        self.n_interp_pts = int(default_config['N_INTERP_PTS'])
        # number of points to use in the monte-carlo integral for each redshift-luminosity bin (Page & Carrera 2000)
        self.n_mc_pts_pc = int(default_config['N_MC_PTS_PC'])
        # number of points to use in the monte-carlo integral for each source in the 1/Vmax calculation
        self.n_mc_pts_vmax = int(default_config['N_MC_PTS_VMAX'])
        # spectral index to use for the k-correction, typically -0.7 for AGN
        self.spectral_index = float(default_config['SPECTRAL_INDEX'])
        # maximum Z (redshift) to consider in RLF calculation
        self.z_max = float(default_config['Z_MAX'])
        # minimum Z (redshift) to consider in RLF calculation
        self.z_min = float(default_config['Z_MIN'])
        # maximum luminosity to plot, to the power 10 (max lum = 10**L_MAX)
        self.l_max = float(default_config['L_MAX'])
        # minimum luminosity to plot, to the power 10 (min lum = 10**L_MIN)
        self.l_min = float(default_config['L_MIN'])

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
    def _get_completeness(self,
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
        # If resolved is a single boolean value, we can just return the completeness for that case without needing to
        # evaluate both resolved and unresolved completeness for each source.
        if np.ndim(resolved) == 0:
            if resolved:
                completeness = self.completeness_fit.evaluate(integ_fluxes * 1000, s0_shift_mjy=self.bias)
            elif self.use_shimwell:
                completeness = np.interp(integ_fluxes, shimwell_data[0] / 1000, shimwell_data[1])
            else:
                completeness = np.ones_like(integ_fluxes)
            return np.where(integ_fluxes > self.flux_cut_jy, completeness, 0)

        # Otherwise fall back to evluating both resolved and unresolved completeness for each source, and returning the
        # appropriate value based on the resolved array.
        func_completeness = self.completeness_fit.evaluate(integ_fluxes * 1000, s0_shift_mjy=self.bias)
        resolved_completeness = np.where(integ_fluxes > self.flux_cut_jy, func_completeness, 0)

        # Use Shimwell et al. (2023) completeness for unresolvedd sources if use_shimwell is True, otherwise use a step
        # function at the flux_cut_jy threshold.
        if self.use_shimwell:
            shimwell_completeness = np.interp(integ_fluxes, shimwell_data[0] / 1000, shimwell_data[1])
            unresolved_completeness = np.where(integ_fluxes > self.flux_cut_jy, shimwell_completeness, 0)
        else:
            unresolved_completeness = np.where(integ_fluxes > self.flux_cut_jy, 1, 0)

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
        return self._get_completeness(self.flux_from_coordinate(v, l), resolved=resolved)


    def flux_from_coordinate(self,
                             vols: float | np.ndarray,
                             lums: float | np.ndarray,
                             reds: float | np.ndarray | None = None) -> np.ndarray:
        """
        Generate luminosities + redshifts -> fluxes. Flux values here are in Jy, luminosities in W/Hz
        
        Parameters
        ----------
        vols : float | np.ndarray
            The volume(s) to compute the flux for
        lums : float | np.ndarray
            The luminosity/luminosities to compute the flux for
        reds : float | np.ndarray | None, optional
            The redshift(s) to compute the flux for, by default None. If None, the redshift is computed from the volume
            
        Returns
        -------
        np.ndarray
            The fluxes corresponding to the input volumes and luminosities
        """
        if reds is None:
            reds = z_from_v(vols, *self.zvparams)

        # Find the luminosity distance & convert into flux with a k-correction
        d_l = self.cosmo.luminosity_distance(reds).to(u.m).value
        s = 1e26 * lums / (4 * np.pi * d_l**2) * func.k_corr_factor(reds, spectral_index = self.spectral_index)
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
        mc_pts = self.n_mc_pts_vmax if vmax_method else self.n_mc_pts_pc

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
        bin_integrals = np.sum(self.get_completeness_from_coord(random_volumes,
                                                                lums,
                                                                resolved=resolved), axis=0) / mc_pts

        # divide by the log luminosity-volume bin area so the result is / MPc^3 / log10(W/Hz)
        if isinstance(l_min, np.ndarray) and isinstance(l_max, np.ndarray):
            bin_integrals *= (v_max - v_min) * (np.log10(l_max[0, :] / l_min[0, :]))
        else:
            bin_integrals *= (v_max - v_min) * (np.log10(l_max / l_min))

        # if n_integrals is 1, cut out the last axis so we just return a scalar
        if isinstance(bin_integrals, np.ndarray) and (bin_integrals.shape[-1] == 1):
            bin_integrals = bin_integrals[0]

        return bin_integrals


    # ---------- UTILITIES ----------
    def _preprocess_sources(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive luminosities from flux and redshift (the catalogue's own luminosity column is always recomputed to
        avoid inconsistencies at the margin), optionally save a debug comparison plot, and drop sources with
        non-positive flux or a non-physical redshift.

        Returns
        -------
        redshifts : np.ndarray
            The array of redshifts of the AGN
        luminosities : np.ndarray
            The array of luminosities of the AGN
        resolved : np.ndarray
            The array indicating whether each AGN is resolved
        """
        fluxes = self.fluxes
        redshifts = self.redshifts
        resolved = self.resolved

        if redshifts is None:
            # assign each source a comoving volume with a uniform dist s.t. dN/dV = const
            volumes = np.random.uniform(self.v_min, self.v_max, fluxes.shape[0])

            # conversion from comoving volume to redshift
            redshifts = z_from_v(volumes, *self.zvparams)

        # Drop sources with non-positive flux or a non-physical redshift (<=0, including catalogue placeholder values
        # such as z=-1 for "missing") before running the redshift-dependent cosmology calculation below. Every z_bin
        # starts at self.z_min > 0, so such sources could never land in a redshift bin anyway -- excluding them here is
        # a no-op for the final RLF, and avoids feeding invalid input into astropy's comoving-distance integral and the
        # k-correction (which previously produced RuntimeWarnings, and at z=-1 exactly, a literal division by zero).
        mask = (fluxes > 0) & np.isfinite(redshifts) & (redshifts > 0)
        fluxes = fluxes[mask]
        redshifts = redshifts[mask]
        resolved = resolved[mask]

        # use the redshift to calculate luminosity distance and luminosity
        # recalculate luminosities from fluxes and redshifts to avoid inconsistencies at the margin
        luminosity_distances = self.cosmo.luminosity_distance(redshifts).to(u.m).value
        luminosities = 4 * np.pi * 1e-26 * fluxes * luminosity_distances**2 \
            / func.k_corr_factor(redshifts, spectral_index=self.spectral_index) # W/Hz

        return redshifts, luminosities, resolved


    def _warn_on_zero_bin_integrals(self,
                                    v_min: float,
                                    v_max: float,
                                    l_mins: np.ndarray,
                                    l_maxs: np.ndarray,
                                    bin_integs_res: np.ndarray,
                                    bin_integs_unres: np.ndarray,
                                    n_res_in_lum_bins: np.ndarray,
                                    n_unres_in_lum_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Log an error for any bin that has sources but a 0 Monte Carlo integral, which indicates n_mc_pts is too low for
        the number of bins used (or a completeness/flux_cut_jy mismatch), and return masks identifying those bins so the
        caller can mark their phi estimate as undefined instead of dividing by zero.

        Parameters
        ----------
        v_min : float
            The minimum volume of the redshift bin
        v_max : float
            The maximum volume of the redshift bin
        l_mins : np.ndarray
            The minimum luminosities of the luminosity bins
        l_maxs : np.ndarray
            The maximum luminosities of the luminosity bins
        bin_integs_res : np.ndarray
            The Monte Carlo integrals for the resolved sources in each luminosity bin
        bin_integs_unres : np.ndarray
            The Monte Carlo integrals for the unresolved sources in each luminosity bin
        n_res_in_lum_bins : np.ndarray
            The number of resolved sources in each luminosity bin
        n_unres_in_lum_bins : np.ndarray
            The number of unresolved sources in each luminosity bin

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Boolean masks of shape (n_lum_bins,), for resolved and unresolved respectively, marking luminosity bins that
            have real sources but a zero completeness-weighted Monte Carlo integral. Such bins have no well-defined phi
            estimate (dividing a nonzero count by a zero integral gives +inf) and should be treated as undefined (NaN)
            by the caller.
        """
        categories = [
            ('resolved', bin_integs_res, n_res_in_lum_bins),
            ('unresolved', bin_integs_unres, n_unres_in_lum_bins),
        ]
        problematic = {name: (integrals == 0) & (counts > 0) for name, integrals, counts in categories}
        if not any(np.any(mask) for mask in problematic.values()):
            return problematic['resolved'], problematic['unresolved']

        self.logger.error(f"Monte Carlo failure - {self.n_mc_pts} points insufficient for number of bins")
        for name, _, counts in categories:
            mask = problematic[name]
            if not np.any(mask):
                continue
            indices = np.nonzero(mask)[0]
            if indices.shape[0] == 1:
                index = indices[0]
                self.logger.error(f'{name.capitalize()} bin {index} had {counts[index]} sources but a 0 bin integral')
                max_flux = self.flux_from_coordinate(v_min, l_maxs[0, index])
                min_flux = self.flux_from_coordinate(v_max, l_mins[0, index])
                self.logger.error(f'Min flux in bin {min_flux}, max flux in bin {max_flux}, cutoff {self.flux_cut_jy}')
            else:
                self.logger.error(f'{indices.shape[0]} {name} bins had sources but a 0 bin integral, indices {indices}')

        return problematic['resolved'], problematic['unresolved']


    # ---------- RLF ESTIMATION ----------
    def calculate_rlf(self, plot_rlf: bool = True) -> None:
        """
        Calculate the Radio Luminosity Function, either using the Page & Carrera 2000 method or the traditional 1/Vmax
        method
        
        Parameters
        ----------
        plot_rlf : bool, optional
            Whether to plot the resulting RLF, by default True
        """
        self.logger.info(f"Starting '{'1/Va' if self.vmax_method else 'P&C2000'}' RLF calculation")

        # reset phi in case it's not our first time calling this
        self.phi = np.zeros((self.n_z_bins, self.n_lum_bins))
        self.counts = np.zeros((self.n_z_bins, self.n_lum_bins))

        redshifts, luminosities, resolved = self._preprocess_sources()

        if self.vmax_method:
            self._calculate_rlf_vmax(redshifts, luminosities, resolved)
        else:
            self._calculate_rlf_page_carrera(redshifts, luminosities, resolved)

        # sky coverage completeness factor
        # 5700 lotss dr2 area from hardcastle et al. 2023, 41253 deg^2 is solid angle of a sphere
        self.phi /= 5700 / 41253
        self.phi_err /= 5700 / 41253

        # fit parameters to the RLFs
        self.fit_rlf_individually()

        if plot_rlf:
            # plot the resulting graph
            if self.vmax_method:
                title = f'1/Va RLF - {self.n_mc_pts // 10} pts per source'
                output = 'rlf_vmax.png'
            else:
                title = f'Page & Carrera RLF - {self.n_mc_pts} pts per bin'
                output = 'rlf_page_and_carrera.png'
            self.plot_rlf(title, colors, output=output)


    def _calculate_rlf_vmax(self, redshifts: np.ndarray, luminosities: np.ndarray, resolved: np.ndarray) -> None:
        """
        Populate self.phi, self.counts and self.phi_err using the traditional 1/Vmax method: bin sources by
        redshift and luminosity, and sum 1/Vmax within each bin.
        
        Parameters
        ----------
        redshifts : np.ndarray
            The array of redshifts of the AGN
        luminosities : np.ndarray
            The array of luminosities of the AGN
        resolved : np.ndarray
            The array indicating whether each AGN is resolved
        """
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


    def _calculate_rlf_page_carrera(self,
                                    redshifts: np.ndarray,
                                    luminosities: np.ndarray,
                                    resolved: np.ndarray) -> None:
        """
        Populate self.phi, self.counts and self.phi_err using the Page & Carrera (2000) method: estimate phi per
        bin from source counts divided by completeness-weighted volume-luminosity Monte Carlo integrals.
        
        Parameters
        ----------
        redshifts : np.ndarray
            The array of redshifts of the AGN
        luminosities : np.ndarray
            The array of luminosities of the AGN
        resolved : np.ndarray
            The array indicating whether each AGN is resolved
        """
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
            n_res_in_lum_bins = np.sum(redshift_mask & luminosity_mask & resolved[:, np.newaxis], axis=0)
            n_unres_in_lum_bins = np.sum(redshift_mask & luminosity_mask & ~resolved[:, np.newaxis], axis=0)
            self.logger.debug(f"n_sources_in_lum_bins: {n_sources_in_lum_bins}")
            self.logger.debug(f"n_resolved_in_lum_bins: {n_res_in_lum_bins}")
            self.logger.debug(f"n_unresolved_in_lum_bins: {n_unres_in_lum_bins}")

            bin_int_res = self.monte_carlo_integral(v_min, v_max, l_mins, l_maxs, resolved=True)
            bin_int_unres = self.monte_carlo_integral(v_min, v_max, l_mins, l_maxs, resolved=False)

            self.logger.debug(f"bin integrals resolved: {bin_int_res}")
            self.logger.debug(f"bin integrals unresolved: {bin_int_unres}")

            impossible_res, impossible_unres = self._warn_on_zero_bin_integrals(
                v_min, v_max, l_mins, l_maxs,
                bin_int_res, bin_int_unres,
                n_res_in_lum_bins, n_unres_in_lum_bins)

            bin_int_unres[n_unres_in_lum_bins == 0] = 1
            bin_int_res[n_res_in_lum_bins == 0] = 1

            # For "impossible" bins (real sources, zero completeness-weighted integral -- already reported by
            # _warn_on_zero_bin_integrals above), the divisions below are 0-count-safe (count/1 above) except at
            # exactly these bins, where they hit a genuine x/0. That is expected and immediately overwritten with
            # NaN below, so the resulting RuntimeWarnings are suppressed here rather than left to clutter output.
            with np.errstate(divide='ignore', invalid='ignore'):
                # now we have phi_est as given by Page & Carrera 2000
                self.phi[i_z] = n_unres_in_lum_bins / bin_int_unres + n_res_in_lum_bins / bin_int_res
                self.counts[i_z] = n_sources_in_lum_bins

                # get errors from poisson statistics
                phi_err_range_res = astropy.stats.poisson_conf_interval(n_res_in_lum_bins) / bin_int_res
                phi_err_range_unres = astropy.stats.poisson_conf_interval(n_unres_in_lum_bins) / bin_int_unres
                phi_err_res = np.abs(phi_err_range_res[1] - phi_err_range_res[0]) / 2
                phi_err_unres = np.abs(phi_err_range_unres[1] - phi_err_range_unres[0]) / 2

                phi_err_res[n_res_in_lum_bins == 0] = 0
                phi_err_unres[n_unres_in_lum_bins == 0] = 0

                self.logger.debug(f"phi err resolved: {phi_err_res}")
                self.logger.debug(f"phi err unresolved: {phi_err_unres}")
                self.phi_err[i_z] = np.sqrt(phi_err_res**2 + phi_err_unres**2 + (0.05*self.phi[i_z])**2)

            # Bins with real sources but a geometrically zero completeness-weighted integral (e.g. flux_cut_jy set above
            # the brightest flux physically reachable in that bin -- see _warn_on_zero_bin_integrals) have no
            # well-defined phi estimate. Mark them NaN instead of leaving the +inf that dividing by the untouched zero
            # integral above would otherwise silently carry into self.phi/self.phi_err and any downstream sum, plot, or
            # fit.
            impossible = impossible_res | impossible_unres
            self.phi[i_z, impossible] = np.nan
            self.phi_err[i_z, impossible] = np.nan

            self.logger.info(f'Redshift range {z_min:.2f}-{z_max:.2f} complete')


    # ---------- FITTING & PLOTTING ----------
    def _fit_and_log_params(self,
                            model_func: Callable,
                            xdata: np.ndarray,
                            ydata: np.ndarray,
                            yerr: np.ndarray,
                            p0: list,
                            bounds: tuple,
                            param_names: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Run scipy's curve_fit with this project's standard fit settings (absolute_sigma, high maxfev) and log each
        fitted parameter with its uncertainty.
        
        Parameters
        ----------
        model_func : Callable
            The model function to fit to the data
        xdata : np.ndarray
            The x data to fit the model to
        ydata : np.ndarray
            The y data to fit the model to
        yerr : np.ndarray
            The uncertainties in the y data
        p0 : list
            The initial guess for the parameters
        bounds : tuple
            The bounds for the parameters, as a tuple of (lower_bounds, upper_bounds)
        param_names : list
            The names of the parameters, for logging purposes

        Returns
        -------
        tuple
            (popt, perr) - the fitted parameter values and their 1-sigma uncertainties.
        """
        popt, pcov = curve_fit(model_func, xdata, ydata,
                               p0=p0, bounds=bounds,
                               absolute_sigma=True, sigma=yerr,
                               maxfev=1000000)
        perr = np.sqrt(np.diag(pcov))
        for param_name, value, err in zip(param_names, popt, perr):
            self.logger.info(f'    {param_name}={value:.3f} +/- {err:.3f}')
        return popt, perr


    def fit_rlf_individually(self):
        """
        Fit a dual power law to the RLFs in each redshift bin individually, using the function rlf_power_law. The
        parameters are fitted using scipy's curve_fit function, with initial guesses and bounds provided. The fitted
        parameters and their errors are stored in self.rlf_fit_params, and the results are logged.
        """
        self.logger.info('Fitting Parameters to RLFs')

        # fit a dual power law to each redshift RLF
        bin_centres = (self.l_bins[:-1] + self.l_bins[1:]) / 2
        param_names = ['alpha', 'beta', 'Log10C', 'Log10Lstar']

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

            self.logger.info(f'z={self.z_bins[i_z]}-{self.z_bins[i_z+1]}: ')
            popt, perr = self._fit_and_log_params(func.rlf_power_law, fit_bin_centres, fit_phi, fit_phi_err,
                                                   p0=[0.5, 1.5, -5.5, 26],
                                                   bounds=([0, 1, -10, 20], [1, 4, -1, 30]),
                                                   param_names=param_names)
            self.rlf_fit_params[i_z] = np.array([popt, perr]).T

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
        yerr = self.phi_err.ravel()[ydata > 0]
        ydata = ydata[ydata > 0]

        p0_powerlaw = [0.5, 1.5, -5.5, 26, 0, 0]
        bounds_powerlaw = ([0, 1, -10, 20, -100, -100], [1, 4, -1, 30, 100, 100])
        param_names_powerlaw = ['alpha', 'beta', 'Log10C', 'Log10Lstar', 'alphaD', 'alphaL']

        # (n_params,2) being fitted values, [:, 0], and errors, [:, 1]
        popt, perr = self._fit_and_log_params(func.rlf_power_law_evolution, xdata, ydata, yerr,
                                               p0=p0_powerlaw, bounds=bounds_powerlaw,
                                               param_names=param_names_powerlaw)
        self.rlf_fit_params = np.array([popt, perr]).T


    def plot_rlf(self,
                 title: str,
                 colors: list,
                 ax = None,
                 ylim: tuple | None = (1e-9, 3e-4),
                 xlim: tuple | None = (1e21, 1e29),
                 output: str = 'rlf.png',
                 draw_ylabel: bool = True):
        """
        Plot the Radio Luminosity Function (RLF) for each redshift bin, along with the fitted dual power law model.

        Parameters
        ----------
        title : str
            The title of the plot
        colors : list
            The colors to use for plotting
        ax : _type_, optional
            The axes object to plot on, by default None
        ylim : tuple | None, optional
            The y-axis limits, by default (1e-9, 3e-4)
        xlim : tuple | None, optional
            The x-axis limits, by default (1e21, 1e29)
        output : str, optional
            The output file name, by default 'rlf.png'
        draw_ylabel : bool, optional
            Whether to draw the y-axis label, by default True
        """
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
            ax.errorbar(bin_centres[mask], specific_phi[mask], yerr=self.phi_err[i_z][mask],
                        color=colors[i_z], fmt='none')
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
