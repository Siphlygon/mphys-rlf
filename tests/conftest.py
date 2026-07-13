"""
Shared pytest fixtures for the diffracc test suite.

Several classes in diffracc (notably RLF) read a live config.ini and other on-disk files at construction time. Rather
than depend on the real diffracc/config.ini (which is tuned for full-scale runs - e.g. N_MC_PTS = 100000 - and would
make unit tests slow), these fixtures write small temp files and monkeypatch diffracc.utils.paths so the code under
test believes it's reading the real thing.
"""
import astropy.cosmology
import astropy.units as u
import numpy as np
import pytest

from diffracc.utils import paths


@pytest.fixture
def flat_lcdm_cosmo() -> astropy.cosmology.FlatLambdaCDM:
    """A small, fixed FlatLambdaCDM cosmology (matches diffracc/config.ini's h/Tcmb0/Om0) for reuse across tests."""
    return astropy.cosmology.FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)


@pytest.fixture
def rlf_config_path(tmp_path):
    """
    Write a minimal config.ini [DEFAULT] section containing only the keys RLF.__init__ reads, with a small
    N_MC_PTS/LUM_BINS so tests run fast, and return its path.
    """
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[DEFAULT]\n"
        "dz = 0.5\n"
        "LUM_BINS = 9\n"
        "N_INTERP_PTS = 200\n"
        "N_MC_PTS = 20000\n"
        "SPECTRAL_INDEX = -0.7\n"
        "Z_MIN = 0.01\n"
        "Z_MAX = 1.0\n"
        "L_MIN = 21\n"
        "L_MAX = 29\n"
        "HARDCASTLE_Z_BINS = False\n"
        "DEJONG_Z_BINS = False\n",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def completeness_args_path(tmp_path):
    """
    Write a temp completeness_args_sigmoid.txt (x0, k, a, b for utils.functions.sigmoid) and return its path.
    x0=0 (log10(mJy)=0, i.e. 1 mJy), k=5 (fairly sharp transition), a=1, b=0 -> a completeness curve rising from 0 to 1
    around 1 mJy, which is a realistic-shaped stand-in for the real fitted values.
    """
    args_path = tmp_path / "completeness_args_sigmoid.txt"
    np.savetxt(args_path, np.array([0.0, 5.0, 1.0, 0.0]))
    return args_path


@pytest.fixture
def rlf_factory(monkeypatch, rlf_config_path, completeness_args_path, flat_lcdm_cosmo):
    """
    Returns a factory function that builds an RLF instance against the temp config/completeness fixtures, so tests
    only need to supply the source arrays (and any RLF.__init__ kwarg overrides) that matter for what they're
    checking.
    """
    from diffracc.rlf.rlf import RLF

    monkeypatch.setattr(paths, "PROGRAM_CONFIG", rlf_config_path)

    def _make_rlf(fluxes: np.ndarray,
                 redshifts: np.ndarray,
                 luminosities: np.ndarray,
                 resolved: np.ndarray,
                 **kwargs) -> RLF:
        kwargs.setdefault("completeness_path", completeness_args_path)
        kwargs.setdefault("cosmo", flat_lcdm_cosmo)
        return RLF(fluxes, redshifts, luminosities, resolved, **kwargs)

    return _make_rlf
