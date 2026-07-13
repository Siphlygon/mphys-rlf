"""
Shared pytest fixtures for the diffracc test suite.

Several classes in diffracc read a live config.ini and other on-disk files at construction time. Rather than depend on
the real diffracc/config.ini (which is tuned for full-scale runs - e.g. N_MC_PTS = 100000 - and would make unit tests
slow), these fixtures write small temp files and monkeypatch diffracc.utils.paths so the code under test believes it's
reading the real thing.
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


@pytest.fixture
def np_array_parent(tmp_path, monkeypatch):
    """Fixture to set the NP_ARRAY_PARENT path to a temporary directory."""
    monkeypatch.setattr(paths, "NP_ARRAY_PARENT", tmp_path)
    return tmp_path


@pytest.fixture
def completeness_config_path(tmp_path):
    """
    Write a minimal config.ini [DEFAULT] section containing only the keys CompletenessEstimator.__init__ reads
    directly, and return its path. RMS_PERCENTAGE_THRESHOLD (read by RMSDistribution) is deliberately omitted -
    RMSDistribution itself is monkeypatched out by fake_rms_distribution, so CompletenessEstimator never needs it.
    """
    config_path = tmp_path / "completeness_config.ini"
    config_path.write_text(
        "[DEFAULT]\n"
        "DETECTION_SIGMA_THRESHOLD = 5\n"
        "COMPLETENESS_FLUX_BINS = 6\n"
        "COMPLETENESS_MIN_LOG_FLUX = -1\n"
        "COMPLETENESS_MAX_LOG_FLUX = 1\n"
        "N_NOISE_PATCHES = 3\n",
        encoding="utf-8",
    )
    return config_path


class _FakeRMSDistribution:
    """
    Stand-in for diffracc.data.catalogue_distributions.RMSDistribution that skips loading the real Hardcastle
    catalogue entirely - CompletenessEstimator only ever calls .sample(), so that's all this needs to provide.
    """
    def __init__(self, rms: float = 95e-3):
        self.rms = rms

    def sample(self, size: int = 1) -> float:
        return self.rms


@pytest.fixture
def fake_rms_distribution(monkeypatch):
    """Returns a factory to monkeypatch RMSDistribution in a given module with a fixed-RMS fake."""
    def _patch(module, rms: float = 95e-3):
        monkeypatch.setattr(module, "RMSDistribution", lambda *args, **kwargs: _FakeRMSDistribution(rms))
    return _patch


@pytest.fixture
def completeness_estimator_factory(monkeypatch, completeness_config_path, fake_rms_distribution):
    """
    Returns a factory function that builds a CompletenessEstimator with override_data=True (skipping
    ImageDataArrays entirely) against the temp config and a fake RMSDistribution, so tests never touch the real
    dataset or catalogue.
    """
    from diffracc.completeness import completeness_estimator as ce_module

    monkeypatch.setattr(paths, "PROGRAM_CONFIG", completeness_config_path)
    fake_rms_distribution(ce_module)

    def _make(config_str: str = "DEFAULT", **kwargs) -> "ce_module.CompletenessEstimator":
        kwargs.setdefault("override_data", True)
        return ce_module.CompletenessEstimator(config_str, **kwargs)

    return _make


@pytest.fixture
def cutout_preprocessor_config_path(tmp_path):
    """Write a minimal config.ini [DEFAULT] section with only the cosmology keys CutoutPreprocessor.__init__ reads."""
    config_path = tmp_path / "cutout_preprocessor_config.ini"
    config_path.write_text(
        "[DEFAULT]\n"
        "h = 0.70\n"
        "Tcmb0 = 2.275\n"
        "Om0 = 0.3\n",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def cutout_preprocessor_factory(monkeypatch, cutout_preprocessor_config_path):
    """Returns a factory function that builds a CutoutPreprocessor against the temp cosmology-only config."""
    from diffracc.data.apply_preprocessing import CutoutPreprocessor

    monkeypatch.setattr(paths, "PROGRAM_CONFIG", cutout_preprocessor_config_path)

    def _make(**kwargs) -> CutoutPreprocessor:
        return CutoutPreprocessor(**kwargs)

    return _make


@pytest.fixture
def cutout_downloader_config_path(tmp_path):
    """Write a minimal config.ini [DEFAULT] section with only the FOLDER_SIZE key CutoutDownloader.__init__ reads."""
    config_path = tmp_path / "cutout_downloader_config.ini"
    config_path.write_text("[DEFAULT]\nFOLDER_SIZE = 100\n", encoding="utf-8")
    return config_path


@pytest.fixture
def cutout_downloader_factory(monkeypatch, cutout_downloader_config_path):
    """Returns a factory function that builds a CutoutDownloader against the temp FOLDER_SIZE-only config."""
    from diffracc.data.cutout_downloader import CutoutDownloader

    monkeypatch.setattr(paths, "PROGRAM_CONFIG", cutout_downloader_config_path)

    def _make() -> CutoutDownloader:
        return CutoutDownloader()

    return _make
