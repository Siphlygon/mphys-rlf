"""
Unit tests for diffracc/scripts/generate_rlf_plots.py.

main()'s heavy dependencies (get_catalogue_info, RLF) are monkeypatched out with fakes below - this tests main()'s
own orchestration logic (config reading, argument wiring, RLF construction, calculate_rlf/plot_rlf calls, figure
saving), not RLF's actual Monte Carlo numerics, which are already covered by tests/test_rlf.py.
"""
import numpy as np
import pytest

from diffracc.scripts import generate_rlf_plots as gp
from diffracc.utils import paths


class TestBuildArgParser:
    """Unit tests for the _build_arg_parser function."""

    def test_defaults(self):
        """Test that the default values for flux_cut_jy and plot_rlagn_selection_contour are set correctly."""
        parser = gp._build_arg_parser()
        args = parser.parse_args([])
        assert args.flux_cut_jy == pytest.approx(1.1e-3)
        assert args.plot_rlagn_selection_contour is False

    def test_custom_flux_cut(self):
        """Test that a custom flux_cut_jy value is correctly parsed."""
        parser = gp._build_arg_parser()
        args = parser.parse_args(["--flux_cut_jy", "2.5e-3"])
        assert args.flux_cut_jy == pytest.approx(2.5e-3)

    def test_plot_rlagn_selection_contour_flag(self):
        """Test that the --plot_rlagn_selection_contour flag is correctly parsed."""
        parser = gp._build_arg_parser()
        args = parser.parse_args(["--plot_rlagn_selection_contour"])
        assert args.plot_rlagn_selection_contour is True


class _FakeRLF:
    """A fake RLF class to simulate the behavior of the real RLF class for testing purposes."""
    instances = []

    def __init__(self, fluxes, redshifts, luminosities, resolved, cosmo,
                bias=0, flux_cut_jy=1.1e-3, vmax_method=False):
        self.fluxes = fluxes
        self.redshifts = redshifts
        self.luminosities = luminosities
        self.resolved = resolved
        self.cosmo = cosmo
        self.bias = bias
        self.flux_cut_jy = flux_cut_jy
        self.vmax_method = vmax_method
        self.calculate_rlf_calls = []
        self.plot_rlf_calls = []
        _FakeRLF.instances.append(self)

    def calculate_rlf(self, plot_rlf=False):
        self.calculate_rlf_calls.append(plot_rlf)

    def plot_rlf(self, title, colors, ax, draw_ylabel=True):
        self.plot_rlf_calls.append((title, ax, draw_ylabel))


@pytest.fixture(autouse=True)
def _clear_fake_rlf_instances():
    """
    Ensure that the _FakeRLF.instances list is cleared before and after each test to avoid cross-test contamination.
    """
    _FakeRLF.instances = []
    yield
    _FakeRLF.instances = []


@pytest.fixture
def cosmology_config_path(tmp_path):
    """Fixture to create a temporary config.ini file with cosmological parameters for testing."""
    config_path = tmp_path / "config.ini"
    config_path.write_text("[DEFAULT]\nh = 0.70\nTcmb0 = 2.725\nOm0 = 0.3\n", encoding="utf-8")
    return config_path


@pytest.fixture
def patched_main_deps(monkeypatch, tmp_path, cosmology_config_path):
    """Fixture to patch dependencies of main() for testing."""
    n = 5
    rng = np.random.default_rng(0)
    catalogue_info = (
        rng.uniform(0.01, 1.0, n),   # redshifts
        rng.uniform(1.1e-3, 1.0, n),  # fluxes
        rng.uniform(21, 29, n),      # luminosities
        rng.integers(0, 2, n).astype(bool),  # resolved
    )
    monkeypatch.setattr(gp, "get_catalogue_info",
                        lambda cosmo, flux_cut_jy, plot_rlagn_selection_contour=False: catalogue_info)
    monkeypatch.setattr(gp, "RLF", _FakeRLF)
    monkeypatch.setattr(paths, "PROGRAM_CONFIG", cosmology_config_path)
    monkeypatch.chdir(tmp_path)
    return catalogue_info


class TestMain:
    """Unit tests for the main function of generate_rlf_plots.py."""

    def _args(self, **overrides):
        """Helper function to create an argparse.Namespace with default values, overridden by any provided arguments."""
        defaults = dict(flux_cut_jy=1.1e-3, plot_rlagn_selection_contour=False)
        defaults.update(overrides)
        return __import__("argparse").Namespace(**defaults)

    def test_calls_get_catalogue_info_with_flux_cut_and_contour_flag(self, patched_main_deps, monkeypatch):
        """
        Test that main() calls get_catalogue_info with the correct flux_cut_jy and plot_rlagn_selection_contour
        arguments.
        """
        calls = []
        monkeypatch.setattr(gp, "get_catalogue_info",
                            lambda cosmo, flux_cut_jy, plot_rlagn_selection_contour=False:
                            (calls.append((flux_cut_jy, plot_rlagn_selection_contour)) or patched_main_deps))
        gp.main(self._args(flux_cut_jy=2.0e-3, plot_rlagn_selection_contour=True))
        assert calls == [(2.0e-3, True)]

    def test_constructs_two_rlf_instances_with_zero_bias_and_matching_flux_cut(self, patched_main_deps):
        """Test that main() constructs two RLF instances with bias=0 and the correct flux_cut_jy."""
        gp.main(self._args(flux_cut_jy=1.1e-3))
        assert len(_FakeRLF.instances) == 2
        for rlf in _FakeRLF.instances:
            assert rlf.bias == 0
            assert rlf.flux_cut_jy == pytest.approx(1.1e-3)

    def test_only_the_second_rlf_instance_uses_vmax_method(self, patched_main_deps):
        """Test that only the second RLF instance is constructed with vmax_method=True."""
        gp.main(self._args())
        assert _FakeRLF.instances[0].vmax_method is False
        assert _FakeRLF.instances[1].vmax_method is True

    def test_calls_calculate_rlf_and_plot_rlf_on_both_instances(self, patched_main_deps):
        """Test that main() calls calculate_rlf and plot_rlf on both RLF instances."""
        gp.main(self._args())
        for rlf in _FakeRLF.instances:
            assert rlf.calculate_rlf_calls == [False]  # plot_rlf=False passed to calculate_rlf
            assert len(rlf.plot_rlf_calls) == 1

    def test_draw_ylabel_true_only_for_first_axis(self, patched_main_deps):
        """Test that draw_ylabel is True for the first RLF instance and False for the second when plotting."""
        gp.main(self._args())
        draw_ylabels = [rlf.plot_rlf_calls[0][2] for rlf in _FakeRLF.instances]
        assert draw_ylabels == [True, False]

    def test_saves_figure_to_expected_filename(self, patched_main_deps, tmp_path):
        """Test that main() saves the figure to 'rlfs_vmax.png' in the current working directory."""
        gp.main(self._args())
        assert (tmp_path / "rlfs_vmax.png").exists()

    def test_resolved_passed_to_rlf_matches_get_catalogue_info(self, patched_main_deps):
        """Test that the resolved array passed to RLF instances matches the one returned by get_catalogue_info."""
        gp.main(self._args())
        expected_resolved = patched_main_deps[3]
        for rlf in _FakeRLF.instances:
            np.testing.assert_array_equal(rlf.resolved, expected_resolved)
