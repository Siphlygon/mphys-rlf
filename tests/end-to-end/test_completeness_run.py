"""
End-to-end tests for the completeness estimation pipeline, including _detect_mock_sources,
_compute_completeness_per_bin, and _fit_function.
"""
import numpy as np


class TestEstimateCompleteness:
    """End-to-end test tying _detect_mock_sources, _compute_completeness_per_bin, and _fit_function together."""

    def test_runs_end_to_end_and_completeness_rises_with_flux(self,
                                                              completeness_estimator_factory,
                                                              monkeypatch,
                                                              tmp_path):
        """Test that estimate_completeness runs end-to-end and produces a completeness curve that rises with flux."""
        monkeypatch.chdir(tmp_path)
        np.random.seed(0)
        ce = completeness_estimator_factory()
        n_sources = 30
        model_fluxes = np.logspace(-1, 1, n_sources)  # spans the configured 0.1-10 mJy flux-bin range
        # A background level equal to each source's own flux, so brighter sources are more reliably detected -
        # mirrors how a real source's peak pixel value relates to its integrated/model flux.
        ce.data.model_images = np.stack([np.full((80, 80), flux) for flux in model_fluxes])
        ce.data.model_fluxes = model_fluxes

        log_bin_centers, completeness, yerr, fitted_params, pcov = ce.estimate_completeness(
            plot_completeness=False, show_progress=False)

        assert log_bin_centers.shape[0] == ce.num_flux_bins - 1
        assert completeness.shape == log_bin_centers.shape
        assert np.all((completeness >= 0) & (completeness <= 1))
        # the faintest bin (well below the ~0.475 mJy detection threshold) should be no more complete than the
        # brightest bin (well above it).
        assert completeness[0] <= completeness[-1]
        assert fitted_params.shape == (4,)

    def test_saves_completeness_and_fit_params_to_output_files(self,
                                                               completeness_estimator_factory,
                                                               monkeypatch,
                                                               tmp_path):
        """Test that estimate_completeness saves the completeness and fit parameters to output files if specified."""
        monkeypatch.chdir(tmp_path)
        np.random.seed(0)
        ce = completeness_estimator_factory()
        model_fluxes = np.logspace(-1, 1, 10)
        ce.data.model_images = np.stack([np.full((80, 80), flux) for flux in model_fluxes])
        ce.data.model_fluxes = model_fluxes

        ce.estimate_completeness(
            comp_output_file=tmp_path / "completeness.txt",
            func_output_file=tmp_path / "fit_params.json",
            plot_completeness=False, show_progress=False,
        )

        assert (tmp_path / "completeness.txt").exists()
        assert (tmp_path / "fit_params.json").exists()
        assert (tmp_path / "mock_fluxes_detectability.txt").exists()

        # The fit is written as a self-describing record that reads back and reports the x-axis it was fitted against.
        from diffracc.completeness.completeness_io import X_SPACE_LOG10_MJY, read_completeness_fit
        fit = read_completeness_fit(tmp_path / "fit_params.json")
        assert fit.x_space == X_SPACE_LOG10_MJY
