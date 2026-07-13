"""
Unit tests for diffracc/utils/power_transform.py's PeakFluxPowerTransformer.

paths.NP_ARRAY_PARENT is monkeypatched to a tmp_path for every test here, so nothing touches the real nparrays/
directory - only small synthetic maxvals arrays and temp files are used.
"""
import numpy as np
import pytest

from diffracc.utils import paths
from diffracc.utils.power_transform import PeakFluxPowerTransformer


class TestConstruction:
    """
    Tests for constructing PeakFluxPowerTransformer instances. Uses the np_array_parent fixture to ensure nothing
    touches the real nparrays/ directory.
    """

    def test_missing_maxvals_file_and_no_fallback_raises(self, np_array_parent):
        """
        Test that PeakFluxPowerTransformer raises FileNotFoundError if the maxvals file is missing and no maxvals
        argument is provided.
        """
        with pytest.raises(FileNotFoundError):
            PeakFluxPowerTransformer("subdir")

    def test_maxvals_argument_is_saved_to_the_expected_path(self, np_array_parent):
        """Test that providing a maxvals argument saves it to the expected path in the subdir."""
        subdir = np_array_parent / "subdir"
        subdir.mkdir()
        maxvals = np.abs(np.random.default_rng(0).normal(loc=10, scale=2, size=50)) + 1.0  # box-cox needs > 0

        PeakFluxPowerTransformer("subdir", maxvals=maxvals)

        saved_path = subdir / paths.MAXVALS
        assert saved_path.exists()
        np.testing.assert_allclose(np.load(saved_path), maxvals)

    def test_loads_existing_maxvals_file_without_needing_the_argument(self, np_array_parent):
        """
        Test that PeakFluxPowerTransformer can load an existing maxvals file without needing the maxvals argument.
        """
        subdir = np_array_parent / "subdir"
        subdir.mkdir()
        maxvals = np.abs(np.random.default_rng(1).normal(loc=10, scale=2, size=50)) + 1.0
        np.save(subdir / paths.MAXVALS, maxvals)

        # Should not raise even though maxvals=None, since the file already exists.
        transformer = PeakFluxPowerTransformer("subdir")
        assert transformer.pt is not None


class TestTransformRoundTrip:
    """Tests for the transform/inverse_transform round-trip behavior of PeakFluxPowerTransformer."""
    
    @pytest.fixture
    def transformer(self, np_array_parent):
        """A PeakFluxPowerTransformer instance with a synthetic maxvals array, using the np_array_parent fixture."""
        subdir = np_array_parent / "subdir"
        subdir.mkdir()
        maxvals = np.abs(np.random.default_rng(2).normal(loc=10, scale=2, size=200)) + 1.0
        return PeakFluxPowerTransformer("subdir", maxvals=maxvals)

    def test_inverse_transform_undoes_transform(self, transformer):
        """Test that inverse_transform(transform(x)) == x for a range of values."""
        values = np.array([5.0, 8.0, 10.0, 12.0, 15.0])
        round_tripped = transformer.inverse_transform(transformer.transform(values))
        np.testing.assert_allclose(round_tripped, values, rtol=1e-6)

    def test_transform_preserves_input_shape(self, transformer):
        """Test that transform and inverse_transform preserve the shape of the input array."""
        values = np.linspace(5.0, 15.0, 7)
        assert transformer.transform(values).shape == values.shape
        assert transformer.inverse_transform(values).shape == values.shape
