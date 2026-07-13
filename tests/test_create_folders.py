"""
Unit tests for diffracc/program_prep/create_folders.py's make_folders().

All of paths.* that make_folders() reads are monkeypatched to point into tmp_path, so nothing here touches the
real project directories.
"""
import tempfile
from pathlib import Path

import pytest

from diffracc.program_prep import create_folders as cf


def _symlinks_supported() -> bool:
    """Windows requires elevated privileges or Developer Mode to create symlinks; probe once at collection time."""
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "target"
        target.mkdir()
        link = Path(d) / "link"
        try:
            link.symlink_to(target)
            return True
        except OSError:
            return False


SYMLINKS_SUPPORTED = _symlinks_supported()


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """
    Point every paths.* constant make_folders() reads at a tmp_path-scoped equivalent, mirroring the real
    derivation (e.g. PYBDSF_CATALOG_PARENT = PYBDSF_PARENT / 'catalogs') but rooted under tmp_path.
    """
    model_parent = tmp_path / "model_results"
    analysis_parent = tmp_path / "analysis_results"
    img_data_parent = tmp_path / "image_data"
    fits_parent = tmp_path / "fits_images"
    pybdsf_parent = tmp_path / "pybdsf"
    np_array_parent = tmp_path / "nparrays"
    dataset_parent = tmp_path / "datasets"

    monkeypatch.setattr(cf.paths, "BASE_PARENT", tmp_path)
    monkeypatch.setattr(cf.paths, "STORAGE_PARENT", tmp_path)
    monkeypatch.setattr(cf.paths, "MODEL_PARENT", model_parent)
    monkeypatch.setattr(cf.paths, "ANALYSIS_PARENT", analysis_parent)
    monkeypatch.setattr(cf.paths, "IMG_DATA_PARENT", img_data_parent)
    monkeypatch.setattr(cf.paths, "FITS_PARENT", fits_parent)
    monkeypatch.setattr(cf.paths, "PYBDSF_PARENT", pybdsf_parent)
    monkeypatch.setattr(cf.paths, "NP_ARRAY_PARENT", np_array_parent)
    monkeypatch.setattr(cf.paths, "DATASET_PARENT", dataset_parent)
    monkeypatch.setattr(cf.paths, "MODEL_NAMES", ["ModelA", "ModelB"])
    monkeypatch.setattr(cf.paths, "PRETRAINED_PARENT", model_parent / "pretrained")
    monkeypatch.setattr(cf.paths, "PYBDSF_CATALOG_PARENT", pybdsf_parent / "catalogs")
    monkeypatch.setattr(cf.paths, "PYBDSF_EXPORT_IMAGE_PARENT", pybdsf_parent / "images")
    monkeypatch.setattr(cf.paths, "PYBDSF_LOG_PARENT", pybdsf_parent / "logs")
    monkeypatch.setattr(cf.paths, "SUBDIRS", ["subA", "subB"])
    monkeypatch.setattr(cf.paths, "CUTOUTS_PATH", fits_parent / "cutouts")
    monkeypatch.setattr(cf.paths, "PREPROCESSING_PARENT", dataset_parent / "preprocessing")

    return {
        "model_parent": model_parent, "analysis_parent": analysis_parent, "img_data_parent": img_data_parent,
        "fits_parent": fits_parent, "pybdsf_parent": pybdsf_parent, "np_array_parent": np_array_parent,
        "dataset_parent": dataset_parent,
    }


class TestMakeFoldersMainStructure:
    """Tests for the main structure of folders created by make_folders()."""

    def test_creates_main_storage_folders(self, patched_paths):
        """Test that make_folders() creates the main storage folders."""
        cf.make_folders()
        for path in patched_paths.values():
            assert path.exists()

    def test_creates_per_model_folders(self, patched_paths):  # NOTE: deprecated
        """Test that make_folders() creates folders for each model."""
        cf.make_folders()
        assert (patched_paths["img_data_parent"] / "ModelA").exists()
        assert (patched_paths["img_data_parent"] / "ModelB").exists()
        assert (patched_paths["model_parent"] / "ModelA_model").exists()
        assert (patched_paths["model_parent"] / "ModelB_model").exists()

    def test_creates_pretrained_folder(self, patched_paths):  # NOTE: deprecated
        """Test that make_folders() creates the pretrained folder."""
        cf.make_folders()
        assert (patched_paths["model_parent"] / "pretrained").exists()

    def test_creates_pybdsf_and_nparray_subdirs_per_subdir(self, patched_paths):
        """Test that make_folders() creates the pybdsf and nparray subdirectories for each subdir."""
        cf.make_folders()
        for parent_key in ("pybdsf_parent",):
            for sub in ("catalogs", "images", "logs"):
                for subdir in ("subA", "subB"):
                    assert (patched_paths[parent_key] / sub / subdir).exists()
        for subdir in ("subA", "subB"):
            assert (patched_paths["np_array_parent"] / subdir).exists()

    def test_creates_cutouts_and_preprocessing_folders(self, patched_paths):
        """Test that make_folders() creates the cutouts and preprocessing folders."""
        cf.make_folders()
        assert (patched_paths["fits_parent"] / "cutouts").exists()
        assert (patched_paths["dataset_parent"] / "preprocessing").exists()

    def test_idempotent_when_run_twice(self, patched_paths):
        """Test that make_folders() can be run multiple times without error."""
        cf.make_folders()
        cf.make_folders()  # must not raise on already-existing folders


@pytest.mark.skipif(not SYMLINKS_SUPPORTED, reason="symlink creation needs elevated privileges/Developer Mode here")
class TestMakeFoldersSymlinks:
    """Tests for make_folders() when the storage parent is different from the base parent, requiring symlinks."""

    @pytest.fixture
    def patched_paths_with_separate_storage(self, tmp_path, monkeypatch):
        """
        This fixture sets up a scenario where the storage parent is different from the base parent, requiring symlinks.
        """
        base = tmp_path / "base"
        base.mkdir()
        storage = tmp_path / "storage"
        storage.mkdir()
        model_parent = storage / "model_results"

        monkeypatch.setattr(cf.paths, "BASE_PARENT", base)
        monkeypatch.setattr(cf.paths, "STORAGE_PARENT", storage)
        monkeypatch.setattr(cf.paths, "MODEL_PARENT", model_parent)
        monkeypatch.setattr(cf.paths, "ANALYSIS_PARENT", storage / "analysis_results")
        monkeypatch.setattr(cf.paths, "IMG_DATA_PARENT", storage / "image_data")
        monkeypatch.setattr(cf.paths, "FITS_PARENT", storage / "fits_images")
        monkeypatch.setattr(cf.paths, "PYBDSF_PARENT", storage / "pybdsf")
        monkeypatch.setattr(cf.paths, "NP_ARRAY_PARENT", storage / "nparrays")
        monkeypatch.setattr(cf.paths, "DATASET_PARENT", storage / "datasets")
        monkeypatch.setattr(cf.paths, "MODEL_NAMES", [])
        monkeypatch.setattr(cf.paths, "PRETRAINED_PARENT", model_parent / "pretrained")
        monkeypatch.setattr(cf.paths, "PYBDSF_CATALOG_PARENT", storage / "pybdsf" / "catalogs")
        monkeypatch.setattr(cf.paths, "PYBDSF_EXPORT_IMAGE_PARENT", storage / "pybdsf" / "images")
        monkeypatch.setattr(cf.paths, "PYBDSF_LOG_PARENT", storage / "pybdsf" / "logs")
        monkeypatch.setattr(cf.paths, "SUBDIRS", [])
        monkeypatch.setattr(cf.paths, "CUTOUTS_PATH", storage / "fits_images" / "cutouts")
        monkeypatch.setattr(cf.paths, "PREPROCESSING_PARENT", storage / "datasets" / "preprocessing")

        return base, model_parent

    def test_creates_symlinks_in_base_when_storage_differs(self, patched_paths_with_separate_storage):
        """Test that make_folders() creates symlinks in the base directory when the storage parent is different."""
        base, model_parent = patched_paths_with_separate_storage
        cf.make_folders()

        symlink = base / "model_results"
        assert symlink.is_symlink()
        assert symlink.resolve() == model_parent.resolve()

    def test_raises_on_broken_pre_existing_symlink(self, tmp_path, patched_paths_with_separate_storage):
        """Test that make_folders() raises an AssertionError if a pre-existing symlink points to the wrong target."""
        base, model_parent = patched_paths_with_separate_storage
        wrong_target = tmp_path / "somewhere_else"
        wrong_target.mkdir()
        (base / "model_results").symlink_to(wrong_target)

        with pytest.raises(AssertionError):
            cf.make_folders()
