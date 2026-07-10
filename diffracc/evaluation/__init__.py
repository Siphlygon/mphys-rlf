"""
Evaluation metrics for the diffusion model.

* :mod:`~diffracc.evaluation.metrics` -- embedding-agnostic statistical distances (1-D Wasserstein / KS, multivariate
  Fréchet distance, polynomial-kernel MMD / KID).
* :mod:`~diffracc.evaluation.source_properties` -- per-image physical properties (peak/total flux, rms, S/N, component
  count, size, concentration) built with a fast numpy source finder (with a hook for a PyBDSF backend).
* :mod:`~diffracc.evaluation.evaluate` -- the three headline reports: physical-distribution match, conditioning
  calibration (recovered vs prompted), and a memorisation check.

The "physical FID / KID" computed here is the standard Fréchet / MMD distance evaluated on a vector of physical summary
statistics rather than on the activations of a pretrained image network. This gives the same distribution-level
comparison as neural FID/KID while keeping every feature interpretable and defensible -- avoiding the need for (and the
need to defend) a radio-specific classifier, which is left as a future neural-embedding extension.
"""
from . import evaluate, metrics, source_properties
from .evaluate import (
    calibration_report,
    full_report,
    memorization_report,
    physical_distribution_report,
)
from .source_properties import SourceProperties, extract_batch, extract_properties

__all__ = [
    "metrics",
    "source_properties",
    "evaluate",
    "SourceProperties",
    "extract_properties",
    "extract_batch",
    "physical_distribution_report",
    "calibration_report",
    "memorization_report",
    "full_report",
]
