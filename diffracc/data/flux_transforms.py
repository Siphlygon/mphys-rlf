"""
Global, invertible flux-space transforms for radio images.

Standard per-image min-max normalisation applies a different affine map to every image and therefore destroys absolute
flux calibration and cross-image relative-noise information. The transforms in this module instead apply a single,
fixed, monotonic bijection to every image in the dataset. Their sole purpose is to bring the data's standard deviation
to O(1), which is what the EDM preconditioning and mixed-precision (fp16) training require for numerical stability --
see the fp16 underflow of ``sigma_data**2`` when ``sigma_data`` is left at the raw ~1e-2 Jy scale.

Because each transform is a fixed bijection with an exact analytic inverse:

* absolute flux calibration is preserved -- a physical peak-flux prompt maps to a well-defined value and is recovered
exactly at sampling time via :meth:`inverse`;
* relative-noise information between images is preserved -- the same map is applied to all;
* the transform is non-destructive -- ``inverse(forward(x)) == x`` up to float precision.

Two transforms are provided:

``GlobalLinearScale``
    ``y = k * x``. The purely non-destructive change of units. Preserves everything but does not compress dynamic range,
    so it is best paired with a bright-source flux cut.

``GlobalAsinhScale``
    ``y = k * asinh(x / beta)``. Approximately linear for ``|x| << beta`` (faithful at the fainter end), approximately 
    logarithmic for ``|x| >> beta`` (compresses the Jy-level bright tail so a single ``sigma_data`` value is 
    appropriate for faint and bright images at once). This is the recommended transform for a low-flux focused study.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)

Arrayish = Union[torch.Tensor, np.ndarray]

# Name of the JSON file the fitted transform is persisted to, alongside a model or dataset.
FLUX_TRANSFORM_FILE = "flux_transform.json"


def _to_tensor(x: Arrayish) -> tuple[torch.Tensor, bool]:
    """
    Return ``x`` as a floating-point tensor, plus a flag recording whether the input was a numpy array (so the result
    can be handed back in the same type).
    
    Parameters
    ----------
    x : Arrayish
        Input array, either a numpy array or a torch tensor.
    
    Returns
    -------
    tuple[torch.Tensor, bool]
        A tuple containing the input converted to a floating-point torch tensor and a boolean indicating whether the
        original input was a numpy array.
    """
    was_numpy = isinstance(x, np.ndarray)
    t = torch.from_numpy(x) if was_numpy else x
    if not t.is_floating_point():
        t = t.float()
    return t, was_numpy


def _flat_sample(images: Arrayish, sample_size: int, seed: int) -> np.ndarray:
    """
    Flatten ``images`` to a 1-D float32 numpy array of finite pixels, subsampled to at most ``sample_size`` pixels for a
    cheap-but-representative fit.
    
    Parameters
    ----------
    images : Arrayish
        Input images, either a numpy array or a torch tensor.
    sample_size : int
        Maximum number of pixels to sample. If the total number of finite pixels in ``images`` exceeds this value, a
        random subset of ``sample_size`` pixels is returned.
    seed : int
        Random seed for reproducibility when subsampling.
    
    Returns
    -------
    np.ndarray
        A 1-D numpy array of finite pixel values, subsampled to at most ``sample_size`` pixels.
    """
    arr = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    arr = arr.astype(np.float32).ravel()
    arr = arr[np.isfinite(arr)]
    if sample_size and arr.size > sample_size:
        rng = np.random.default_rng(seed)
        arr = rng.choice(arr, size=sample_size, replace=False)
    return arr


def robust_noise(pixels: np.ndarray) -> float:
    """
    Robust (MAD-based) estimate of the background noise sigma of a pixel sample.

    Uses ``1.4826 * median(|x - median(x)|)``, which for radio cut-outs is dominated by the empty-sky background rather
    than the compact source, giving a stable noise scale. Useful for setting the ``beta`` parameter of
    :class:`GlobalAsinhScale` to a few times the background noise.
    
    Parameters
    ----------
    pixels : np.ndarray
        Input pixel values, typically a 1-D array of finite pixels from an image or a collection of images.
    
    Returns
    -------
    float
        Robust estimate of the background noise sigma.
    """
    med = np.median(pixels)
    return float(1.4826 * np.median(np.abs(pixels - med)))


class _GlobalFluxTransform:
    """
    Shared interface for the global, invertible flux transforms.
    
    For a transform ``T`` with forward and inverse methods, the following should hold for all finite inputs ``x``:
    .. code-block:: python
        T.inverse(T.forward(x)) == x
    
    This is to allow the transform to be applied to the training data and then exactly inverted at sampling time,
    preserving absolute flux calibration and relative noise information between images.
    """

    name: str = "base"

    def __call__(self, x: Arrayish) -> Arrayish:
        return self.forward(x)


    def forward(self, x: Arrayish) -> Arrayish:  # pragma: no cover - overridden
        raise NotImplementedError


    def inverse(self, y: Arrayish) -> Arrayish:  # pragma: no cover - overridden
        raise NotImplementedError


    # ---- persistence ----
    def to_dict(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError


    def save(self, path: Path | str) -> Path:
        """
        Save the transform parameters to ``path`` (a file, or a directory in which the standard
        :data:`FLUX_TRANSFORM_FILE` is written). Returns the file path written.
        
        Parameters
        ----------
        path : Path | str
            The file path or directory where the transform parameters should be saved. If a directory is provided, the
            parameters will be saved to a file named ``flux_transform.json`` within that directory.
        
        Returns
        -------
        Path
            The file path where the transform parameters were saved.
        """
        path = Path(path)
        if path.is_dir():
            path = path / FLUX_TRANSFORM_FILE
        path.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")
        logger.info(f"Saved flux transform to {path}: {self.to_dict()}")
        return path


    def max_abs_roundtrip_error(self, images: Arrayish, sample_size: int = 20000, seed: int = 0) -> float:
        """
        Calculates the largest absolute error of ``inverse(forward(x))`` over a pixel sample -- a direct check that the
        transform is non-destructive on real data.
        
        Parameters
        ----------
        images : Arrayish
            Input images, either a numpy array or a torch tensor.
        sample_size : int, optional
            Maximum number of pixels to sample for the error calculation. Default is 20,000.
        seed : int, optional
            Random seed for reproducibility when subsampling. Default is 0.
        
        Returns
        -------
        float
            The maximum absolute error of the round-trip transformation over the sampled pixels.
        """
        arr = _flat_sample(images, sample_size, seed)
        recovered = self.inverse(self.forward(arr))
        return float(np.max(np.abs(recovered - arr)))



class GlobalLinearScale(_GlobalFluxTransform):
    """
    Global linear rescale``y = k * x`` (exact inverse ``x = y / k``).
    """

    name = "linear"

    def __init__(self, k: float):
        """
        Initialise the linear scale transform with a scaling factor ``k``.

        Parameters
        ----------
        k : float
            The scaling factor for the linear transformation.
        """
        assert k > 0, f"k must be positive, got {k}."
        self.k = float(k)


    def forward(self, x: Arrayish) -> Arrayish:
        """
        Apply the linear transformation to the input array ``x``.
        
        Uses the formula ``y = k * x`` to transform the input array.

        Parameters
        ----------
        x : Arrayish
            The input array to be transformed.

        Returns
        -------
        Arrayish
            The transformed array.
        """
        t, was_numpy = _to_tensor(x)
        y = self.k * t
        return y.numpy() if was_numpy else y


    def inverse(self, y: Arrayish) -> Arrayish:
        """
        Apply the inverse linear transformation to the input array ``y``.
        
        Uses the formula ``x = y / k`` to recover the original values from the transformed array.

        Parameters
        ----------
        y : Arrayish
            The input array to be inverse-transformed.

        Returns
        -------
        Arrayish
            The inverse-transformed array.
        """
        t, was_numpy = _to_tensor(y)
        x = t / self.k
        return x.numpy() if was_numpy else x


    @classmethod
    def fit(
        cls,
        images: Arrayish,
        sigma_data: float = 0.5,
        sample_size: int = 100_000,
        seed: int = 0,
    ) -> "GlobalLinearScale":
        """
        Fit ``k`` so that the transformed data has pooled std ``sigma_data``, using a random sample of pixels from
        ``images``.
        
        This is a simple linear rescaling that preserves the relative flux calibration and noise characteristics of the
        images while adjusting the overall scale to achieve the desired standard deviation to match the diffusion
        model's expected noise level for the scheduler.
        
        Parameters
        ----------
        images : Arrayish
            Input images, either a numpy array or a torch tensor.
        sigma_data : float, optional
            The target standard deviation for the transformed data. Default is 0.5.
        sample_size : int, optional
            Maximum number of pixels to sample for fitting. Default is 100,000.
        seed : int, optional
            Random seed for reproducibility when subsampling. Default is 0.
        
        Returns
        -------
        GlobalLinearScale
            An instance of the GlobalLinearScale class with the fitted scaling factor ``k``.
        """
        arr = _flat_sample(images, sample_size, seed)
        std = float(arr.std())
        obj = cls(k=sigma_data / std)
        logger.info(f"Fitted GlobalLinearScale: k={obj.k:.6g} (raw std={std:.4e}, target sigma_data={sigma_data}).")
        return obj


    def to_dict(self) -> dict:
        """
        Convert the GlobalLinearScale instance to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing the name of the transform and the scaling factor ``k``.
        """
        return {"name": self.name, "k": self.k}



class GlobalAsinhScale(_GlobalFluxTransform):
    r"""Global asinh flux compression ``y = k * asinh(x / beta)``.

    Exact inverse: ``x = beta * sinh(y / k)``.
    """

    name = "asinh"

    def __init__(self, beta: float, k: float):
        """
        Initialise the GlobalAsinhScale transform with parameters ``beta`` and ``k``.

        Parameters
        ----------
        beta : float
            Linear-to-logarithmic transition scale, in the same physical units as the images (Jy/beam). For
            ``|x| << beta`` the map is ~linear (faithful, noise stays ~Gaussian); for ``|x| >> beta`` it is ~logarithmic
            (compresses the bright tail). Choose ``beta`` a few times the background noise so the faint end you care
            about stays quasi-linear.
        k : float
            Output gain, chosen so the transformed data has std ~ ``sigma_data``.
        """
        assert beta > 0, f"beta must be positive, got {beta}."
        assert k > 0, f"k must be positive, got {k}."
        self.beta = float(beta)
        self.k = float(k)


    def forward(self, x: Arrayish) -> Arrayish:
        """
        Apply the asinh transformation to the input array ``x``.
        
        Uses the formula ``y = k * asinh(x / beta)`` to transform the input array.

        Parameters
        ----------
        x : Arrayish
            The input array to be transformed.

        Returns
        -------
        Arrayish
            The transformed array.
        """
        t, was_numpy = _to_tensor(x)
        y = self.k * torch.asinh(t / self.beta)
        return y.numpy() if was_numpy else y


    def inverse(self, y: Arrayish) -> Arrayish:
        """
        Apply the inverse asinh transformation to the input array ``y``.
        
        Uses the formula ``x = beta * sinh(y / k)`` to recover the original values from the transformed array.

        Parameters
        ----------
        y : Arrayish
            The input array to be inverse-transformed.

        Returns
        -------
        Arrayish
            The inverse-transformed array.
        """
        t, was_numpy = _to_tensor(y)
        x = self.beta * torch.sinh(t / self.k)
        return x.numpy() if was_numpy else x


    @classmethod
    def fit(
        cls,
        images: Arrayish,
        sigma_data: float = 0.5,
        beta: float | None = None,
        noise: float | None = None,
        beta_scale: float = 3.0,
        sample_size: int = 100_000,
        seed: int = 0,
    ) -> "GlobalAsinhScale":
        """Fit ``beta`` and ``k`` from the data.

        ``beta`` defaults to ``beta_scale * noise``, where ``noise`` is the robust (MAD) background sigma if not
        supplied. ``k`` is then set so the transformed pooled std is ``sigma_data``.
        
        Parameters
        ----------
        images : Arrayish
            Input images, either a numpy array or a torch tensor.
        sigma_data : float, optional
            The target standard deviation for the transformed data. Default is 0.5.
        beta : float | None, optional
            Linear-to-logarithmic transition scale. If ``None``, it is set to ``beta_scale * noise``. Default is
            ``None``.
        noise : float | None, optional
            Robust (MAD) estimate of the background noise sigma. If ``None``, it is computed from the data. Default is
            ``None``.
        beta_scale : float, optional
            Scale factor for ``beta`` when it is not provided. Default is 3.0.
        sample_size : int, optional
            Maximum number of pixels to sample for fitting. Default is 100,000.
        seed : int, optional
            Random seed for reproducibility when subsampling. Default is 0.
        
        Returns
        -------
        GlobalAsinhScale
            An instance of the GlobalAsinhScale class with the fitted parameters ``beta`` and ``k``.
        """
        arr = _flat_sample(images, sample_size, seed)

        if beta is None:
            if noise is None:
                noise = robust_noise(arr)
            beta = beta_scale * noise

        # Fit k so that the transformed data has pooled std sigma_data
        y = np.arcsinh(arr / beta)
        k = float(sigma_data / y.std())
        obj = cls(beta=beta, k=k)

        logger.info(
            f"Fitted GlobalAsinhScale: beta={obj.beta:.4e}, k={obj.k:.6g} "
            f"(target sigma_data={sigma_data}, beta_scale={beta_scale})."
        )
        return obj


    def to_dict(self) -> dict:
        return {"name": self.name, "beta": self.beta, "k": self.k}



_REGISTRY = {cls.name: cls for cls in (GlobalLinearScale, GlobalAsinhScale)}


def from_dict(params: dict) -> _GlobalFluxTransform:
    """
    Reconstruct a flux transform from its :meth:`to_dict` representation.
    
    Parameters
    ----------
    params : dict
        A dictionary containing the parameters of a flux transform, including the key ``"name"`` that specifies the type
        of transform.
    
    Returns
    -------
    _GlobalFluxTransform
        An instance of the corresponding flux transform class.

    Raises
    ------
    ValueError
        If the ``"name"`` key in ``params`` does not correspond to a known flux transform.
    """
    params = dict(params)
    name = params.pop("name")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown flux transform '{name}'. Known: {list(_REGISTRY)}.")
    return _REGISTRY[name](**params)


def load(source: Path | str | dict | _GlobalFluxTransform | None) -> _GlobalFluxTransform | None:
    """
    Flexibly load a flux transform from an instance, a parameter dict, a JSON file, or a directory containing
    :data:`FLUX_TRANSFORM_FILE`. Returns ``None`` for ``None`` input.
    
    Parameters
    ----------
    source : Path | str | dict | _GlobalFluxTransform | None
        The source from which to load the flux transform. It can be:
        - An instance of a flux transform (returned as-is).
        - A dictionary containing the parameters of a flux transform (reconstructed using :func:`from_dict`).
        - A path to a JSON file containing the parameters of a flux transform.
        - A path to a directory containing a file named ``flux_transform.json`` with the parameters of a flux transform.
        - ``None``, in which case ``None`` is returned.
    
    Returns
    -------
    _GlobalFluxTransform | None
        The loaded flux transform instance, or ``None`` if the input was ``None``.
    
    Raises
    ------
    FileNotFoundError
        If the specified file or directory does not exist or does not contain a valid flux transform file.
    """
    if source is None or isinstance(source, _GlobalFluxTransform):
        return source
    if isinstance(source, dict):
        return from_dict(source)

    path = Path(source)
    if path.is_dir():
        path = path / FLUX_TRANSFORM_FILE
    if not path.exists():
        raise FileNotFoundError(f"No flux transform file at {path}.")
    return from_dict(json.loads(path.read_text(encoding="utf-8")))
