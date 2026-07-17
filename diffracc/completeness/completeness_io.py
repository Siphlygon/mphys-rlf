"""
The on-file connection between CompletenessEstimator, which fits a completeness curve, and RLF, which applies it.

For a completeness curve to be evaluated, the caller must know which function it is and which x-axis it was fitted
against. A CompletenessFit is therefore a self-describing record of a fit, and it is written to and read from a JSON
file.

This replaces the legacy system of a bare text file of fitted floats, which was uninformative and led to a bug that
silently inflated every phi estimate by an order of magnitude.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils import functions as func

# The x-axis a completeness curve was fitted against. `mJy` means the fit saw linear flux density in mJy; `log10_mJy`
# means it saw log10(flux / mJy). These are not interchangeable: a logistic in linear flux is a different functional
# form from a logistic in log flux, so a fit in one space cannot be re-expressed in the other by transforming its
# parameters. It has to be refitted, or evaluated in the space it was made for.
X_SPACE_MJY = "mJy"
X_SPACE_LOG10_MJY = "log10_mJy"
VALID_X_SPACES = (X_SPACE_MJY, X_SPACE_LOG10_MJY)

# Only functions that are meaningful as a completeness curve. Every one of these takes x as its first argument and the
# midpoint x0 as its second, which `CompletenessFit.evaluate` relies on when it applies a flux-space shift.
FUNCTION_REGISTRY = {
    "sigmoid": func.sigmoid,
    "sigmoid01": func.sigmoid01,
    "richards01": func.richards01,
    "erf01": func.erf01,
}


@dataclass(frozen=True)
class CompletenessFit:
    """
    A fitted completeness curve, together with the two facts needed to evaluate it correctly: which function it is, and
    which x-axis it was fitted against.

    Attributes
    ----------
    function_name : str
        The name of the fitted function, which must be a key of FUNCTION_REGISTRY.
    x_space : str
        The x-axis the fit was performed against, one of VALID_X_SPACES.
    popt : np.ndarray
        The fitted parameters, in the order the function takes them (x0 first).
    pcov : np.ndarray | None
        The covariance matrix of the fit, or None if it was not recorded.
    param_names : list[str]
        The names of the parameters, for logging and for reading the file by eye.
    provenance : str
        Free text recording where this fit came from. Worth filling in: the bug this module exists to prevent was only
        diagnosable because the numbers could be traced back to the code that produced them.
    """
    function_name: str
    x_space: str
    popt: np.ndarray
    pcov: np.ndarray | None = None
    param_names: list[str] = field(default_factory=list)
    provenance: str = ""

    def __post_init__(self):
        # Validate the function name and x_space, and convert popt and pcov to float arrays. This is the only place
        # where the function name and x_space are checked, so it is the only place a caller can be sure they are valid.
        if self.function_name not in FUNCTION_REGISTRY:
            raise ValueError(f"Unknown completeness function {self.function_name!r}. "
                             f"Known functions: {sorted(FUNCTION_REGISTRY)}")
        if self.x_space not in VALID_X_SPACES:
            raise ValueError(f"Unknown completeness x_space {self.x_space!r}. Valid spaces: {list(VALID_X_SPACES)}")

        popt = np.asarray(self.popt, dtype=float)
        object.__setattr__(self, "popt", popt)
        if self.pcov is not None:
            object.__setattr__(self, "pcov", np.asarray(self.pcov, dtype=float))

        # Catch a parameter-count mismatch at load time rather than as a TypeError from deep inside a Monte Carlo
        # integral.
        n_expected = self.function.__code__.co_argcount - 1  # drop x
        if popt.ndim != 1 or popt.shape[0] != n_expected:
            raise ValueError(f"{self.function_name} takes {n_expected} parameters, but popt has shape {popt.shape}. "
                             f"The file and the function disagree about which curve this is.")

    @property
    def function(self):
        """The callable this fit's parameters belong to."""
        return FUNCTION_REGISTRY[self.function_name]

    def evaluate(self, fluxes_mjy: np.ndarray | float, s0_shift_mjy: float = 0.0) -> np.ndarray:
        """
        Evaluate the completeness curve at the given fluxes.

        Callers pass a flux density in mJy and nothing else. Applying (or not applying) the log10 is deliberately this
        method's responsibility rather than the caller's, because getting that wrong is the entire bug this module
        exists to prevent.

        Parameters
        ----------
        fluxes_mjy : np.ndarray | float
            The flux density/densities to evaluate the completeness at, in mJy.
        s0_shift_mjy : float, optional
            A shift applied to the curve's midpoint in linear flux space, in mJy, by default 0. Used by RLF to probe the
            effect of a systematic flux-scale bias on the resulting luminosity function.

        Returns
        -------
        np.ndarray
            The completeness at each flux, in [0, 1] for the bounded functions.
        """
        popt = self.popt.copy()

        if s0_shift_mjy:
            # x0 is the first parameter of every registered function, but it lives in x_space -- so the shift has to be
            # applied in linear flux and then converted back, not added to x0 directly.
            s0_mjy = popt[0] if self.x_space == X_SPACE_MJY else 10.0 ** popt[0]
            s0_mjy = s0_mjy + s0_shift_mjy
            if s0_mjy <= 0:
                raise ValueError(f"s0_shift_mjy={s0_shift_mjy} moves the completeness midpoint to {s0_mjy} mJy, "
                                 f"which has no log and no physical meaning.")
            popt[0] = s0_mjy if self.x_space == X_SPACE_MJY else np.log10(s0_mjy)

        fluxes_mjy = np.asarray(fluxes_mjy, dtype=float)
        if self.x_space == X_SPACE_MJY:
            x = fluxes_mjy
        else:
            # A non-positive flux has no log. -inf is the honest answer and drives every registered curve to 0
            # completeness, which is the physically right result, so let it through quietly rather than warn.
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.log10(fluxes_mjy)

        return self.function(x, *popt)


def write_completeness_fit(path: str | Path, fit: CompletenessFit) -> Path:
    """
    Write a fitted completeness curve to a self-describing JSON file.

    Parameters
    ----------
    path : str | Path
        The path to write to. A `.json` suffix is expected.
    fit : CompletenessFit
        The fit to write.

    Returns
    -------
    Path
        The path written to.
    """
    path = Path(path)

    # Enforce the .json suffix on write, symmetrically with read_completeness_fit's refusal to read anything else.
    if path.suffix.lower() != ".json":
        raise ValueError(f"Completeness fits are written as self-describing JSON, but {path} has suffix "
                         f"{path.suffix!r}. Use a .json path so it can be read back by read_completeness_fit.")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "function": fit.function_name,
        "x_space": fit.x_space,
        "param_names": list(fit.param_names),
        "popt": np.asarray(fit.popt, dtype=float).tolist(),
        "pcov": None if fit.pcov is None else np.asarray(fit.pcov, dtype=float).tolist(),
        "provenance": fit.provenance,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def read_completeness_fit(path: str | Path) -> CompletenessFit:
    """
    Read a completeness fit written by `write_completeness_fit`.

    Parameters
    ----------
    path : str | Path
        The path to read. Must be a JSON file written by `write_completeness_fit`.

    Returns
    -------
    CompletenessFit
        The fit, validated for function/parameter-count consistency.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file is a legacy bare-parameter text file, or is missing the metadata needed to evaluate it. Guessing
        the missing units is precisely the failure this module exists to prevent, so it is refused instead.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find completeness fit at {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"{path} is not a self-describing completeness fit. Legacy files hold bare fitted floats with no record "
            f"of which function they belong to or whether they were fitted against linear mJy or log10(mJy), and "
            f"guessing wrong silently corrupts every phi estimate. Convert it with `write_completeness_fit` once you "
            f"have established its provenance, or regenerate it with CompletenessEstimator.")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    missing = {"function", "x_space", "popt"} - set(payload)
    if missing:
        raise ValueError(f"Completeness fit at {path} is missing required field(s) {sorted(missing)}, so it cannot be "
                         f"evaluated without guessing. Regenerate it with CompletenessEstimator.")

    return CompletenessFit(
        function_name=payload["function"],
        x_space=payload["x_space"],
        popt=np.asarray(payload["popt"], dtype=float),
        pcov=None if payload.get("pcov") is None else np.asarray(payload["pcov"], dtype=float),
        param_names=payload.get("param_names", []),
        provenance=payload.get("provenance", ""),
    )
