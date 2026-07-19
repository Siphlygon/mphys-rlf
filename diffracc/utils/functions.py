import numpy as np
import scipy
from scipy.special import erf


def sigmoid(x: np.ndarray,
            x0: float,
            k: float,
            a: float,
            b: float) -> np.ndarray:
    """
    A generalized logistic function (also known as a sigmoid function) that can be used to model growth processes.
    Sigmoid function: a / (1 + exp(-k*(x-x0))) + b

    Parameters
    ----------
    x : np.ndarray
        The input values for which the sigmoid function is to be calculated.
    x0 : float
        The x-value of the sigmoid's midpoint.
    k : float
        The steepness of the sigmoid.
    a : float
        The curve's maximum value.
    b : float
        The curve's minimum value.

    Returns
    -------
    np.ndarray
        The output values of the sigmoid function.
    """
    return scipy.special.expit(k * (x - x0)) * a + b


def sigmoid01(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    """
    Standard logistic in (0, 1) with fixed asymptotes.

    This is equivalent to `sigmoid(x, x0, k, a=1, b=0)`. For k>0: approaches 0 as x -> -inf and 1 as x -> +inf.
    
    Sigmoid01 function: 1 / (1 + exp(-k*(x-x0)))

    Parameters
    ----------
    x : np.ndarray
        The input values for which the sigmoid function is to be calculated.
    x0 : float
        The x-value of the sigmoid's midpoint.
    k : float
        The steepness of the sigmoid.

    Returns
    -------
    np.ndarray
        The output values of the sigmoid function.
    """
    return scipy.special.expit(k * (x - x0))


def richards01(x: np.ndarray, x0: float, k: float, nu: float) -> np.ndarray:
    """
    Richards / generalized logistic with fixed asymptotes 0 and 1.

    Adds a shape parameter `nu` to allow asymmetric transitions while preserving the long-term behaviour.
    For k>0 and nu>0: approaches 0 as x -> -inf and 1 as x -> +inf. Setting nu=1 reduces to the standard logistic.
   
    Richards function: (1 / (1 + exp(-k*(x-x0))))^ (1/nu)
    
    Parameters
    ----------
    x : np.ndarray
        The input values for which the Richards function is to be calculated.
    x0 : float
        The x-value of the Richards function's midpoint.
    k : float
        The steepness of the Richards function.
    nu : float
        The shape parameter that allows asymmetric transitions.

    Returns
    -------
    np.ndarray
        The output values of the Richards function.
    """
    return scipy.special.expit(k * (x - x0)) ** (1/nu)


def erf01(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """
    Error-function CDF with fixed asymptotes 0 and 1.

    Often a good model for detection completeness under Gaussian noise.
    For sigma>0: approaches 0 as x -> -inf and 1 as x -> +inf.
    
    Parameters
    ----------
    x : np.ndarray
        The input values for which the error function is to be calculated.
    x0 : float
        The x-value of the error function's midpoint.
    sigma : float
        The standard deviation of the Gaussian distribution.
        
    Returns
    -------
    np.ndarray
        The output values of the error function.
    """
    z = (x - x0) / (np.sqrt(2.0) * sigma)
    return 0.5 * (1.0 + erf(z))


def mag_to_flux_w3(mag: np.ndarray) -> np.ndarray:
    """
    Converts magnitude to flux for W3 band.

    Parameters
    ----------
    mag : np.ndarray
        The magnitudes to convert to flux.

    Returns
    -------
    np.ndarray
        The corresponding fluxes in Jy for the W3 band.
    """
    # https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html
    F_v0 = 31.674
    return F_v0 * 10**(-mag / 2.5)


def mag_to_flux_w2(mag: np.ndarray) -> np.ndarray:
    """
    Converts magnitude to flux for W2 band.

    Parameters
    ----------
    mag : np.ndarray
        The magnitudes to convert to flux.

    Returns
    -------
    np.ndarray
        The corresponding fluxes in Jy for the W2 band.
    """
    # https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html
    F_v0 = 171.787
    return F_v0 * 10**(-mag / 2.5)


def k_corr_factor(redshift: np.ndarray,
                  mag_space: bool = False,
                  spectral_index: float = -0.7) -> np.ndarray:
    """
    Returns the k-correction factor for one or more objects at given redshifts
    
    K-correction factor: (1 + z)^(1 - alpha) for luminosity space,
    or -2.5 * log10((1 + z)^(1 - alpha)) for magnitude space.

    Parameters
    ----------
    redshift : np.ndarray
        The redshifts of the object(s) for which to calculate the k-correction factor.
    mag_space : bool, optional
        If True, returns the k-correction factor in magnitude space. If False, returns the k-correction factor in
        luminosity space. Default is False.
    spectral_index : float, optional
        The spectral index of the object(s). Default is -0.7.
    
    Returns
    -------
    np.ndarray
        The k-correction factors for the given redshift(s) in either magnitude or luminosity space, depending on the
        value of `mag_space`.
    """
    k_corr_lum_space = (1 + redshift) ** (1 - spectral_index)
    if not mag_space:
        return k_corr_lum_space
    return -2.5 * np.log10(k_corr_lum_space)


def rlf_power_law(luminosity: np.ndarray,
                  alpha: float,
                  beta: float,
                  Log10C: float,
                  Log10Lstar: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity using a double power-law model.
    
    Double power-law: C / ((L/Lstar)^alpha + (L/Lstar)^beta)

    Parameters
    ----------
    luminosity : np.ndarray
        The luminosities at which to calculate the RLF.
    alpha : float
        The first power-law index.
    beta : float
        The second power-law index.
    Log10C : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosity.
    """
    C = 10**Log10C
    Lstar = 10**Log10Lstar
    return C / ((luminosity / Lstar)**alpha + (luminosity / Lstar)**beta)


def rlf_schechter(luminosity: np.ndarray,
                  beta: float,
                  gamma: float,
                  Log10Phi: float,
                  Log10Lstar: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity using a Schechter function model.

    Schechter function: phi / (ln(10)) * (L/Lstar)^(-beta) * exp(-(L/Lstar)^gamma)

    Parameters
    ----------
    luminosity : np.ndarray
        The luminosities at which to calculate the RLF.
    beta : float
        The beta parameter of the Schechter function.
    gamma : float
        The gamma parameter of the Schechter function.
    Log10Phi : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosity.
    """
    phi = 10**Log10Phi
    Lstar = 10**Log10Lstar
    return phi / (np.log(10)) * (luminosity / Lstar)**(-beta) * np.exp(-(luminosity / Lstar)**gamma)


def rlf_power_law_evolution(x: np.ndarray,
                            alpha: float,
                            beta: float,
                            Log10C: float,
                            Log10Lstar: float,
                            alphaD: float,
                            alphaL: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity and redshift using a double power-law
    model with evolution.
    
    Power-law evolution: (1 + z)^alphaD * C / ((L/(1+z)^alphaL/Lstar)^alpha + (L/(1+z)^alphaL/Lstar)^beta)
    or equivalently: (1 + z)^alphaD * rlf_power_law(L/(1+z)^alphaL, alpha, beta, Log10C, Log10Lstar)

    Parameters
    ----------
    x : np.ndarray
        A 2D array where x[0] is the luminosity and x[1] is the redshift.
    alpha : float
        The alpha parameter of the power-law function.
    beta : float
        The beta parameter of the power-law function.
    Log10C : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.
    alphaD : float
        The evolution parameter for the power-law function.
    alphaL : float
        The luminosity evolution parameter.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosities and redshifts.
    """
    luminosity = x[ 0 ]
    redshift = x[ 1 ]
    power_law = rlf_power_law(luminosity / (1 + redshift)**(alphaL), alpha, beta, Log10C, Log10Lstar)
    return (1 + redshift)**(alphaD) * power_law


def rlf_pde(x: np.ndarray,
            alpha: float,
            beta: float,
            Log10C: float,
            Log10Lstar: float,
            alphaD: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity and redshift using a double power-law
    model with pure density evolution (PDE).
    
    Pure density evolution: (1 + z)^alphaD * C / ((L/Lstar)^alpha + (L/Lstar)^beta)
    or equivalently: (1 + z)^alphaD * rlf_power_law(L, alpha, beta, Log10C, Log10Lstar)

    Parameters
    ----------
    x : np.ndarray
        A 2D array where x[0] is the luminosity and x[1] is the redshift.
    alpha : float
        The alpha parameter of the power-law function.
    beta : float
        The beta parameter of the power-law function.
    Log10C : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.
    alphaD : float
        The evolution parameter for the power-law function.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosities and redshifts.
    """
    luminosity = x[ 0 ]
    redshift = x[ 1 ]
    return (1 + redshift)**(alphaD) * rlf_power_law(luminosity, alpha, beta, Log10C, Log10Lstar)


def rlf_ple(x: np.ndarray,
            alpha: float,
            beta: float,
            Log10C: float,
            Log10Lstar: float,
            alphaL: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity and redshift using a double power-law
    model with pure luminosity evolution (PLE).

    Parameters
    ----------
    x : np.ndarray
        A 2D array where x[0] is the luminosity and x[1] is the redshift.
    alpha : float
        The alpha parameter of the power-law function.
    beta : float
        The beta parameter of the power-law function.
    Log10C : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.
    alphaL : float
        The evolution parameter for the power-law function.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosities and redshifts.
    """
    luminosity = x[ 0 ]
    redshift = x[ 1 ]
    return rlf_power_law(luminosity / (1 + redshift)**(alphaL), alpha, beta, Log10C, Log10Lstar)


def yuan_evolution_a(x: np.ndarray,
                     alpha: float,
                     beta: float,
                     Log10C: float,
                     Log10Lstar: float,
                     m: float,
                     z0: float,
                     zsigma: float,
                     k1: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity and redshift using a double power-law
    model with evolution as described in Yuan et al. (2018).
    
    Yuan et al. (2018) evolution: e1 * rlf_power_law(L/e2, alpha, beta, Log10C, Log10Lstar)

    Parameters
    ----------
    x : np.ndarray
        A 2D array where x[0] is the luminosity and x[1] is the redshift.
    alpha : float
        The alpha parameter of the power-law function.
    beta : float
        The beta parameter of the power-law function.
    Log10C : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.
    m : float
        The evolution parameter for the power-law function.
    z0 : float
        The reference redshift for the evolution.
    zsigma : float
        The width of the redshift distribution for the evolution.
    k1 : float
        The evolution parameter for the luminosity evolution.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosities and redshifts.
    """
    l = x[ 0 ]
    z = x[ 1 ]

    e1 = np.where(z > z0, z**m, z**m * np.exp(-0.5 * ((z - z0) / zsigma)**2))
    e2 = (1 + z)**k1

    return e1 * rlf_power_law(l / e2, alpha, beta, Log10C, Log10Lstar)


def yuan2018_evolution_a(x: np.ndarray,
                         p1: float,
                         p2: float,
                         zc: float,
                         Log10Phi: float,
                         Log10Lstar: float,
                         beta: float,
                         gamma: float,
                         k1: float) -> np.ndarray:
    """
    Returns the value of the radio luminosity function (RLF) at a given luminosity and redshift using a double power-law
    model with evolution as described in Yuan et al. (2018a).

    Yuan et al. (2018a) evolution: e1 * rlf_schechter(L/e2, beta, gamma, Log10Phi, Log10Lstar)
    but e1 = p0 * (((1+zc) / (1+z))**p1 + ((1+zc) / (1+z))**p2)**(-1)

    Parameters
    ----------
    x : np.ndarray
        A 2D array where x[0] is the luminosity and x[1] is the redshift.
    p1 : float
        The first evolution parameter for the power-law function.
    p2 : float
        The second evolution parameter for the power-law function.
    zc : float
        The characteristic redshift for the evolution.
    Log10Phi : float
        The base-10 logarithm of the normalization constant.
    Log10Lstar : float
        The base-10 logarithm of the characteristic luminosity.
    beta : float
        The beta parameter of the Schechter function.
    gamma : float
        The gamma parameter of the Schechter function.
    k1 : float
        The evolution parameter for the luminosity evolution.

    Returns
    -------
    np.ndarray
        The values of the RLF at the given luminosities and redshifts.
    """
    l = x[ 0 ]
    z = x[ 1 ]

    p0 = (1+zc)**p1 + (z+zc)**p2
    e1 = p0 * (((1+zc) / (1+z))**p1 + ((1+zc) / (1+z))**p2)**(-1)
    e2 = (1 + z)**k1
    return e1 * rlf_schechter(l / e2, beta, gamma, Log10Phi, Log10Lstar)
