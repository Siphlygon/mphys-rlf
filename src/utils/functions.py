import numpy as np
from scipy.special import erf

def sigmoid(x, x0, k, a, b):
    """Sigmoid function: a / (1 + exp(-k*(x-x0))) + b"""
    return a / (1 + np.exp(-k * (x - x0))) + b


def sigmoid01(x, x0, k):
    """Standard logistic in (0, 1) with fixed asymptotes.

    This is equivalent to `sigmoid(x, x0, k, a=1, b=0)`.
    For k>0: approaches 0 as x -> -inf and 1 as x -> +inf.
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def richards01(x, x0, k, nu):
    """Richards / generalized logistic with fixed asymptotes 0 and 1.

    Adds a shape parameter `nu` to allow asymmetric transitions while
    preserving the long-term behaviour.

    For k>0 and nu>0: approaches 0 as x -> -inf and 1 as x -> +inf.
    Setting nu=1 reduces to the standard logistic.
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0))) ** nu


def erf01(x, x0, sigma):
    """Error-function CDF with fixed asymptotes 0 and 1.

    Often a good model for detection completeness under Gaussian noise.
    For sigma>0: approaches 0 as x -> -inf and 1 as x -> +inf.
    """
    z = (x - x0) / (np.sqrt(2.0) * sigma)
    return 0.5 * (1.0 + erf(z))

def polynomial_deg4(x, a, b, c, d, e):
    """Quadratic polynomial: ax^2 + bx + c"""
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def polynomial(degree, x, *args):
    var = 0
    for i in range( degree + 1 ):
        var += args[ i ] * x ** i
    return var

def mag_to_flux_w3( mag ):
    # https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html
    F_v0 = 31.674
    return F_v0 * 10**(-mag / 2.5)

def mag_to_flux_w2( mag ):
    # https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html
    F_v0 = 171.787
    return F_v0 * 10**(-mag / 2.5)

def k_corr_factor( redshift, mag_space: bool = False, spectral_index = -0.7 ):
    """
    Returns the k-correction factor for one or more objects at given redshifts

    mag_space: bool = False
    - whether or not to give the k correction in magnitude space (-2.5 * log10( k_corr_lum_space )), default lum space
    spectral_index = None
    - override for spectral index to use instead of self.spectral_index, broadcastable with redshift
    """
    k_corr_lum_space = ( 1 + redshift ) ** ( 1 - spectral_index )
    if not mag_space:
        return k_corr_lum_space
    else:
        return -2.5 * np.log10( k_corr_lum_space )
    
def rlf_power_law( luminosity, alpha, beta, Log10C, Log10Lstar ):
    C = 10**Log10C
    Lstar = 10**Log10Lstar
    return C / ( ( luminosity / Lstar )**alpha + ( luminosity / Lstar )**beta )

def rlf_schechter( luminosity, beta, gamma, Log10Phi, Log10Lstar ):
    phi = 10**Log10Phi
    Lstar = 10**Log10Lstar
    return phi / ( np.log( 10 ) ) * ( luminosity / Lstar )**(-beta) * np.exp( -(luminosity / Lstar)**gamma )

def rlf_power_law_evolution( x, alpha, beta, Log10C, Log10Lstar, alphaD, alphaL ):
    luminosity = x[ 0 ]
    redshift = x[ 1 ]
    return ( 1 + redshift )**( alphaD ) * rlf_power_law( luminosity / ( 1 + redshift )**( alphaL ), alpha, beta, Log10C, Log10Lstar )

def rlf_pde( x, alpha, beta, Log10C, Log10Lstar, alphaD ):
    luminosity = x[ 0 ]
    redshift = x[ 1 ]
    return ( 1 + redshift )**( alphaD ) * rlf_power_law( luminosity, alpha, beta, Log10C, Log10Lstar )

def rlf_ple( x, alpha, beta, Log10C, Log10Lstar, alphaL ):
    luminosity = x[ 0 ]
    redshift = x[ 1 ]
    return rlf_power_law( luminosity / ( 1 + redshift )**( alphaL ), alpha, beta, Log10C, Log10Lstar )

def yuan_evolution_a( x, alpha, beta, Log10C, Log10Lstar, m, z0, zsigma, k1 ):
    l = x[ 0 ]
    z = x[ 1 ]

    e1 = np.where( z > z0, z**m, z**m * np.exp( -0.5 * ( ( z - z0 ) / zsigma )**2 ) )
    e2 = (1 + z)**k1

    return e1 * rlf_power_law( l / e2, alpha, beta, Log10C, Log10Lstar )

def yuan2018_evolution_a( x, p1, p2, zc, Log10Phi, Log10Lstar, beta, gamma, k1 ):
    l = x[ 0 ]
    z = x[ 1 ]

    p0 = ( (1+zc)**p1 + (z+zc)**p2 )
    e1 = p0 * ( ( (1+zc) / (1+z) )**p1 + ( (1+zc) / (1+z) )**p2 )**(-1)
    e2 = (1 + z)**k1
    return e1 * rlf_schechter( l / e2, beta, gamma, Log10Phi, Log10Lstar )
