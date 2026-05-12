import numpy as np

def sigmoid(x, x0, k, a, b):
    """Sigmoid function: a / (1 + exp(-k*(x-x0))) + b"""
    return a / (1 + np.exp(-k * (np.log10( x ) - x0))) + b

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
