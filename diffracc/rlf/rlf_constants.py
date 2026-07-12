import numpy as np


# from Hardcastle et al. 2022, https://github.com/mhardcastle/agn-selection/blob/main/plots.py
def ccol(i):
    colours=[[0,0,0],[0,73,73],[0,146,146],[255,109,182],[255,182,219],[73,0,146],[0,109,219],[182,109,255],
             [109,182,255],[182,219,255],[146,0,0],[146,73,0],[219,209,0],[36,255,36],[255,255,109]]
    i-=1
    return [v/255.0 for v in colours[i]]

colors=[ccol(2),ccol(3),ccol(11),ccol(4),ccol(5),ccol(6),ccol(8),ccol(7),ccol(9),ccol(10)]

def z_from_v( v, a, b ):
    """
    Redshift from comoving volume, where a and b are respective points in log-log space
    """
    return np.interp( v, a, b )

shimwell_data = np.array( [
    [ 0.20,   0.00000 ],
    [ 0.22,   0.015625 ],
    [ 0.24,   0.015625 ],
    [ 0.27,   0.03125 ],
    [ 0.30,   0.06250 ],
    [ 0.34,   0.12500 ],
    [ 0.38,   0.18750 ],
    [ 0.42,   0.28125 ],
    [ 0.46,   0.34375 ],
    [ 0.52,   0.46875 ],
    [ 0.58,   0.53125 ],
    [ 0.64,   0.62500 ],
    [ 0.72,   0.71875 ],
    [ 0.80,   0.78125 ],
    [ 0.88,   0.81250 ],
    [ 0.98,   0.87500 ],
    [ 1.10,   0.90625 ],
    [ 1.19,   0.96875 ],
    [ 1.25,   1.00000 ],
] ).transpose()
