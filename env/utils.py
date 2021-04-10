import numpy as np


def interpolate(vb, weights, N):
    """
    weights: np.ndarray with shape (N+3, 3), whose each row is the weights of three polynomial interpolation
    """
    vm = np.zeros(shape=N + 3)
    vp = np.zeros(shape=N + 3)

    for i in range(N + 3):
        vm0 = (1 / 3) * vb[i + 2] + (5 / 6) * vb[i + 3] + (-1 / 6) * vb[i + 4]
        vm1 = (-1 / 6) * vb[i + 1] + (5 / 6) * vb[i + 2] + (1 / 3) * vb[i + 3]
        vm2 = (1 / 3) * vb[i] + (-7 / 6) * vb[i + 1] + (11 / 6) * vb[i + 2]

        vp0 = (11 / 6) * vb[i + 2] + (-7 / 6) * vb[i + 3] + (1 / 3) * vb[i + 4]
        vp1 = (1 / 3) * vb[i + 1] + (5 / 6) * vb[i + 2] + (-1 / 6) * vb[i + 3]
        vp2 = (-1 / 6) * vb[i] + (5 / 6) * vb[i + 1] + (1 / 3) * vb[i + 2]

        vm[i] = weights[i, 0] * vm0 + weights[i, 1] * vm1 + weights[i, 2] * vm2
        vp[i] = weights[i, 2] * vp0 + weights[i, 1] * vp1 + weights[i, 0] * vp2

    return vm, vp


def total_variation_norm(x):
    """
    The total variation norm of an one dimensional signal is defined by \sum_{i=0}^{N-1}{|f(x_{i+1})-f(x_{i})|}.
    
    Parameters
    ----------
    x: np.ndarray whose "ndim" attribute is equal to 1.
    
    References
    ----------
    https://en.wikipedia.org/wiki/Total_variation_denoising#1D_signal_series
    """
    return np.sum(np.abs(x[1:] - x[:-1]))


def maximum_temporal_difference(x, x_prime):
    """
    The maximum of temporal differences

    Parameters
    ----------
    x, x_prime: np.ndarray whose "ndim" attribute is equal to 1.
    """
    return np.max(np.abs(x_prime - x))
