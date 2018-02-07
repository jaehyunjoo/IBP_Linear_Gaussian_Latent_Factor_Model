"""This module will calculate the log likelihood P(X | Z, A, sigma_a, sigma_x)."""
import numpy as np


def logPX(X, Z, A, sigma_x, N, D):
    """Compute log P(X | Z, A, sigma_a, sigma_x)."""
    """
    The D-dimensional vector of properties of an object i, x_i
    is generated from a Gaussian distribution with mean z_iA and
    covariance matrix sigma_x*I
    """
    assert(Z.shape[0] == N)
    assert(A.shape[1] == D)
    mat = X - np.dot(Z, A)
    lp = -0.5 * N * D * np.log(2*np.pi*(sigma_x**2))
    lp -= (0.5 / (sigma_x**2)) * np.trace(np.dot(mat.T, mat))
    return lp
