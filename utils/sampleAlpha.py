"""Update alpha using Gibbs sampler."""
import numpy as np
import numpy.random as nr


def sampleAlpha(alpha_a, alpha_b, K, N):
    """Sample alpha from conjugate posterior."""
    postshape = alpha_a + K
    H_N = np.array([range(N)]) + 1.0
    H_N = np.sum(1.0 / H_N)
    postscale = 1.0 / (alpha_b + H_N)
    alpha = nr.gamma(postshape, scale=postscale)
    return alpha
