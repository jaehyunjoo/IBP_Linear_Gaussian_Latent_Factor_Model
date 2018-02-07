"""This module will calculate the log likelihood of IBP prior."""
import numpy as np
from scipy.special import gammaln, betaln
# gammaln(n+1) = log(n!)
# betaln: natural logarithm of absolute value of beta function


def logIBP(Z, alpha, K, N):
    """Calculate the marginal probability of a binary matrix Z"""
    lp = 0.
    for k in range(K):
        m_k = np.sum(Z[:, k])
        lp += betaln(m_k+alpha/K, N-m_k+1.) - betaln(alpha/N, 1.)
    return lp

