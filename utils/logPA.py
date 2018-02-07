"""This module will calculate the log likelihood of mixture weight."""
import numpy as np


def logPA(A, sigma_a):
    """log likelihood for P(A|sigma_a**2), N(0, sigma_a**2)."""
    (K, D) = A.shape
    prior = -0.5*K*D*np.log(2*np.pi*(sigma_a**2))
    prior -= (0.5 / (sigma_a**2)) * (np.trace(np.dot(A.T, A)))
    return prior
