"""This module will update sigma_a and sigma_x using Gibbs sampler."""
import numpy as np
import numpy.random as nr


def sampleSigma_x(X, Z, A, sigma_xa, sigma_xb, N, D):
    """update sigma_x using Gibbs sampler."""
    # Sample noise variance
    # Let E = X - (Z o V)A, then
    # P(sigma_x | E, sigma_xa, sigmx_xb) is proportional to
    # P(E | sigma_x)P(sigma_x | sigma_xa, sigma_xb)
    # This distribution has a canonical form of inverse gamma ditribution
    # To make it simple we define
    # P(1/sigma_x ** 2) ~ Ga(sigma_xa, sigma_xb)
    # Then,
    # posterior_shape = sigma_xa + ND/2
    # posterior_scale = 1/(sigma_xb + 0.5 * tr(E'E))
    E = X - np.dot(Z, A)
    var_x = np.trace(np.dot(E.T, E))
    postshape_x = sigma_xa + N * D * 0.5
    postscale_x = 1.0 / (sigma_xb*D + var_x * 0.5)
    tau_x = nr.gamma(postshape_x, scale=postscale_x)
    sigma_x = np.sqrt(1.0 / tau_x)
    return sigma_x


def sampleSigma_a(X, Z, A, sigma_aa, sigma_ab, K, D):
    """update sigma_a using Gibbs sampler."""
    # Sample feature variance A
    # P(sigma_a | A, sigma_aa, sigma_ab) is proportional to
    # P(A | sigma_a)P(sigma_a | sigma_aa, sigma_ab)
    # This distribution has a canonical form of inverse gamma distribution
    # P(1/sigma_a ** 2) ~ Ga(sigma_aa, sigma_ab)
    # Then,
    # postrior_shape = sigma_xa + KD/2
    # posterior_scale = 1/(sigma_ab + 0.5 * tr(A'A))
    Acomp_square = np.trace(np.dot(A.T, A))
    postshape_a = sigma_aa + K * D * 0.5
    postscale_a = 1.0 / (sigma_ab + Acomp_square * 0.5)
    tau_a = nr.gamma(postshape_a, scale=postscale_a)
    sigma_a = np.sqrt(1.0 / tau_a)
    return sigma_a
