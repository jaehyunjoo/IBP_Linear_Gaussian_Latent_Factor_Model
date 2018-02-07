"""This module will update mixing matrix A."""
import numpy as np
import numpy.random as nr


def sampleA(X, Z, A, sigma_x, sigma_a, K, D):
    """Sample mixture weight A."""
    assert(Z.shape[1] == K)

    M = np.linalg.inv(np.dot(Z.T, Z) + ((sigma_x/sigma_a)**2) * np.eye(K))
    meanA = np.dot(M, np.dot(Z.T, X))
    covA = (sigma_x**2) * M
    L = np.linalg.cholesky(covA)

    for d in range(D):
        meanA_d = meanA[:, d:(d+1)]
        normal_sample = nr.normal(0, 1, size=(K, 1))
        A_sample = meanA_d + np.dot(L, normal_sample)
        A[:, d:(d+1)] = A_sample

    return A
