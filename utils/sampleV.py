"""This module will update feature weight V using slice sampler."""
import numpy as np
import numpy.random as nr

from . import logPX
from . import logPV


def sampleV(i, k, X, Z, A, sigma_x, N, D):
    """Sample feature weight using a slice sampler."""

    curv = Z[i, k]  # check the current value of v

    # Log posterior of current value
    # do not consider (z_ik = 1 | Z_-ik) because this prior value is same.
    # for the same reason, we don't have to consider P(A | sigma_a)
    # for the prior of feature weight, only consider value Z[i, k]
    curlp = logPX.logPX(X, Z, A, sigma_x, N, D) + logPV.logPvi(curv)

    # Vertically sample beneath this value
    # sample from uniform(0, exp(curlp)) = uniform(0, 1)*exp(curlp)
    # take log, so that curlp + log(uniform(0, 1))
    # we prefer log domain to avoid a possible floating-point underflow
    logy = curlp + np.log(nr.uniform(0, 1))

    Z_new = np.copy(Z)  # leave Z untouched and work with Z_new
    assert(not np.may_share_memory(Z, Z_new))
    (L, R) = Interval(i, k, curv, logy, X, Z, A, sigma_x, N, D)
    assert(L < curv < R)

    newv = L + nr.uniform(0, 1)*(R-L)
    Z_new[i, k] = newv
    newlp = logPX.logPX(X, Z_new, A, sigma_x, N, D) + logPV.logPvi(newv)

    # Repeat until valid sample obtained
    # sample should be drawn from the slice
    while(newlp < logy):
        # Shrink interval
        if (newv < curv):
            L = newv
        else:
            R = newv
        newv = L + nr.uniform(0, 1)*(R-L)
        Z_new[i, k] = newv
        newlp = logPX.logPX(X, Z_new, A, sigma_x, N, D) + logPV.logPvi(newv)
    assert(L < newv < R)
    return newv


def Interval(i, k, curv, logy, X, Z, A, sigma_x, N, D):
    """Stepping-out procedure to find an interval around the current point."""

    w = 0.3  # estimate of the typical size of a slice
    m = 10  # integer limiting the size of a slice to mw

    U = nr.uniform(0, 1)
    L = curv-w*U
    R = L + w
    V = nr.uniform(0, 1)
    J = np.floor(m*V)
    T = (m-1)-J

    Z_tmp = np.copy(Z)  # leave Z untouched and work with Z_tmp
    assert(not np.may_share_memory(Z, Z_tmp))

    Z_tmp[i, k] = L
    Llp = logPX.logPX(X, Z_tmp, A, sigma_x, N, D) + logPV.logPvi(L)

    Z_tmp[i, k] = R
    Rlp = logPX.logPX(X, Z_tmp, A, sigma_x, N, D) + logPV.logPvi(R)

    # Stepping out
    while(J > 0 and logy < Llp):
        L -= w
        J -= 1
        Z_tmp[i, k] = L
        Llp = logPX.logPX(X, Z_tmp, A, sigma_x, N, D) + logPV.logPvi(L)

    while(T > 0 and logy < Rlp):
        R += w
        T -= 1
        Z_tmp[i, k] = R
        Rlp = logPX.logPX(X, Z_tmp, A, sigma_x, N, D) + logPV.logPvi(R)
    return (L, R)
