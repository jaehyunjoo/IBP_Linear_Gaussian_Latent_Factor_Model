"""This module will update a binary feature matrix Z."""
import numpy as np
import numpy.random as nr
from . import logPX
from . import logPV
from . import sampleV  # slice sampler


def sampleZ(X, Z, A, sigma_x, sigma_a, alpha, K, N, D, realvaluedZ, proposeNewfeature):
    """Drawing Z using an uncollapsed Gibbs sampler."""

    for i in range(N):
        # Calculate m_-i,k
        m = (Z != 0).astype(np.int).sum(axis=0)
        assert(Z.shape[1] == len(m))
        m_i = ((Z[i, :] != 0).astype(np.int))
        m_neg_i = m - m_i

        # emulate IBP-FM using a finite model
        # Compute prior p(z_ik = 1 or = 0 | z_-i,k) = (m_-i,k+alpha/K)/ (N+alpha/K)
        # need only condition on z_-i,k rather than Z_-(ik) because the columns
        # of the matrix are generated independently under this prior
        prior_z1 = (m_neg_i+alpha/K)/(float(N)+alpha/K)
        prior_z0 = (N-m_neg_i)/(float(N)+alpha/K)  # 1 - Pz1
        assert(np.isfinite(prior_z0).all())
        assert(np.isfinite(prior_z1).all())

        # Iterate through the columns of the matrix
        for k in range(K):
            if (realvaluedZ):
                old_zik = Z[i, k]

            # Compute a log likelihood p(z_ik = 0 | Z_-(ik), X)
            Z[i, k] = 0
            logp0 = logPX.logPX(X, Z, A, sigma_x, N, D)
            assert(np.isfinite(logp0))
            logp0 += np.log(prior_z0[k])
            assert(np.isfinite(logp0))

            # Compute a log likelihood p(z_ik = 1 | Z_-(ik), X)
            Z[i, k] = 1
            if (realvaluedZ):
                if (old_zik == 0):
                    Z[i, k] = nr.normal(0, 1)  # propose v from prior N(0, 1)
                else:
                    Z[i, k] = old_zik  # recycle the current value

            logp1 = logPX.logPX(X, Z, A, sigma_x, N, D)
            assert(np.isfinite(logp1))
            logp1 += np.log(prior_z1[k])
            assert(np.isfinite(logp1))

            # Add log prior for feature weight v
            # need only calculate single v because the rest weights are same
            # between z_ik = 0 and z_ik = 1
            if (realvaluedZ):
                logp1 += logPV.logPvi(Z[i, k])
                assert(np.isfinite(logp1))

            log_diff = logp1 - logp0

            # If np.exp(log_diff) overflows, then numpy will handle overflows gracefully
            # np.exp(1000) result in inf and 1/inf therefor simply results in 0
            # if set np.seterr('raise') RuntimeWarning -> FloatingPointError
            try:
                exp_log_diff = np.exp(log_diff)
            except FloatingPointError as e:
                print (e)
                print ("Cannot exponentiate ", log_diff)

            p0 = 1.0 / (1 + exp_log_diff)

            if (nr.uniform(0, 1) < p0):
                Z[i, k] = 0
            else:
                if (realvaluedZ):
                    # sample new v through a slice sampler
                    Z[i, k] = sampleV.sampleV(i, k, X, Z, A, sigma_x, N, D)
                else:
                    Z[i, k] = 1

    return (Z, K, A)
