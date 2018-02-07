"""This module implement Gaussian Factor Model using a IBP prior."""
from __future__ import division
import numpy as np
import numpy.random as nr
import os
import datetime
import scipy.io
from numbers import Number

from utils.logIBPprior import logIBP
from utils.logPX import logPX
from utils.logPA import logPA
from utils.logPV import logPV
from utils.sampleSigma import sampleSigma_x
from utils.sampleSigma import sampleSigma_a
from utils.sampleZ import sampleZ
from utils.sampleA import sampleA
from utils.sampleAlpha import sampleAlpha


def IBPFM(iteration, upperbound_K, data, alpha, sigma_x, sigma_a, stdData=True,
          initZ=None, initA=None, realvaluedZ=False,
          proposeNewfeature=True, updateZ=True, updateA=True,
          updateAlpha=True, updateSigma_x=True, updateSigma_a=True,
          save_interim=False, interim_interval=100, resume=False):
    """uncollapsed Gibbs sampler for sparse FA"""
    # X = (Z o V)A + noise

    # @iteration: # of simulation.
    # @upperbound_K: K that restricts max # of features
    # @param data: N X D data matrix
    # @param alpha: Fixed IBP parameter or with Gamma hyperprior.
    # @param sigma_x: Fixed noise std or with Gamma hyperprior.
    # @param sigma_a: Fixed weight std or with Gamma hyperprior.

    # OPTIONAL ARGS
    # stdData: Standardize data X
    # initZ: Optional initial state for Z
    # initA: Optional initial state for A
    # realvaluedZ: Use real-valued feature assignment matrix Z
    # proposeNewfeature: propose new feature based on the Poisson(alpha/N)
    # updateZ: update Z
    # updateA: update A
    # updateAlpha: update alpha
    # updateSigma_x: update sigma_x
    # updateSigma_a: update sigma_a
    # interim_interval: Write all the temporary outputs by predfined interval
    # resume: resume the saved simulation previously

    # Data matrix
    X = data
    (N, D) = X.shape

    K = upperbound_K

    if (stdData):
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    if (not resume):
        assert(isinstance(alpha, Number) or type(alpha) == tuple)
        if(type(alpha) == tuple):
            (alpha, alpha_a, alpha_b) = alpha
        else:
            (alpha, alpha_a, alpha_b) = (alpha, None, None)

        assert(isinstance(sigma_x, Number) or type(sigma_x) == tuple)
        if(type(sigma_x) == tuple):
            (sigma_x, sigma_xa, sigma_xb) = sigma_x
        else:
            (sigma_x, sigma_xa, sigma_xb) = (sigma_x, None, None)

        assert(isinstance(sigma_a, Number) or type(sigma_a) == tuple)
        if(type(sigma_a) == tuple):
            (sigma_a, sigma_aa, sigma_ab) = sigma_a
        else:
            (sigma_a, sigma_aa, sigma_ab) = (sigma_a, None, None)

        # Initialize Z
        if (initZ is None):
            Z = nr.binomial(1, 0.5, size=(N, K))
            Z = Z.astype(np.float)
        else:
            Z = initZ
            # Initial Z should have a dimension (N, upperbound_K)
            assert(Z.shape[0] == N and Z.shape[1] <= K)
            if (Z.shape[1] < K):
                K_diff = K - Z.shape[1]
                Z = np.hstack((Z, np.zeros((N, K_diff))))

            assert(Z.shape == (N, K))
            if not realvaluedZ:
                Z = (Z != 0).astype(np.int)
                Z = Z.astype(np.float)

        # Generate real value V when Z was provided as binary matrix
        # even if realvaluedZ V model was enabled
        if (realvaluedZ):
            test_binary_mat = np.array([0, 1])
            if np.in1d(Z, test_binary_mat).all():
                for (i, k) in zip(*Z.nonzero()):
                    Z[i, k] = nr.normal(0, 1)

        # Initialize A
        if (initA is None):
            A = nr.normal(0, sigma_a, size=(K, D))
            assert(A.shape[0] == Z.shape[1])
        else:
            A = initA
            # Initial Z should have a dimension (upperbound K, D)
            assert (A.shape[0] <= K and A.shape[1] == D)
            if (A.shape[0] < K):
                K_gab = K - A.shape[0]
                A = np.vstack((A, np.zeros((K_gab, D))))

            assert(A.shape == (K, D))

        assert(all((N, K, D, alpha, alpha_a, alpha_b, sigma_x, sigma_xa,
                   sigma_xb, sigma_a, sigma_aa, sigma_ab)) > 0)

        # output containers for params
        K_save = np.zeros(iteration + 1, dtype=np.int)
        alpha_save = np.zeros(iteration + 1)
        sigma_x_save = np.zeros(iteration + 1)
        sigma_a_save = np.zeros(iteration + 1)
        loglikelihood_save = np.zeros(iteration + 1)

        Z_save = np.zeros((N, K, iteration + 1))
        A_save = np.zeros((K, D, iteration + 1))

    else:
        assert(isinstance(alpha, Number) or type(alpha) == tuple)
        if(type(alpha) == tuple):
            (alpha, alpha_a, alpha_b) = alpha
        else:
            (alpha, alpha_a, alpha_b) = (alpha, None, None)

        assert(isinstance(sigma_x, Number) or type(sigma_x) == tuple)
        if(type(sigma_x) == tuple):
            (sigma_x, sigma_xa, sigma_xb) = sigma_x
        else:
            (sigma_x, sigma_xa, sigma_xb) = (sigma_x, None, None)

        assert(isinstance(sigma_a, Number) or type(sigma_a) == tuple)
        if(type(sigma_a) == tuple):
            (sigma_a, sigma_aa, sigma_ab) = sigma_a
        else:
            (sigma_a, sigma_aa, sigma_ab) = (sigma_a, None, None)

        tmp_file = open("simulation.tmp", "rb")
        s_stopped = np.load(tmp_file)
        K_save = np.load(tmp_file)
        alpha_save = np.load(tmp_file)
        sigma_x_save = np.load(tmp_file)
        sigma_a_save = np.load(tmp_file)
        loglikelihood_save = np.load(tmp_file)
        Z_save = np.load(tmp_file)
        A_save = np.load(tmp_file)

        K_real = int(K_save[s_stopped+1])
        alpha = alpha_save[s_stopped+1]
        sigma_x = sigma_x_save[s_stopped+1]
        sigma_a = sigma_a_save[s_stopped+1]
        logModel = loglikelihood_save[s_stopped+1]
        Z = Z_save[:, 0:K, s_stopped+1]
        A = A_save[0:K, :, s_stopped+1]

        print ("=========================================================")
        print ("Resuming simulation: start from %d iteration......"
               % (s_stopped+1))
        print ("Data shape: Obs = %d\t Dim = %d" % (N, D))
        print ("K = %d\tLikelihood = %f" % (K_real, logModel))
        print ("alpha = %f\tsigma_x = %f\tsigma_a = %f"
               % (alpha, sigma_x, sigma_a))
        print ("=========================================================")

    for s in range(iteration):
        # Resume iterations if it starts from the saved chains
        if (resume and s < s_stopped+1):
            continue

        if (resume and s == s_stopped+1):
            time = datetime.datetime.now()
            output = "output_" + str(iteration) + "iterations_" + \
                time.strftime(("%m%d%y_%Hh%Mm%Ss")) + "_resumed"
            output_dir = os.path.abspath(output)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        # Store init params
        if (s == 0):
            m = (Z != 0).astype(np.int).sum(axis=0)
            K_real = len(m[m > 0])

            K_save[s] = K_real
            alpha_save[s] = alpha
            sigma_x_save[s] = sigma_x
            sigma_a_save[s] = sigma_a
            Z_save[:, :, s] = Z

            A_save[:, :, s] = A

            logModel = logPX(X, Z, A, sigma_x, N, D)
            logModel += logPA(A, sigma_a)
            logModel += logIBP(Z, alpha, K, N)
            if (realvaluedZ):
                logModel += logPV(Z)

            loglikelihood_save[s] = logModel

            print ("=========================================================")
            print ("Initializing.....................")
            print ("Emulate Infinite Factor Model with the upperbound of K = %d" % K)
            print ("Data shape: Obs = %d\t Dim = %d" % (N, D))
            print ("K = %d\tLikelihood = %f" % (K_real, logModel))
            print ("alpha = %f\tsigma_x = %f\tsigma_a = %f"
                   % (alpha, sigma_x, sigma_a))
            print ("=========================================================")

            # Create an output folder
            time = datetime.datetime.now()
            output = "output_" + str(iteration) + "iterations_" + \
                time.strftime(("%m%d%y_%Hh%Mm%Ss"))
            output_dir = os.path.abspath(output)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        # Update Z
        if (updateZ):
            K_old = np.copy(K)
            (Z, K, A) = sampleZ(X, Z, A, sigma_x, sigma_a, alpha, K, N, D,
                                realvaluedZ, proposeNewfeature)
            assert(A.shape[0] == Z.shape[1] == K_old)
            assert(A.shape[1] == D)

        # update A
        if (updateA):
            A = sampleA(X, Z, A, sigma_x, sigma_a, K, D)

        # update sigma using Gibbs sampler
        if (updateSigma_x):
            sigma_x = sampleSigma_x(X, Z, A, sigma_xa, sigma_xb, N, D)

        if (updateSigma_a):
            sigma_a = sampleSigma_a(X, Z, A, sigma_aa, sigma_ab, K, D)

        # Update alpha
        if (updateAlpha):
            alpha = sampleAlpha(alpha_a, alpha_b, K, N)

        # Latent feature count
        mcount = (Z != 0).astype(np.int).sum(axis=0)
        mcount_active = mcount[mcount > 0]

        # Store params
        K_real = len(mcount_active)
        K_save[s + 1] = K_real
        alpha_save[s + 1] = alpha
        sigma_x_save[s + 1] = sigma_x
        sigma_a_save[s + 1] = sigma_a
        Z_save[:, :, s + 1] = Z
        A_save[:, :, s + 1] = A

        # Log likelihood of full model
        logModel = logPX(X, Z, A, sigma_x, N, D)
        logModel += logPA(A, sigma_a)
        logModel += logIBP(Z, alpha, K, N)
        if (realvaluedZ):
            logModel += logPV(Z)
        print ("Iteration %d: K = %d\tlogLik = %f" % ((s+1), K_real, logModel))
        print ("alpha = %f\tsigma_x = %f\tsigma_a = %f"
               % (alpha, sigma_x, sigma_a))
        print ("\tLatent feature count: " + np.array_str(mcount_active))

        loglikelihood_save[s + 1] = logModel

        # Save the interim chain based on the predfined interval
        if save_interim:
            if ((s+1) % interim_interval == 0 and s != (iteration-1)):
                f = open("simulation.tmp", "wb")
                np.save(f, s)
                np.save(f, K_save)
                np.save(f, alpha_save)
                np.save(f, sigma_x_save)
                np.save(f, sigma_a_save)
                np.save(f, loglikelihood_save)
                np.save(f, Z_save)
                np.save(f, A_save)

        # save resulting outputs
        if (s == (iteration-1)):

            os.chdir(output_dir)

            np.savetxt("K_out.txt", K_save, delimiter=",")
            np.savetxt("alpha_out.txt", alpha_save, delimiter=",")
            np.savetxt("sigma_a_out.txt", sigma_a_save, delimiter=",")
            np.savetxt("sigma_x_out.txt", sigma_x_save, delimiter=",")
            np.savetxt("loglikelihood_model_out.txt", loglikelihood_save, delimiter=",")

            Afile = "A_out.mat"
            scipy.io.savemat(Afile, mdict={'A': A_save})

            Zfile = "Z_out.mat"
            scipy.io.savemat(Zfile, mdict={'Z': Z_save})

    return (K_save, alpha_save, sigma_x_save, sigma_a_save,
            loglikelihood_save, Z_save, A_save)
