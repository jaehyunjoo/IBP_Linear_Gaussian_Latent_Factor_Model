"""This module will calculate the log likelihood of Gaussian feature weight."""
import numpy as np


def logPV(Z):
    """log likelihood for V prior Normal."""
    mean = 0  # default
    sd = 1  # default: change for your needs

    prior = 0
    for (i, k) in zip(*Z.nonzero()):
        prior += -0.5*np.log(2*np.pi*(sd**2))- 0.5*((Z[i, k]-mean)**2)/float(sd**2)
    return prior


def logPvi(v):
    mean = 0  # default
    sd = 1  # default: change for your needs

    """log likelihood for Vi (single element)."""
    lp = -0.5*np.log(2*np.pi*(sd**2)) - 0.5*((v-mean)**2)/float(sd**2)
    return lp
