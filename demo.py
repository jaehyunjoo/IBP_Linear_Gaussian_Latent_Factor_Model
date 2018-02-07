"""Demo for latent factor model"""
from __future__ import division
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from IBPFM import IBPFM
from utils.tracePlot import trace
from utils.scaledimage import scaledimage


N = 100
chain = 1000
K_finite = 6

# # read the keyboard input for the number of images
# N = raw_input("Enter the number of noisy images for learning features: ")

# try:
#     N = int(N)
# except ValueError:
#     print "Not a number"
#     sys.exit('Try again')

# # read the keyboard input for the number of MCMC chain
# chain = raw_input("Enter the number of MCMC chain: ")

# try:
#     chain = int(chain)
# except ValueError:
#     print "Not a number"
#     sys.exit('Try again')

# # read the keyboard input for the number of finite K
# K_finite = raw_input("Enter the finite number (upper bound) of features K: ")

# try:
#     K_finite = int(K_finite)
# except ValueError:
#     print "Not a number"
#     sys.exit('Try again')


# ------------------------------------------------------------------------------
# Model parameter
(alpha, alpha_a, alpha_b) = (1., 1., 1.)
(sigma_x, sigma_xa, sigma_xb) = (.5, 1., 1.)
(sigma_a, sigma_aa, sigma_ab) = (1., 1., 1.)


# ------------------------------------------------------------------------------
# Generate image data from the known features

feature1 = np.array([[0,1,0,0,0,0],[1,1,1,0,0,0],[0,1,0,0,0,0],\
                   [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
feature2 = np.array([[0,0,0,1,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1],\
                   [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
feature3 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],\
                   [1,0,0,0,0,0],[1,1,0,0,0,0],[1,1,1,0,0,0]])
feature4 = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],\
                   [0,0,0,1,1,1],[0,0,0,0,1,0],[0,0,0,0,1,0]])

D = 36
f1 = feature1.reshape(D)
f2 = feature2.reshape(D)
f3 = feature3.reshape(D)
f4 = feature4.reshape(D)
trueWeights = np.vstack((f1, f2, f3, f4))

# ------------------------------------------------------------------------------
# Generate noisy image data
K = 4
sig_x_true = 0.5
A = np.vstack((f1, f2, f3, f4)).astype(np.float)

Z_orig = nr.binomial(1, 0.5, (N, K)).astype(np.float)
V_orig = nr.normal(0, 1, size=(N, K))
# V_orig = nr.exponential(1, size=(N, K))
Z_orig = np.multiply(Z_orig, V_orig)
X = np.dot(Z_orig, A)
noise = nr.normal(0, sig_x_true, (N, D))
X += noise


# ------------------------------------------------------------------------------
# Return MCMC result
(K_save, alpha_save, sigma_x_save, sigma_a_save, loglikelihood_save, Z_save, A_save) = \
    IBPFM(iteration=chain, data=X, upperbound_K=K_finite,
          alpha=(alpha, alpha_a, alpha_b),
          sigma_x=(sigma_x, sigma_xa, sigma_xb),
          sigma_a=(sigma_a, sigma_aa, sigma_ab), realvaluedZ=True,
          proposeNewfeature=True,
          updateAlpha=True, updateSigma_x=True, updateSigma_a=True,
          initZ=None, stdData=False)

# Save trace plots
trace(K_save, alpha_save, sigma_x_save, sigma_a_save, loglikelihood_save)

# Save true latent feature plot
(orig, sub) = plt.subplots(1, 4)
for sa in sub.flatten():
    sa.set_visible(False)

orig.suptitle('True Latent Features')

for (i, true) in enumerate(trueWeights):
    ax = sub[i]
    ax.set_visible(True)
    scaledimage(true.reshape(6, 6), pixwidth=3, ax=ax)

orig.set_size_inches(13, 3)
orig.savefig('Original_Latent_Features.png')
plt.close()

# Save the posterior distribution of P(K | - )
burnin = np.rint((chain*0.5))
K_list = K_save[int(burnin+1):].astype(np.int)
hist_bin = range(min(K_list), max(K_list)+1)
hist_bin = np.asarray(hist_bin) - 0.5
plt.hist(K_save, bins=hist_bin, normed=1, facecolor="green", alpha=0.3)
plt.xlabel(r'$K^+$')
plt.ylabel("Density")
plt.savefig("Histogram_K")
plt.close()

# Save some of example figures from data X
examples = X[0:4, :]
(ex, sub) = plt.subplots(1, 4)
for sa in sub.flatten():
    sa.set_visible(False)
ex.suptitle('Image Examples')
for (i, true) in enumerate(examples):
    ax = sub[i]
    ax.set_visible(True)
    scaledimage(true.reshape(6, 6), pixwidth=3, ax=ax)

ex.set_size_inches(13, 3)
ex.savefig('Image_Examples.png')
plt.close()

# Show and save result
lastZ = Z_save[:, :, chain]
mcount = (lastZ != 0).astype(np.int).sum(axis=0)
index = np.where(mcount > 0)
lastK = K_save[chain].astype(np.int)
lastA = A_save[index, :, chain]
A = lastA.reshape(len(index[0]), D)

A_row = A.shape[0]
for i in range(A_row):
    cur_row = A[i, :].tolist()
    abs_row = [abs(j) for j in cur_row]
    max_index = abs_row.index(max(abs_row))
    if cur_row[max_index] < 0:
        A[i, :] = -np.array(cur_row)

K = max(len(trueWeights), len(A))
(fig, subaxes) = plt.subplots(2, K)
for sa in subaxes.flatten():
    sa.set_visible(False)
fig.suptitle('Ground truth (top) vs learned factors (bottom)')
for (idx, trueFactor) in enumerate(trueWeights):
    ax = subaxes[0, idx]
    ax.set_visible(True)
    scaledimage(trueFactor.reshape(6, 6),
                pixwidth=3, ax=ax)
for (idx, learnedFactor) in enumerate(A):
    ax = subaxes[1, idx]
    scaledimage(learnedFactor.reshape(6, 6),
                pixwidth=3, ax=ax)
    ax.set_visible(True)
fig.savefig("IBP_meanA.png")
plt.show()
