"""This module will draw several trace plots for the result."""
import matplotlib.pyplot as plt


def trace(K_save, alpha_save, sigma_x_save, sigma_a_save, loglikelihood_save):
    """Draw traceplots for K, alpha, sigma_x, and sigma_a."""
    trace = plt.figure(figsize=(10, 10))
    trace_K = trace.add_subplot(511)
    trace_K.plot(K_save)
    trace_K.set_ylabel(r'$K^+$')
    trace_alpha = trace.add_subplot(512)
    trace_alpha.plot(alpha_save)
    trace_alpha.set_ylabel(r'$\alpha$')
    trace_sigx = trace.add_subplot(513)
    trace_sigx.plot(sigma_x_save)
    trace_sigx.set_ylabel(r'$\sigma_x$')
    trace_siga = trace.add_subplot(514)
    trace_siga.plot(sigma_a_save)
    trace_siga.set_ylabel(r'$\sigma_a$')
    trace_loglikelihood = trace.add_subplot(515)
    trace_loglikelihood.plot(loglikelihood_save)
    trace_loglikelihood.set_ylabel('log likelihood')
    trace.savefig("Trace_params")
    plt.close('all')
