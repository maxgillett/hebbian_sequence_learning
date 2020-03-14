from __future__ import division
import itertools

import numpy as np
from numpy import exp, sqrt, pi
from scipy.special import erf
from scipy.stats import norm

from tqdm import trange


def g(M, params):
    mu, sigma, alpha_eff = params
    alpha = alpha_eff / M
    R = np.sqrt(alpha * M)
    phif = lambda x: 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)))
    return M - norm.expect(lambda v: phif(v * R)**2, scale=1)


def integrate_M(M, params, dt=0.2):
    return M + -dt * g(M, params)

def find_M(M, params):
    eps = 1e-8
    while True:
        temp = integrate_M(M, params)
        if np.sum((temp - M)**2) < eps: break
        M = temp
    return M

def h(alpha, params):
    'effective alpha'
    mu, sigma = params
    return 1 - exp(-mu**2 /
                   (2 * (alpha + sigma**2))) / sqrt(2 * pi *
                                                        (alpha + sigma**2))

def integrate_G(alpha, params, dt=0.10):
    return alpha + -dt * h(alpha, params)

def effective_alpha(a, params):
    eps = 1e-8
    while True:
        temp = integrate_G(a, params)
        if np.sum((temp - a)**2) < eps: break
        if np.isnan(temp): return 0
        if temp < 0: return 0
        a = temp
    return a

def capacity(mu=np.asarray([0.2]), sigma=np.linspace(0.000, 0.35, 30)[::-1], debug=False):
    params = (mu, sigma)
    alpha = np.zeros((mu.size, sigma.size))
    alpha[:] = np.NaN
    for i in trange(len(mu)):
        for j in range(len(sigma)):
            alpha_eff = effective_alpha(6, (mu[i], sigma[j]))
            M = find_M(0.5, (mu[i], sigma[j], alpha_eff))
            alpha[i,j] = alpha_eff / M
            if debug:
                print(mu[i], sigma[j], alpha[i,j])
    return alpha

def M_c(mu, sigma):
    alpha_eff = effective_alpha(6, (mu, sigma))
    M = find_M(0.5, (mu, sigma, alpha_eff))
    return M