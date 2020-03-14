import os
import sys
import logging
import argparse
import pathlib

import numpy as np

sys.path.insert(0, 'theory')
from overlaps import overlaps_erf_bilinear as compute_overlaps

def main(args):

    mu = 0.22
    sigma = 0.10

    N = 40000
    c = 0.005
    K = N*c
    P = np.asarray(args.patterns, dtype=int)
    alpha = np.r_[np.arange(1,25,2)*8/K, 0.48]

    for i in range(P.size):
        logging.info("Computing P=%i" % P[i])
        if args.alpha_index >= 0:
            # Compute and save for single alpha
            logging.info("Computing alpha=%f" % alpha[args.alpha_index])
            datapath = args.datapath % (
                "theta_%0.2f_sigma_%0.2f_P_%i_alpha_%i" % (
                    mu,sigma,P[i],args.alpha_index))
            data = compute_max_correlation(
                P[i],
                mu,
                sigma,
                alpha[args.alpha_index], 
                args.num_cpus)
        else:
            datapath = args.datapath % ("theta_%0.2f_sigma_%0.2f_P_%i"%(mu,sigma,P[i]))
            data = []
            # Compute and save for range of alphas
            for j in range(alpha.size):
                logging.info("Computing alpha=%f" % alpha[j])
                data.append(compute_max_correlation(P[i], mu, sigma, alpha[j], args.num_cpus))
        if args.save:
            logging.info("Saving data")
            np.save(open(datapath, "wb"), data)

def compute_max_correlation(P, mu, sigma, alpha, n_workers=8):
    
    A = 1
    tau = 1e-2
    dt = 1e-3
    T = P*tau

    def terminate_on(t, m, M, C, rbar):
        """
        Early termination condition
        """
        if t > T:
            rho = m[-1,:] / np.sqrt(M-rbar**2)
            if rho[-2] > rho[-1]:
                return True
        return False

    m, M, _, rbar = compute_overlaps(
        np.around(T*1.1,2),
        P,
        alpha,
        A,
        mu,
        sigma,
        tau,
        dt,
        n_workers,
        terminate_on)

    return m, M, rbar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--num_cpus', type=int, default=20)
    parser.add_argument('-P', '--patterns', nargs='+', default=[8,16,32,64,128])
    parser.add_argument('-i', '--alpha_index', type=int, default=-1)
    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-d', '--datapath', default='figures/article/3/data/data_c.%s.npy')
    args = parser.parse_args()

    main(args)