import os
import sys
import logging
import argparse
import pathlib

import numpy as np

sys.path.insert(0, 'theory')

# compute_overlaps1 is a 3d Gaussian integral formulation
# compute_overlaps2 is a 2d Gaussian integral formulation
# They are equivalent, but may differ due to numerical error
from overlaps import overlaps_erf_bilinear as compute_overlaps1
from overlaps_fast import overlaps_erf_bilinear as compute_overlaps2

def main(args):

    n_cpu = args.num_cpus
    P = args.patterns
    theta = args.theta
    sigma = args.sigma
    alpha_lower = args.alpha_lower
    alpha_upper = args.alpha_upper
    alpha_final = np.NaN
    omega = args.omega
    eps = args.epsilon
    formulation = args.mf_formulation

    # Criteria to determine if sequence retrieval is present
    def criterion_1(m_max):
        "The sequence of maximal overlaps {max_t(m_\mu)} is non-monotonically decaying"
        return np.any(np.diff(m_max) > 0)
    def criterion_2(rho_max):
        "The final maximum correlation rho_max = max_t(rho_P) is above a threshold omega"
        return rho_max > omega

    # Select desired criterion
    if args.criterion == 1:
        criterion = criterion_1
    elif args.criterion == 2:
        criterion = criterion_2
    else:
        raise Exception("Valid criterion not specified")

    fargs = (P, theta, sigma, args.criterion, formulation, n_cpu)

    # Bisection method to find maximal alpha for which criterion is valid
    n = 0
    while True:
        alpha_mid = (alpha_lower + alpha_upper)/2.
        logging.info("Computing alpha_mid=%f" % alpha_mid)
        logging.info("Computing alpha_lower=%f" % alpha_lower)
        logging.info("Computing alpha_upper=%f" % alpha_upper)

        # Compute midpoint
        seq_mid = criterion(compute_max_overlaps(alpha_mid, *fargs))
        logging.info("Seq alpha_mid=%i" % seq_mid)

        # Compute lower bound (if necessary)
        if n == 0:
            seq_lower = criterion(compute_max_overlaps(alpha_lower, *fargs))
        logging.info("Seq alpha_lower=%i" % seq_lower)

        # Compute upper bound (if necessary)
        if n == 0:
            seq_upper = criterion(compute_max_overlaps(alpha_upper, *fargs))
        logging.info("Seq alpha_upper=%i" % seq_upper)

        if seq_lower and not seq_upper:
            if (alpha_upper - alpha_lower) < eps:
                alpha_final = alpha_lower
                break
            else:
                if seq_mid:
                    alpha_lower = alpha_mid
                else:
                    alpha_upper = alpha_mid
        else:
            # Bounds do not contain capacity
            break

        n += 1

    if args.save:
        logging.info("Saving data")
        if args.criterion == 1:
            datapath = args.datapath % ("theta_%0.2f_sigma_%0.2f_P_%i" % \
                        (theta, sigma, P))
        else:
            datapath = args.datapath % ("theta_%0.2f_sigma_%0.2f_P_%i_crit_%i" % \
                        (theta, sigma, P, args.criterion))
        np.save(open(datapath, "wb"), [theta, sigma, P, alpha_final])


def compute_max_overlaps(alpha, P, theta, sigma, criterion, formulation, n_workers=8):
    "Compute the sequence of maximal overlaps or correlations"
    
    A = 1
    tau = 1e-2
    dt = 1e-3
    T = P*tau

    # Terminate once we have reached retrieval time (the maximum of the final overlap)
    def terminate_on(t, m, M, C, rbar):
        "Early termination condition"
        if t > T:
            m_P = m[-1,:]
            if m_P[-2] > m_P[-1]:
                return True
        return False

    # Which mean-field formulation to use
    if formulation == 1:
        compute_overlaps = compute_overlaps1
    elif formulation == 2:
        compute_overlaps = compute_overlaps2
    else:
        raise Exception("Valid formulation not specified")

    m, M, _, rbar = compute_overlaps(
        np.around(T*1.1,2),
        P,
        alpha,
        A,
        theta,
        sigma,
        tau,
        dt,
        n_workers,
        terminate_on)

    # Return sequence of maximal overlaps or correlations
    if int(criterion) == 1:
        return m.max(axis=1)
    elif int(criterion) == 2:
        rho = m/np.sqrt(M-rbar**2)
        value =  np.nanmax(rho[-1,:])
        logging.info("Value=%f" % value[-1])
        return value
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-theta', '--theta', type=float)
    parser.add_argument('-sigma', '--sigma', type=float)
    parser.add_argument('-P', '--patterns', type=int, default=8)

    parser.add_argument('-alpha_lower', '--alpha_lower', type=float)
    parser.add_argument('-alpha_upper', '--alpha_upper', type=float)

    parser.add_argument('-crit', '--criterion', type=int, default=1)
    parser.add_argument('-mf', '--mf_formulation', type=int, default=1)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.05)
    parser.add_argument('-omega', '--omega', type=float, default=0.01)

    parser.add_argument('-c', '--num_cpus', type=int, default=8)
    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-d', '--datapath', default='figures/article/3/data/data_b_mf.%s.npy')
    args = parser.parse_args()

    main(args) 
