import sys
import logging
import argparse

import numpy as np

sys.path.insert(0, 'theory')
from overlaps import overlaps_erf_bilinear

logging.basicConfig(level=logging.INFO)

def main(args):
    
    T = 0.30
    S = 2
    P = 16
    N = 40000
    c = 0.005
    alpha = S*P/(N*c)
    A = 1
    mu = 0.22
    sigma = 0.1
    tau = 1e-2
    dt = 1e-3

    m, M, C = overlaps_erf_bilinear(T,P,alpha,A,mu,sigma,tau,dt)

    if args.plot:
        pass

    if args.save:
        logging.info("Saving data")
        np.save(open(args.datapath, "wb"), [m, M, C])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-d', '--datapath', default='figures/article/3/data/data_a_mf.npy')
    args = parser.parse_args()

    main(args) 
