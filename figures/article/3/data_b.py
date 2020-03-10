import sys
import argparse
import logging

import numpy as np

sys.path.insert(0, '.')
from theory.capacity import capacity

def main(args):
    mu = np.asarray([-0.2, 0, 0.2])
    sigma = np.linspace(0.000, 0.35, 30)[::-1]
    alpha = capacity(mu, sigma)

    if args.save:
        logging.info("Saving data")
        np.save(open(args.datapath, "wb"), [[mu, sigma], alpha])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-d', '--datapath', default='figures/article/3/data/data_b.npy')
    args = parser.parse_args()

    main(args) 






