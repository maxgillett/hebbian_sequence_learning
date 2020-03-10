import pdb
import argparse
import os
import re
import tempfile
import itertools
import subprocess
import numpy as np
from scipy import interpolate
from subprocess import call

# Mean field formulation + retrieval criterion
mf = 2
criterion = 1

# Cpu/thread count
n_cpu = 10
n_threads = 16

# Error tolerance (critical alpha)
eps = 0.01

# Correlation cutoff threshold
omega = 0.05

# Default parameters
P = 64
theta = [-0.2,0,-0.2]
sigma = np.arange(0.05,0.35,0.05)
datapath = 'figures/article/3/data/data_b.npy'
(theta_c, sigma_c), alpha_c = np.load(open(datapath, "rb"), allow_pickle=True)
alpha_analytic = interpolate.interp2d(theta_c, sigma_c, z=alpha_c.T)
params = dict()
for theta_,sigma_ in itertools.product(theta, sigma):
    alpha_true = alpha_analytic(theta_,sigma_)
    alpha_upper = alpha_true*1.1
    alpha_lower = alpha_upper * 0.6 # Set lower bound as a fraction of upper
    params[(theta_, sigma_)] = (alpha_lower, alpha_upper)

# Custom parameters (Override default alpha boundaries)
custom_params = {
    # (theta, sigma) : (alpha_lower, alpha_upper)
    #(0.2, 0.05): (0.4,0.45), 
    #(0.2, 0.10): (0.45,0.48), 
    #(0.2, 0.15): (0.45,0.4625), 
}
params.update(custom_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-slurm', '--slurm_submit', action="store_true")
    args = parser.parse_args()

    for (theta,sigma), (alpha_lower, alpha_upper) in params.items():
        print(theta, sigma)
        id_str = "theta_%.2f_sigma_%.2f_P_%i_crit_%i" % (theta, sigma, P, criterion)
        executable_str = "python3 figures/article/3/data_b_mf.py \
                -theta '%.2f' \
                -sigma '%.2f' \
                -alpha_lower '%f' \
                -alpha_upper '%f' \
                -P %i \
                -eps '%0.2f' \
                -omega '%0.2f' \
                -crit %i \
                -mf %i \
                -c %i\n" % (
                    theta,
                    sigma,
                    alpha_lower,
                    alpha_upper,
                    P,
                    eps,
                    omega,
                    criterion,
                    mf,
                    n_threads)
        # TODO: Automatically call module load before running
        # Need to manually call at the moment.
        if args.slurm_submit:
            tmp = tempfile.NamedTemporaryFile()
            with open(tmp.name, 'w') as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --job-name=job_%s\n"%id_str)
                fh.writelines("#SBATCH --output=logs/output_%s.out\n"%id_str)
                fh.writelines("#SBATCH -N 1\n")
                fh.writelines("#SBATCH -c %i\n"%n_cpu)
                fh.writelines("#SBATCH -t 2880\n")
                fh.writelines("#SBATCH --mem=16000\n")
                fh.writelines("#SBATCH -e logs/err_%s.out\n"%id_str)
                fh.writelines("#module load Python/3.6.4\n")
                fh.writelines(executable_str)
            call("sbatch %s"%tmp.name, shell=True)
        else:
            logfile = "logs/output_%s.out"%id_str
            f = open(logfile, "w")
            subprocess.call([re.sub(' +', ' ', executable_str.rstrip())], stdout=f, shell=True)
    
