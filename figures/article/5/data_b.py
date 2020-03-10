import argparse
import pdb
import sys
import os
import logging
import copy
import signal
import pathos.multiprocessing
import numpy as np
import scipy
from apply import apply

sys.path.insert(0, 'network')
from network import Population, RateNetwork
from transfer_functions import ErrorFunction
from connectivity import SparseConnectivity, LinearSynapse, ThresholdPlasticityRule
from sequences import GaussianSequence

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
    
N = 40000
T = 0.4
S, P = 1, 30

n_days = 30
n_realizations = 1

sigma_A = 0.7
tau_decay = 0.2

def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting")
    
    exc = Population(N, tau=1e-2, phi=ErrorFunction(mu=0.07, sigma=0.05).phi)

    conn = SparseConnectivity(source=exc, target=exc, p=0.005, seed=42)
    sequences = [GaussianSequence(P,exc.size,seed=i) for i in range(S)]
    patterns = np.stack([s.inputs for s in sequences])

    plasticity = ThresholdPlasticityRule(x_f=1.645, q_f=0.8)
    synapse = LinearSynapse(conn.K, A=14)
    
    # Store fixed sequential component
    conn.store_sequences(patterns, synapse.h_EE, plasticity.f, plasticity.g)
    W_sequence = conn.W.copy()
    print(W_sequence.data.std())

    # Define lambda and sigma_z (see paper)
    std_W = W_sequence.data.std()
    lambda_ = np.exp(-1/(n_days*tau_decay))
    sigma_z = sigma_A/np.sqrt(1-lambda_**2)*conn.W.data.std()
    print(sigma_z)

    # Keep track of state and overlap values
    state = n_realizations*[[]]
    overlaps = n_realizations*[[]]
    
    def build_conn_random(n, m):
        "Derive the random connectivity component for day n"
        # Intermediate products were stored for debugging
        # Todo: Enable option for lower memory usage (don't store intermediate values in loop)
        rng = np.random.RandomState(seed=43)
        data = []
        for i in range(n+1):
            data.append(rng.normal(scale=1, size=(1, conn.W.data.size)))
        data = np.vstack(data)
        data *= np.sqrt(1-lambda_**2)*sigma_z
        data *= np.asarray([lambda_**i for i in range(n+1)])[::-1][:,np.newaxis]
        data[0,:] /= np.sqrt(1-lambda_**2)
        data = data.sum(axis=0)
        row = conn.W.tocoo().row
        col = conn.W.tocoo().col
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float16).tocsr()
        return W
    
    def simulate_day(n, l):
        "Simulate sequential activity recalled during day n, realization l"
        logging.info("Simulating day %i"%n)
        W_random = build_conn_random(n, l)
        conn_n = copy.deepcopy(conn)
        conn_n.W = W_sequence + W_random
        net = RateNetwork(exc, c_EE=conn_n, formulation=1)
        net.simulate(T, r0=exc.phi(plasticity.f(patterns[0,0,:])))
        m = sequences[0].overlaps(net, exc, plasticity=plasticity, correlation=True)
        logging.info("Finished simulating day %i (realization %i)"%(n,l))
        return net.exc.state.T, m, l
    
    if args.parallel:
        print("Initializing workers")
        pool = pathos.multiprocessing.Pool(processes=args.n_workers)
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        results = []
        for l in range(n_realizations):
            for n in range(0, n_days):
                print("Starting day %i (realization %i)"%(n,l))
                if args.parallel: res = pool.apply_async(simulate_day, (n,l))
                else:             res = apply(simulate_day, (n,l))
                results.append(res)
        for i, res in enumerate(results):
            if args.parallel: s, m, l = res.get()
            else:             s, m, l = res
            state[l].append(s.astype(np.float16))
            overlaps[l].append(m.astype(np.float16))
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating")
        if args.parallel: pool.terminate()
    else:
        print("Normal termination")
        if args.parallel: pool.close()
    if args.parallel: pool.join()

    if args.save:
        logging.info("Saving data")
        np.save(open(args.data_path, "wb"), [state, overlaps])

def print_conversion(args):
    lambda_ = np.exp(-1/(n_days*tau_decay))
    sigma_z = sigma_A/(np.sqrt(1-lambda_**2))

    exc = Population(N, tau=1e-2, phi=ErrorFunction(mu=0.07, sigma=0.05).phi)
    conn = SparseConnectivity(source=exc, target=exc, p=0.005, seed=42)
    sequences = [GaussianSequence(P,exc.size,seed=i) for i in range(S)]
    patterns = np.stack([s.inputs for s in sequences])
    plasticity = ThresholdPlasticityRule(x_f=1.645, q_f=0.8)
    synapse = LinearSynapse(conn.K, A=14)
    conn.store_sequences(patterns, synapse.h_EE, plasticity.f, plasticity.g)

    A_base = conn.W.data.std()*sigma_z
    print("lambda", lambda_)
    print("sigma_z", sigma_z)
    print("A_base", A_base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--print_conversion', default=False)
    parser.add_argument('-p', '--parallel', action='store_true')
    parser.add_argument('-n', '--n_workers', default=8)
    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-d', '--data_path', default='figures/article/5/data/data_b.npy')
    args = parser.parse_args()

    if args.print_conversion:
        print_conversion(args)
    else:
        main(args) 
