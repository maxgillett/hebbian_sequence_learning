import sys
import logging
import argparse
import numpy as np

sys.path.insert(0, 'network')
from network import Population, RateNetwork
from transfer_functions import ErrorFunction
from connectivity import SparseConnectivity, LinearSynapse
from sequences import GaussianSequence

logging.basicConfig(level=logging.INFO)

def main(args):
    
    T = 0.5

    phi = ErrorFunction(mu=0.22, sigma=0.1).phi
    exc = Population(N=40000, tau=1e-2, phi=phi)

    S, P = 2, 16
    sequences = [GaussianSequence(P,exc.size,seed=i) for i in range(S)]
    patterns = np.stack([s.inputs for s in sequences])

    conn_EE = SparseConnectivity(source=exc, target=exc, p=0.005)
    synapse = LinearSynapse(conn_EE.K, A=1)
    conn_EE.store_sequences(patterns, synapse.h_EE)

    net = RateNetwork(exc, c_EE=conn_EE, formulation=1)

    net.simulate(0.25, r0=exc.phi(patterns[0,0,:]))
    #r0 = patterns[0,0,:]
    #net.simulate(0.01, r0=np.zeros_like(r0), r_ext=r0)
    #net.simulate(0.24, r0=net.exc.state[:,-1])
    r0 = patterns[1,0,:]
    net.simulate(0.01, r0=net.exc.state[:,-1], r_ext=r0)
    net.simulate(0.24, r0=net.exc.state[:,-1])
    #net.simulate(0.25, r0=exc.phi(patterns[1,0,:]))

    M = np.mean(net.exc.state**2, axis=0)
    overlaps1 = sequences[0].overlaps(net, exc)
    overlaps2 = sequences[1].overlaps(net, exc)

    #if args.save:
    #    logging.info("Saving data")
    #    np.save(open(args.data_path, "wb"), [M, overlaps1, overlaps2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-d', '--data_path', default='figures/article/3/data/data_a_net.2.npy')
    args = parser.parse_args()

    main(args) 

