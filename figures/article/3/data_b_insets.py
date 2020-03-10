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

    def simulate(S, P, T=0.25):
        phi = ErrorFunction(mu=0.22, sigma=0.1).phi
        exc = Population(N=40000, tau=1e-2, phi=phi)

        sequences = [GaussianSequence(P,exc.size,seed=i) for i in range(S)]
        patterns = np.stack([s.inputs for s in sequences])

        conn_EE = SparseConnectivity(source=exc, target=exc, p=0.005, seed=43)
        synapse = LinearSynapse(conn_EE.K, A=1)
        conn_EE.store_sequences(patterns, synapse.h_EE)

        net = RateNetwork(exc, c_EE=conn_EE, formulation=1)
        net.simulate(T, r0=exc.phi(patterns[0,0,:]))

        correlations = sequences[0].overlaps(net, exc, correlation=True)
        return correlations


    correlations1 = simulate(S=4, P=16)
    correlations2 = simulate(S=8, P=16)

    if args.save:
        logging.info("Saving data")
        np.save(open(args.datapath, "wb"), [correlations1, correlations2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-d', '--datapath', default='figures/article/3/data/data_b_insets.npy')
    args = parser.parse_args()

    main(args) 

