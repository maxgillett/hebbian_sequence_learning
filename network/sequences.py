import logging
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from helpers import spike_to_rate

logger = logging.getLogger(__name__)

class Sequence(object):
    def __init__(self):
        "P x N"
        pass

    def overlaps(self, net, pop, phi=None, plasticity=None, spikes=False, correlation=False, disable_pbar=False):
        if correlation:
            logger.info("Computing correlations")
        else:
            logger.info("Computing overlaps")
        overlaps = []
        inputs = self.inputs
        if phi:
            inputs = phi(inputs)
        if plasticity:
            inputs = plasticity.g(inputs)
        for pattern in tqdm(inputs, disable=disable_pbar):
            if correlation:
                if spikes:
                    rate = spike_to_rate(pop.spikes)
                    overlap = np.asarray(
                        [pearsonr(pattern, rate[:,t])[0] for t in range(rate.shape[1])])
                else:
                    overlap = np.asarray(
                        [pearsonr(pattern, pop.state[:,t])[0] for t in range(net.exc.state.shape[1])])
            else:
                overlap = net.overlap_with(pattern, pop, spikes)
            overlaps.append(overlap)
        return np.vstack(overlaps)


class GaussianSequence(Sequence):
    def __init__(self, S, N, seed=42):
        super(Sequence, self).__init__()
        self.inputs = np.random.RandomState(seed).normal(0,1,size=(S,N))
