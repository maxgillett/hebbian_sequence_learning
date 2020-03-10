import numpy as np

class Noise(object):
    def __init__(self, seed):
        self.rand = np.random.RandomState(seed)

class WhiteNoise(Noise):
    "White noise generated from a standard normal distribution"

    def __init__(self, seed=42):
        super(WhiteNoise, self).__init__(seed)

    def value(self, dt, tau, N):
        return np.sqrt(dt/tau)*self.rand.randn(1)
