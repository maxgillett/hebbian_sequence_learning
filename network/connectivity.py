import pdb
import logging
from numba import jit, njit
import numpy as np
import scipy.sparse
from tqdm import trange
from scipy.stats import lognorm, norm
from itertools import cycle

logger = logging.getLogger(__name__)

class Connectivity(object):
    def __init__(self):
        self.W = None
        self.disable_pbar = False

    def scale_all(self, value):
        self.W *= value

    def set_connectivity(self):
        raise NotImplementedError

    def reset_connectivity(self):
        raise NotImplementedError

    @staticmethod
    def _store_sequences(ij, inputs, f, g, disable_pbar=False):
        """
        inputs: S x P x N
        Store heteroassociative connectivity
        """
        S, P, N  = inputs.shape
        row = []
        col = []
        data = []
        for n in trange(len(inputs), disable=disable_pbar):
            seq = inputs[n]
            for j in trange(seq.shape[1], disable=disable_pbar):
                i = ij[j]
                # $f(xi_i^{\mu+1}) * g(xi_j^{\mu})$
                w = np.sum(f(seq[1:,i]) * g(seq[:-1,j][:,np.newaxis]), axis=0) 
                row.extend(i)
                col.extend([j]*len(i))
                data.extend(w)
        return data, row, col


    @staticmethod
    def _store_attractors(ij, inputs, f, g, disable_pbar=False):
        """
        inputs: P x N
        Store autoassociative connectivity
        """
        P, N  = inputs.shape
        row = []
        col = []
        data = []
        for j in trange(inputs.shape[1], disable=disable_pbar):
            i = ij[j]
            w = np.sum(f(inputs[:,i]) * g(inputs[:,j][:,np.newaxis]), axis=0) 
            row.extend(i)
            col.extend([j]*len(i))
            data.extend(w)
        return data, row, col

    @staticmethod
    def _set_all(ij, value):
        row = []
        col = []
        data = []
        for j in range(len(ij)):
            i = ij[j]
            row.extend(i)
            col.extend([j]*len(i))
            data.extend([value]*len(i))
        return data, row, col
                

class SparseConnectivity(Connectivity):
    def __init__(self, source, target, p=0.005, fixed_degree=False, seed=42, disable_pbar=False):
        self.W = scipy.sparse.csr_matrix((target.size, source.size), dtype=np.float32)
        self.p = p
        self.K = p*target.size
        self.ij = []
        self.disable_pbar = disable_pbar
        logger.info("Building connections from %s to %s" % (source.name, target.name))
        if fixed_degree:
            n_neighbors = np.asarray([int(p*target.size)]*source.size)
        else:
            n_neighbors = np.random.RandomState(seed).binomial(target.size, p=p, size=source.size)

        @njit
        def func(source_size, target_size):
            np.random.seed(seed)
            ij = []
            for j in range(source_size): # j --> i
                # Exclude self-connections
                arr = np.arange(target_size)
                arr2 = np.concatenate((arr[:j], arr[j+1:]))
                j_subset = np.random.choice(arr2, size=n_neighbors[j], replace=False)
                ij.append(j_subset)
            return ij

        self.ij = func(source.size, target.size)

    def store_sequences(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        N = inputs.shape[2]
        logger.info("Storing sequences")
        data, row, col = Connectivity._store_sequences(self.ij, inputs, f, g, self.disable_pbar)
        logger.info("Applying synaptic transfer function")
        #pdb.set_trace()
        data = h(data)
        logger.info("Building sparse matrix")
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def store_attractors(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        logger.info("Storing attractors")
        data, row, col = Connectivity._store_attractors(self.ij, inputs, f, g, self.disable_pbar)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def store_gaussian(self, mu=0, sigma=1, seed=2):
        w = mu + sigma*np.random.RandomState(seed).randn(self.W.data.size)
        self.W.data[:] += w

    def set_random(self, var, h=lambda x:x):
        data, row, col = Connectivity._set_all(self.ij, 1)
        data = np.asarray(data, dtype=float)
        data[:] = np.sqrt(var)*np.random.randn(data.size)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def set_weights(self, data, row, col):
        "NOTE: Adds to, but does not overwrite existing weights"
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def set_all(self, value):
        "NOTE: Adds to, but does not overwrite existing weights"
        data, row, col = Connectivity._set_all(self.ij, value)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()


class DenseConnectivity(Connectivity):
    def __init__(self, source, target, seed=42, disable_pbar=False):
        self.disable_pbar = disable_pbar
        self.W = np.zeros((target.size, source.size), dtype=np.float32)
        self.K = target.size-1

    def store_sequences(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        """
        inputs: S x P x N
        """
        N = inputs.shape[2]
        logger.info("Storing sequences")
        for n in trange(len(inputs), disable=self.disable_pbar):
            seq = inputs[n]
            for mu in trange(seq.shape[0]-1, disable=self.disable_pbar):
                W = h(np.outer(f(seq[mu+1,:]), g(seq[mu,:])))
                diag = np.diagonal(W)
                diag.setflags(write=True)
                diag.fill(0)
                self.W += W

    def store_attractors(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        pass

    def set_weights(self, data, row, col):
        pass

    def set_all(self, value):
        pass


class ThresholdPlasticityRule(object):
    def __init__(self, x_f, q_f, x_g=None, rv=scipy.stats.norm):
        if not x_g:
            x_g = x_f
        q_g = rv.cdf(x_g)
        self.x_f, self.q_f = x_f, q_f
        self.x_g, self.q_g = x_g, q_g
        self.f = lambda x: np.where(x < x_f, -(1-q_f), q_f)
        self.g = lambda x: np.where(x < x_g, -(1-q_g), q_g)


class SynapticTransferFunction(object):
    def __init__(self, K):
        pass

class LinearSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, A):
        super(LinearSynapse, self).__init__(self)
        self.A = A
        self.K_EE = K_EE

    def h_EE(self, J):
        return self.A * np.asarray(J) / self.K_EE

    def h(self, J):
        return self.h_EE(J)


class RectifiedSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, K_IE, K_EI, K_II, alpha, plasticity, A=1, g=1., o=0):
        super(RectifiedSynapse, self).__init__(self)

        @np.vectorize
        def rectify(x):
            if x<0:
                return 0
            else:
                return x

        self.K_EE, self.K_IE, self.K_EI, self.II = K_EE, K_IE, K_EI, K_II
        f_fun, x_f, q_f = plasticity.f, plasticity.x_f, plasticity.q_f
        g_fun, x_g, q_g = plasticity.g, plasticity.x_g, plasticity.q_g
        self.A, self.g, self.o, = A, g, o

        gamma = norm.expect(lambda x: f_fun(x)**2) * \
                norm.expect(lambda x: g_fun(x)**2)
        self.E_w = norm.expect(
                lambda x: rectify(g*A*np.sqrt(alpha*gamma)*x + o),
                scale=1)
        self.E_w_2 = norm.expect(
                lambda x: rectify(g*A*np.sqrt(alpha*gamma)*x + o)**2,
                scale=1)
    
    def h_EE(self, J):
        return self.A * (self.g * (J / np.sqrt(self.K_EE)) + self.o).clip(min=0) / np.sqrt(self.K_EE)

    def h_IE(self, J):
        return J * self.E_w / self.K_IE

    def h_EI(self, J):
        return -J * 1. / np.sqrt(self.K_EI)

    def h_II(self, J):
        return -J


class ExponentialSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, K_IE, K_EI, K_II, alpha, plasticity, A=1, g=1., o=0):
        super(ExponentialSynapse, self).__init__(self)

        @np.vectorize
        def exp(x):
            return np.exp(x)

        self.K_EE, self.K_IE, self.K_EI, self.II = K_EE, K_IE, K_EI, K_II
        f_fun, x_f, q_f = plasticity.f, plasticity.x_f, plasticity.q_f
        g_fun, x_g, q_g = plasticity.g, plasticity.x_g, plasticity.q_g
        self.A, self.g, self.o, = A, g, o

        gamma = norm.expect(lambda x: f_fun(x)**2) * \
                norm.expect(lambda x: g_fun(x)**2)
        self.E_w = norm.expect(
                lambda x: exp(g*A*np.sqrt(alpha*gamma)*x + o),
                scale=1)
        self.E_w_2 = norm.expect(
                lambda x: exp(g*A*np.sqrt(alpha*gamma)*x + o)**2,
                scale=1)
    
    def h_EE(self, J):
        return self.A * (self.g * (J / np.sqrt(self.K_EE)) + self.o).clip(min=0) / np.sqrt(self.K_EE)

    def h_IE(self, J):
        return J * self.E_w / self.K_IE

    def h_EI(self, J):
        return -J * 1. / np.sqrt(self.K_EI)

    def h_II(self, J):
        return -J

