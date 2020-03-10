import pdb
import logging
import numpy as np
import scipy.sparse
import scipy.integrate
from tqdm import tqdm, trange
import progressbar
from numba import jit, njit
from connectivity import Connectivity
from helpers import spike_to_rate

logger = logging.getLogger(__name__)

class Population(object):
    def __init__(self, N, tau, phi=lambda x:x, name="exc"):
        self.name = name
        self.size = N
        self.state = np.array([], ndmin=2).reshape(N,0)
        self.field = np.array([], ndmin=2).reshape(N,0)
        self.tau = tau
        self.phi = phi


class SpikingNeurons(Population):
    def __init__(self, N, tau_mem, tau_syn, thresh, reset, name="exc"):
        super(SpikingNeurons, self).__init__(N, None, None)
        self.name = name
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.thresh = thresh
        self.reset = reset


class Network(object):
    def __init__(self, exc, inh=None, 
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity()):
        """
        """
        self.c_EE = c_EE
        self.c_IE = c_IE
        self.c_EI = c_EI
        self.c_II = c_II
        self.exc = exc
        self.inh = inh
        self.size = exc.size
        self.t = np.array([])
        self.W_EE = c_EE.W
        self.W = scipy.sparse.bmat([
            [c_EE.W, c_EI.W],
            [c_IE.W, c_II.W]
        ]).tocsr()
        if inh:
            self.W_EI = c_EI.W
            self.W_IE = c_IE.W
            self.W_II = c_II.W
            self.tau = np.concatenate([
                np.array([exc.tau]*exc.size),
                np.array([inh.tau]*inh.size)
            ])
            self.size += self.inh.size
        else:
            self.tau = exc.tau
        self.xi = None
        self.r_ext = 0


class RateNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity(),
            formulation=1,
            disable_pbar=False):
        super(RateNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)
        self.formulation = formulation
        self.disable_pbar = disable_pbar
        if self.formulation == 1:
            self._fun = self._fun1
        elif self.formulation == 2:
            self._fun = self._fun2
        elif self.formulation == 3:
            self._fun = self._fun3

    def simulate(self, t, r0, t0=0, dt=1e-3, r_ext=lambda t: 0):
        """
        Runge-Kutta 2nd order
        """
        logger.info("Integrating network dynamics")
        if self.disable_pbar:
            pbar = progressbar.NullBar()
        else:
            pbar = progressbar.ProgressBar(
                maxval=t,
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        pbar.start()
        self.r_ext = r_ext
        sol = scipy.integrate.solve_ivp(
            self._fun(pbar,t),
            t_span=(t0,t),
            t_eval=np.arange(0,t,dt),
            y0=r0,
            method="RK23")
        pbar.finish()
        self.t = sol.t
        state = sol.y

        # Save network state
        self.exc.state = np.hstack([self.exc.state, state[:self.exc.size,:]])
        if self.inh: 
            self.inh.state = np.hstack([self.inh.state, state[-self.inh.size:,:]])

    def simulate_euler(self, t, r0, t0=0, dt=1e-3, r_ext=lambda t: 0, save_field=True):
        """
        Euler-Maryama scheme
        """
        logger.info("Integrating network dynamics")
        self.r_ext = r_ext
        state = np.zeros((self.exc.size, int((t-t0)/dt)))
        field = np.zeros_like(state)
        state[:,0] = r0
        if self.disable_pbar:
            pbar = progressbar.NullBar()
            fun = self._fun(pbar,t)
        else:
            fun = self._fun(tqdm(total=int(t/dt)),t)
        for i, t in enumerate(np.arange(t0,t-dt,dt)[:-1]):
            r = state[:,i]
            dr, field_i = fun(t, r, return_field=save_field)
            if self.xi:
                sigma = np.sqrt(field.var()) / self.tau
                dr += self.xi.value(dt,self.tau,self.exc.size) * sigma
            state[:,i+1] = state[:,i] + dt * dr
            if save_field:
                field[:,i] = field_i
        self.exc.state = np.hstack([self.exc.state, state[:self.exc.size,:]])
        if save_field:
            self.exc.field = np.hstack([self.exc.field, field[:self.exc.size,:]])
        if self.inh: 
            self.inh.state = np.hstack([self.inh.state, state[-self.inh.size:,:]])
            if save_field:
                self.inh.field = np.hstack([self.inh.field, field[:self.inh.size,:]])

    def add_noise(self, xi, pop):
        self.xi = xi

    def _fun1(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 1
            """
            # $ \frac{dx}{dt} = -x + \phi( \sum_{j} J_{ij} x_j + I_0 ) $

            pbar.update(t%t_max)

            if self.inh:
                raise NotImplemented
            else:
                phi_r = self.exc.phi
            r_ext = self.r_ext
            r_sum = phi_r(self.W.dot(r) + r_ext(t))
            dr = (-r + r_sum) / self.tau
            if return_field:
                return dr, r_sum
            else:
                return dr
        return f

    def _fun2(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 2
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_0 $

            pbar.update(t)

            if self.inh:
                phi_r = np.zeros_like(r)
                phi_r[:self.exc.size] = self.exc.phi(r[:self.exc.size])
                phi_r[-self.inh.size:] = self.inh.phi(r[-self.inh.size:])
            else:
                phi_r = self.exc.phi(r)
            r_ext = self.r_ext
            r_sum = self.W.dot(phi_r)
            dr = (-r + r_sum + r_ext(t)) / self.tau
            if return_field:
                return dr, r_sum
            else:
                return dr
        return f

    def _fun3(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 3: instantaneous inhibition
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_inh + I_0 $

            pbar.update(t%t_max)

            if self.inh is None:
                return NotImplementedError

            r_ext = self.r_ext
            phi_r_exc = self.exc.phi(r[:self.exc.size])
            r_sum_exc = self.W_EE.dot(phi_r_exc)
            r_sum_inh = 20*self.W_EI.dot(self.W_IE.dot(phi_r_exc)) #FIXME: 20 is g_phi^I. Do not hardcode
            dr = np.zeros(self.size)
            dr[:self.exc.size] = (-r[self.exc.size] + r_sum_exc + r_sum_inh + r_ext(t)) / self.tau[:self.exc.size]

            return dr
        return f

    def overlap_with(self, vec, pop, spikes=False):
        """
        Compute the overlap of network activity with a given input vector
        """
        return pop.state.T.dot(vec) / self.exc.size

    def clear_state(self):
        self.exc.state = np.array([], ndmin=2).reshape(self.exc.size,0)
        self.exc.field = np.array([], ndmin=2).reshape(self.exc.size,0)
        if self.inh:
            self.inh.state = np.array([], ndmin=2).reshape(self.inh.size,0)
            self.inh.field = np.array([], ndmin=2).reshape(self.inh.size,0)


class PoissonNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity(),
            r_max=100.):
        super(PoissonNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)
        self.r_max=r_max

    def simulate(self, t, r0, t0=0, dt=1e-3, exact=False):
        tau = self.tau
        neighbors = self.c_EE.ij
        W = self.W
        r = r0*1
        phi = self.exc.phi

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size))
        spikes = np.zeros_like(state)

        #@njit -- no speedup with compilation
        def func(state, spikes, W, r, r_max, t_size, tau, dt, phi, neighbors):
            for n in range(t_size-1):
                rv = np.random.random(size=r.size)
                spks = np.nonzero(rv < phi(r)*dt*r_max)[0]

                # Propagate spike to neighbors
                for j in spks:
                    idxs = neighbors[j]
                    r[idxs] += 1./r_max*W[idxs,j].flatten()/tau

                # Decay intensity function for all units
                r = r * np.exp(-dt/tau)

                # Record state and spikes
                state[:,n+1] = r
                spikes[spks,n] = 1

        # Much faster with dense matrix
        func(state, spikes, np.asarray(W.todense()), r, self.r_max,
             t_size, tau, dt, phi, neighbors)

        # Save network state
        self.exc.state = state[:self.exc.size,:]
        self.exc.spikes = spikes[:self.exc.size,:]
        if self.inh: 
            self.inh.state = state[self.exc.size:,:]
            self.inh.spikes = spikes[self.exc.size:,:]

    def rates(self, pop):
	    r = spike_to_rate(pop.spikes)
	    return r

    def overlap_with(self, vec, pop, spikes=False):
        "Compute the overlap of network activity with a given input vector"
        if spikes:
            r = self.rates(pop)
        else:
            r = pop.state
        return r.T.dot(vec) / self.exc.size


class SpikingNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(),
            c_EI=Connectivity(), c_II=Connectivity(),
            r_max=100.):
        super(SpikingNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)

    def simulate(self, t, s0, v0, t0=0, dt=1e-3, tau_rp=1e-3, exact=False):
        neighbors = self.c_EE.ij
        W_EE = np.asarray(self.W_EE.todense())
        s = s0
        v = v0
        thresh = self.exc.thresh
        reset = self.exc.reset
        tau_mem = self.exc.tau_mem
        tau_syn = self.exc.tau_syn

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size, 2))
        spikes = np.zeros_like(state[:,:,0])

        for n in range(t_size-1):
            spks = np.nonzero(v > thresh)[0]

            # Decay intensity function for all units
            v = v * np.exp(-dt/tau_mem) + s * (1 - np.exp(-dt/tau_mem))
            s = s * np.exp(-dt/tau_syn)
            v[spks] = reset

            # Propagate spike to neighbors
            for j in spks:
                idxs = neighbors[j]
                s[idxs] += W_EE[idxs,j].flatten()/tau_syn

            if n > 0:
                idxs = spikes[:,n-1].nonzero()[0]
                v[idxs] = reset

            # Record state and spikes
            state[:,n+1,0] = s
            state[:,n+1,1] = v
            spikes[spks,n] = 1

        # Save network state
        self.exc.state = state[:self.exc.size,:,:]
        self.exc.spikes = spikes[:self.exc.size,:]

    # TODO: Note that refractory period is hardcoded to 1ms right now
    def simulate_two_pop(self, 
            t,
            s0_exc,
            v0_exc,
            s0_inh,
            v0_inh,
            t0=0,
            dt=1e-3,
            tau_rp=1e-3,
            sigma_lif_E=0,
            sigma_lif_I=0,
            v_exc_lower_bound=None,
            white_noise_seed=100):
        """
        Simulate two population LIF spiking network

        Inputs:
            t: 
            s0_exc:
            v0_exc:
            s0_exc:
            v0_exc: 
            t0:
            dt:
            tau_rp:
            v_exc_lower_bound:
        """

        neighbors_EE = self.c_EE.ij
        neighbors_EI = self.c_EI.ij
        neighbors_IE = self.c_IE.ij
        neighbors_II = self.c_II.ij

        W_EE = np.asarray(self.W_EE.todense())
        W_EI = np.asarray(self.W_EI.todense())
        W_IE = np.asarray(self.W_IE.todense())
        W_II = np.asarray(self.W_II.todense())

        s_exc = s0_exc
        v_exc = v0_exc
        s_inh = s0_inh
        v_inh = v0_inh

        thresh_exc = self.exc.thresh
        reset_exc = self.exc.reset
        tau_mem_exc = self.exc.tau_mem
        tau_syn_exc = self.exc.tau_syn

        thresh_inh = self.inh.thresh
        reset_inh = self.inh.reset
        tau_mem_inh = self.inh.tau_mem
        tau_syn_inh = self.inh.tau_syn

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size, 2))
        spikes = np.zeros_like(state[:,:,0])

        xi = np.random.RandomState(seed=white_noise_seed)

        for n in trange(t_size-1):

            # Get indices of neurons that crossed spiking threshold in the last timestep
            spks_exc = np.nonzero(v_exc >= thresh_exc)[0]
            spks_inh = np.nonzero(v_inh >= thresh_inh)[0]

            # Evolve membrane voltage in time
            v_exc = v_exc * np.exp(-dt/tau_mem_exc) + \
                    s_exc * (1 - np.exp(-dt/tau_mem_exc))
            v_inh = v_inh * np.exp(-dt/tau_mem_inh) + \
                    s_inh * (1 - np.exp(-dt/tau_mem_inh))

            # Evolve synaptic currents in time
            s_inh = s_inh * np.exp(-dt/tau_syn_inh)
            s_exc = s_exc * np.exp(-dt/tau_syn_exc)

            # Propagate exc spikes to neighbors
            for j in spks_exc:
                idxs_EE = neighbors_EE[j]
                idxs_IE = neighbors_IE[j]
                s_exc[idxs_EE] += W_EE[idxs_EE,j].flatten()/tau_syn_exc
                s_inh[idxs_IE] += W_IE[idxs_IE,j].flatten()/tau_syn_inh
            # Propagate inh spikes to neighbors
            for j in spks_inh:
                idxs_II = neighbors_II[j]
                idxs_EI = neighbors_EI[j]
                s_inh[idxs_II] += W_II[idxs_II,j].flatten()/tau_syn_inh
                s_exc[idxs_EI] += W_EI[idxs_EI,j].flatten()/tau_syn_exc

            # Reset neurons that spiked in last timestep
            v_exc[spks_exc] = reset_exc
            v_inh[spks_inh] = reset_inh

            # Inject white noise into membrane potentials
            if sigma_lif_E > 0:
                v_exc += np.sqrt(dt/tau_mem_exc)*sigma_lif_E*xi.normal(size=self.exc.size)
            if sigma_lif_I > 0:
                v_inh += np.sqrt(dt/tau_mem_inh)*sigma_lif_I*xi.normal(size=self.inh.size)

            # Clamp membrane potentials above threshold to threshold
            v_exc[np.nonzero(v_exc > thresh_exc)[0]] = thresh_exc
            v_inh[np.nonzero(v_inh > thresh_inh)[0]] = thresh_inh

            # Clamp membrane potentials below voltage floor to floor
            if v_exc_lower_bound:
                v_exc[v_exc < v_exc_lower_bound] = v_exc_lower_bound

            # Refractory period: Clamp voltage of neurons that spiked 1 timestep ago to reset/rest
            if n > 0:
                idxs_exc = spikes[:,n-1][:self.exc.size].nonzero()[0]
                idxs_inh = spikes[:,n-1][self.exc.size:].nonzero()[0]
                v_exc[idxs_exc] = reset_exc
                v_inh[idxs_inh] = reset_inh

            # Record state and spikes
            state[:,n+1,0] = np.concatenate([s_exc,s_inh])
            state[:,n+1,1] = np.concatenate([v_exc,v_inh])
            spikes[spks_exc,n] = 1
            spikes[self.exc.size+spks_inh,n] = 1

        # Save network state
        self.exc.state = state[:self.exc.size,:,:]
        self.exc.spikes = spikes[:self.exc.size,:]
        if self.inh: 
            self.inh.state = state[self.exc.size:,:,:]
            self.inh.spikes = spikes[self.exc.size:,:]

    def rates(self, pop):
	    r = spike_to_rate(pop.spikes)
	    return r

    def overlap_with(self, vec, pop, spikes=False):
        "Compute the overlap of network activity with a given input vector"
        if spikes:
            r = self.rates(pop)
        else:
            r = pop.state[:,:,0]
        return r.T.dot(vec) / self.exc.size

