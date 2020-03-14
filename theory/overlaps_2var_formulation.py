import sys
import argparse
import logging
import os
import pdb
import signal
import pathos
import numpy as np
from numpy import exp, sqrt, pi
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import erf
from scipy.integrate import odeint
from scipy.interpolate import interp1d, interp2d

sys.path.insert(0, '.')
from optimized.routines import C0 as C0_integrate
from optimized.routines import C1 as C1_integrate
from optimized.routines import C2 as C2_integrate
from optimized.routines import C3 as C3_integrate

logging.basicConfig(level=logging.INFO)

def overlaps_erf_bilinear(T, P, alpha, A, mu, sigma, tau=1e-2, dt=1e-3, n_workers=20, terminate_on=None):
    """
    Compute the expected overlaps using a 2 variable mean-field formulation.
    The 3 variable formulation is found in Supporting Information, Section 2.1-2.2.

    Assumes that patterns are i.i.d Gaussians, an erf neural 
    transfer function (see network/transfer_functions.py),
    and a bilinear learning rule.

    Inputs:
        T: simulation time
        P: number of stored patterns 
        alpha: sequence load
        mu: neural transfer function offset
        sigma: neural transfer function inverage gain
    Output:
        m: overlaps 
        M: average squared rate
        C: two-point correlation function
        r: average rate
    """

    PARALLEL = True
    DEBUG = False #True
    
    N_WORKERS = n_workers
    timeout = 1e8
    
    T /= dt
    tau /= dt
    dt /= dt
    N = int(T/dt)
    
    # Transfer function
    phif = lambda x: 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)))
    
    # Quadrature accuracy
    epsabs = 1e-3
    epsrel = 1e-3
    eval_limit = 25
        
    def dmdt(i, t, m, M, m_ext):
        "$-q_l + q_{l-1} * G(\sum{q^2, M}"
        R = np.sqrt(np.sum(m[:-1]**2) + alpha * A * M)
        G = exp(-mu**2 / (2 *
                          (R**2 + sigma**2))) / sqrt(2 * pi * (R**2 + sigma**2))
        J = -np.eye(P)
        diag = np.diagonal(J, offset=-1)
        diag.setflags(write=True)
        diag.fill(G)
        dm = J.dot(m) / tau
        return dm
    
    def dMdt(i, t, m_mat, M_mat, C_mat, m_ext):
        "$-2M + 2 integral_t du exp(u-t) integral_u_t_c phi(b(u,t))phi(b(t,u))$"
        if t == 0:
            m_interp = lambda x: m_mat[:,0]
            M_interp = lambda x: np.array([M_mat[0]])
            C_interp = lambda x,y: C_mat[0,0]
        else:
            t_interp = np.arange(0, t+dt, dt)
            m_interp = interp1d(t_interp, m_mat[:,:t_interp.size], axis=1, fill_value="extrapolate")
            M_interp = interp1d(t_interp, M_mat[:t_interp.size], fill_value="extrapolate")
            C_interp = interp2d(t_interp, t_interp, C_mat[:t_interp.size,:t_interp.size])
        def sigma_t(t):
            return sqrt(sum(m_interp(t)[:-1]**2) + alpha*M_interp(t))
        def rho_t_u(t,u):
            res = (sum(m_interp(t)[:-1]*m_interp(u)[:-1]) + alpha*C_interp(t,u)) / (sigma_t(u)*sigma_t(t))
            #if res > 1.: 
            #    print("WARNING: (M) Correlation value greater than 1:", res[0])
            return res

        res1 = C0_integrate(
                a=sigma_t(t),
                mu=mu,
                sigma=sigma,
                phif=0)
        res1 = C1_integrate(
                a=1,
                b=sigma_t(t),
                mu=mu,
                sigma=sigma,
                phif=0)

        res2 = quad(
            lambda u, t, m, M: np.exp((u - t)/tau) * C3_integrate(
                a=sigma_t(u),
                b=sigma_t(u)*rho_t_u(u,t),
                c=sqrt(np.clip(sigma_t(u)**2 - rho_t_u(u,t)**2 * sigma_t(t)**2, 0, None)),
                mu=mu,
                sigma=sigma,
                phif=0),
            a=0,
            b=t,
            args=(t, m_interp, M_interp),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=eval_limit)[0]
    
        dM = (-2*M_mat[i] + 2*np.exp(-t/tau)*res1 + 1./tau*2*res2) / tau
        return dM
    
    
    def dCdt(i, j, t, u, m_mat, M_mat, C_mat, m_ext):
        "$-C(t,u) + integral_u dv exp(v-u) integral_u_t_c phi(b(v,t))phi(b(t,v))$"
    
        if t == 0:
            m_interp = lambda x: m_mat[:,0]
            M_interp = lambda x: np.array([M_mat[0]])
            C_interp = lambda x,y: C_mat[0,j]
        else:
            t_interp = np.arange(0, t+dt, dt)
            m_interp = interp1d(t_interp, m_mat[:,:t_interp.size], axis=1, fill_value="extrapolate")
            M_interp = interp1d(t_interp, M_mat[:t_interp.size], fill_value="extrapolate")
            C_interp = interp2d(t_interp, t_interp, C_mat[:t_interp.size,:t_interp.size])
        def sigma_t(t):
            return sqrt(sum(m_interp(t)[:-1]**2) + alpha*M_interp(t))
        def rho_t_u(t,u):
            res = (np.sum(m_interp(t)[:-1]*m_interp(u)[:-1]) + alpha*C_interp(t,u)) / (sigma_t(u)*sigma_t(t))
            #if res > 1.:
            #    print("WARNING: (C) Correlation value greater than 1:", res[0])
            return res

        res1 = C0_integrate(
                a=sigma_t(t),
                mu=mu,
                sigma=sigma,
                phif=0)

        res2 = quad(
            lambda v, t, u, m, M: np.exp((v - u)/tau) * C3_integrate(
                a=sigma_t(v),
                b=sigma_t(v)*rho_t_u(v,t),
                c=sqrt(np.clip(sigma_t(v)**2 - rho_t_u(v,t)**2 * sigma_t(t)**2, 0, None)),
                mu=mu,
                sigma=sigma,
                phif=0),
            a=0,
            b=u,
            args=(t, u, m_interp, M_interp),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=eval_limit)[0]
    
        dC = (-C_mat[i,j] + np.exp(-u/tau)*res1 + 1./tau*res2) / tau
        return dC,i,j

    def drbardt(i, t, m_mat, M_mat, rbar_mat, m_ext):
        ""

        if t == 0:
            m_interp = lambda x: m_mat[:,0]
            M_interp = lambda x: np.array([M_mat[0]])
        else:
            t_interp = np.arange(0, t+dt, dt)
            m_interp = interp1d(t_interp, m_mat[:,:t_interp.size], axis=1, fill_value="extrapolate")
            M_interp = interp1d(t_interp, M_mat[:t_interp.size], fill_value="extrapolate")

        def sigma_t(t):
            return sqrt(sum(m_interp(t)[:-1]**2) + alpha*M_interp(t))

        res1 = C0_integrate(
                a=sigma_t(t),
                mu=mu,
                sigma=sigma,
                phif=0)

        res2 = quad(
            lambda u: np.exp(u/tau) * C0_integrate(
                a=sigma_t(u),
                mu=mu,
                sigma=sigma,
                phif=0),
            a=0,
            b=t,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=eval_limit)

        drbar = ( res1 + np.exp(-t/tau) * ( -rbar_mat[0] - 1./tau*res2[0] ) ) / tau
        return drbar


    t = 0
    m = np.zeros((P,N))
    M = np.zeros(N)
    C = np.zeros((N,N))
    rbar = np.zeros(N)
    
    m[0,0] = norm.expect(lambda v: v * phif(v), scale=1)
    M[0] = norm.expect(lambda v: phif(v)**2, scale=1)
    C[0,0] = M[0]
    rbar[0] = norm.expect(lambda v: phif(v), scale=1)

    m_ext = np.zeros((int(T/dt),P))

    if PARALLEL:
        print("Initializing workers")
        pool = pathos.multiprocessing.Pool(processes=N_WORKERS)
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGINT, original_sigint_handler)
    
    try:
        # Integrate over time t=i*dt
        for i in range(N-1): 
            
            print("t:", t, "m1:", m[1,i], "M:", M[i])

            ## Start integration jobs
        
            # Integrate m, M, and r_bar
            if PARALLEL:
                dm_res = pool.apply_async(dmdt, (i,t,m[:,i],M[i],m_ext[i]))
                dM_res = pool.apply_async(dMdt, (i,t,m,M,C,m_ext[i]))
                drbar_res = pool.apply_async(drbardt, (i,t,m,M,rbar,m_ext[i]))
            else:
                dm_res = apply(dmdt, (i,t,m[:,i],M[i],m_ext[i,:]))
                dM_res = apply(dMdt, (i,t,m,M,C,m_ext[i]))
                drbar_res = apply(drbardt, (i,t,m,M,m_ext[i]))
        
            # Integrate C starting from u=j*dt
            dC_res = []
            for j in range(i+1):
                u = j*dt
                if PARALLEL: res = pool.apply_async(dCdt, (i,j,t,u,m,M,C,m_ext[i]))
                else:        res = apply(dCdt, (i,j,t,u,m,M,C,m_ext[i]))
                dC_res.append(res)

            ## Collect job results
        
            # Collect results for m
            if PARALLEL: dm = dm_res.get(timeout)
            else:        dm = dm_res
            if DEBUG: print("\tdm: t=%0.1f, u=%0.1f, val=%0.3f" % (i*dt,i*dt,dm[0]))
            m[:,i+1] = m[:,i] + dt * dm
    
            # Collect results for M
            if PARALLEL: dM = dM_res.get(timeout)
            else:        dM = dM_res
            if DEBUG: print("\tdM: t=%0.1f, u=%0.1f, val=%0.3f" % (i*dt,i*dt,dM))
            M[i+1] = M[i] + dt * dM

            # Collect results for r_bar
            if PARALLEL: drbar = drbar_res.get(timeout)
            else:        drbar = drbar_res
            if DEBUG: print("\tdrbar: t=%0.1f, u=%0.1f, val=%0.3f" % (i*dt,i*dt,drbar))
            rbar[i+1] = rbar[i] + dt * drbar
    
            # Collect results for C
            for res in dC_res:
                if PARALLEL: dC,i,j = res.get(timeout)
                else:        dC,i,j = res
                if DEBUG: print("\tdC: t=%0.1f, u=%0.1f, val=%0.3f" % (i*dt,j*dt,dC))
                C[i+1,j] = C[i,j] + dt * dC
    
            # Use diagonal symmetry of C
            C[i+1,i+1] = M[i+1]
            for j in range(i+1):
                C[j,i+1] = C[i+1,j]
        
            t += dt

            # Early termination condition
            if terminate_on:
                if terminate_on(t, m, M, C, rbar):
                    break

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating")
        if PARALLEL: pool.terminate()
    else:
        print("Normal termination")
        if PARALLEL: pool.close()

    if PARALLEL: pool.join()

    return m, M, C, rbar
