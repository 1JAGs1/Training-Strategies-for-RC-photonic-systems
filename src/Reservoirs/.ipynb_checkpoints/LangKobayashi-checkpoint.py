#from __future__ import annotations
import sys
import matplotlib.pyplot as plt
import numpy as np      


#Setting up ODEs
'''
def expand_virtual_nodes(V, tap_stride, Nd):

    L, N = V.shape
  

    Vs = np.zeros(L * Nd)

    for l in range(L):
        for n in range(N):
            for step in range(n * tap_stride, (n + 1) * tap_stride):
                Vs[step + l * Nd] = V[l, n]

    return Vs
'''
def expand_virtual_nodes(V, tap_stride, Nd):
    L, N = V.shape
   # Nd = N * tap_stride

    # Repeat each column tap_stride times along time axis
    V_expanded = np.repeat(V, tap_stride, axis=1)

    # Flatten into single time series
    Vs = V_expanded.reshape(L * Nd)

    return Vs

def ODEs(E, n, E_delay, v_slot, alpha, kappa, phi, p, eta, D_noise, xi, Tlk):
  
  
    I = abs(E)**2  

   # dEdt = (1.0 + 1j*alpha) * E - I * E + kappa * E_delay * np.exp(1j*phi)
    #dEdt = ((1.0 + 1j*alpha) * E*n ) - (I * E) + (kappa * E_delay * np.exp(1j*phi))
    
    dEdt = ((1.0 + 1j*alpha) * n * E) + (kappa * (np.exp(1j*phi)) * E_delay) + D_noise*xi

   

    dndt = (1/Tlk)*(p + (eta * v_slot) - n -((2*n + 1)*I))

    return dEdt, dndt

'''
### NOISE version
def ODEs(E, n, E_delay, v_slot, alpha, kappa, phi, p, eta, D_noise, xi, Tlk):
  
  
    I = abs(E)**2  

   # dEdt = (1.0 + 1j*alpha) * E - I * E + kappa * E_delay * np.exp(1j*phi)
    #dEdt = ((1.0 + 1j*alpha) * E*n ) - (I * E) + (kappa * E_delay * np.exp(1j*phi))
    
    dEdt = ((1.0 + 1j*alpha) * n * E) + (kappa * (np.exp(1j*phi)) * E_delay) + D_noise*xi

   

    dndt = (1/Tlk)*(p + (eta * v_slot) - n -((2*n + 1)*I))

    return dEdt, dndt
'''

def rk4_step(E, n, E_delay, v_slot, dt, alpha, kappa, phi, p, eta, D_noise, xi, Tlk):
    
 
    # k1
    k1_E, k1_n = ODEs(E, n, E_delay, v_slot, alpha, kappa, phi, p, eta, D_noise, xi, Tlk)

    # k2
    E2 = E + 0.5 * dt * k1_E
    n2 = n + 0.5 * dt * k1_n
    k2_E, k2_n = ODEs(E2, n2, E_delay, v_slot, alpha, kappa, phi, p, eta, D_noise, xi, Tlk)

    # k3
    E3 = E + 0.5 * dt * k2_E
    n3 = n + 0.5 * dt * k2_n
    k3_E, k3_n = ODEs(E3, n3, E_delay, v_slot, alpha, kappa, phi, p, eta, D_noise, xi, Tlk)

    # k4
    E4 = E + dt * k3_E
    n4 = n + dt * k3_n
    k4_E, k4_n = ODEs(E4, n4, E_delay, v_slot, alpha, kappa, phi, p, eta, D_noise, xi, Tlk)


    E_next = E + (dt/6.0) * (k1_E + 2*k2_E + 2*k3_E + k4_E)
    n_next = n + (dt/6.0) * (k1_n + 2*k2_n + 2*k3_n + k4_n)

    return E_next, n_next

'''
#Noise Version
def rk4_step(E, n, E_delay, v_slot, dt, alpha, kappa, phi, p, eta, Tlk, D_noise, rng):

    # k1
    k1_E, k1_n = ODEs(E, n, E_delay, v_slot, alpha, kappa, phi, p, eta, Tlk)

    # k2
    E2 = E + 0.5 * dt * k1_E
    n2 = n + 0.5 * dt * k1_n
    k2_E, k2_n = ODEs(E2, n2, E_delay, v_slot, alpha, kappa, phi, p, eta, Tlk)

    # k3
    E3 = E + 0.5 * dt * k2_E
    n3 = n + 0.5 * dt * k2_n
    k3_E, k3_n = ODEs(E3, n3, E_delay, v_slot, alpha, kappa, phi, p, eta, Tlk)

    # k4
    E4 = E + dt * k3_E
    n4 = n + dt * k3_n
    k4_E, k4_n = ODEs(E4, n4, E_delay, v_slot, alpha, kappa, phi, p, eta, Tlk)

    # deterministic RK4 update
    E_next = E + (dt/6.0) * (k1_E + 2*k2_E + 2*k3_E + k4_E)
    n_next = n + (dt/6.0) * (k1_n + 2*k2_n + 2*k3_n + k4_n)

    # add one complex Gaussian noise kick to E
    noise = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
    E_next = E_next + D_noise * np.sqrt(dt) * noise

    return E_next, n_next
'''


#running ODEs
def simulate_lk(Vs, dt, Nd_delay, *, alpha, kappa, phi, p, eta, D_noise, xi, Tlk, E0=1e-3+0j, n0=0.0):
    Vs = np.asarray(Vs, dtype=float)
    T_steps = Vs.size

    E_hist = np.zeros(T_steps, dtype=np.complex128)
    n_hist = np.zeros(T_steps, dtype=float)

    # delay buffer holds E(t - tau)
    delay_buf = np.full(Nd_delay, E0, dtype=np.complex128)
    di = 0 #Index pointer for delay buffer

    E, n = E0, n0
    for t in range(T_steps):
        v_slot = float(Vs[t])
        E_delay = delay_buf[di]

        E, n = rk4_step(E, n, E_delay, v_slot, dt, alpha, kappa, phi, p, eta, D_noise, xi, Tlk)

        E_hist[t] = E
        n_hist[t] = n

        delay_buf[di] = E
        di = (di + 1) % Nd_delay

    return E_hist, n_hist


def simulate_lk_mini(Vs, dt, Nd_delay, *, alpha, kappa, phi, p, eta, D_noise, xi, Tlk, N_of_thetas, tap_stride, E0=1e-3+0j, n0=0.0):
    Vs = np.asarray(Vs, dtype=float)
    T_steps = N_of_thetas * tap_stride

    if len(Vs) < T_steps:
        raise ValueError("Vs is shorter than N_of_thetas * tap_stride")

    E_hist = np.zeros(T_steps, dtype=np.complex128)
    n_hist = np.zeros(T_steps, dtype=float)

    delay_buf = np.full(Nd_delay, E0, dtype=np.complex128)
    di = 0

    E, n = E0, n0

    for t in range(T_steps):
        v_slot = float(Vs[t])
        E_delay = delay_buf[di]

        E, n = rk4_step(
            E, n, E_delay, v_slot, dt,
            alpha, kappa, phi, p, eta, D_noise, xi, Tlk
        )

        E_hist[t] = E
        n_hist[t] = n

        delay_buf[di] = E
        di = (di + 1) % Nd_delay

    return E_hist, n_hist








#extraction

def extract_R_from_E(E_hist, L, N, tap_stride, Nd_loop):
    R = np.zeros((L, N), dtype=float)
    for l in range(L):
        base = l * Nd_loop
        for n in range(N):
            t_idx = base + (n + 1) * tap_stride - 1  # end of slot
            R[l, n] = np.abs(E_hist[t_idx])**2
    return R


def intensities_after_theta_times(E_hist, N_of_thetas, tap_stride):
    R_array = np.zeros(N_of_thetas, dtype=float)
    
    for k in range(N_of_thetas):
        t = k * tap_stride
        R_array[k] = np.abs(E_hist[t])**2

    return R_array

def charge_at_theta_intervals(n_hist, N_of_thetas, tap_stride):
    
    R_array = np.zeros(N_of_thetas, dtype=float)
    
    for k in range(N_of_thetas):
        t = k * tap_stride
        R_array[k] = n_hist[t]

    return R_array

