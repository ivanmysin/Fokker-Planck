
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def H_function(V, dVdt, tau_m, Vt, sigma):
    T = (Vt - V) / sigma / np.sqrt(2)

    A = np.exp(0.0061 - 1.12 * T - 0.257 * T ** 2 - 0.072 * T ** 3 - 0.0117 * T ** 4)
    dT_dt = -1.0 / sigma / np.sqrt(2) * dVdt
    dT_dt[dT_dt > 0] = 0
    F_T = np.sqrt(2 / np.pi) * np.exp(-T ** 2) / (1.000000001 + erf(T))
    B = -np.sqrt(2) * dT_dt * F_T * tau_m

    H = (A + B) / tau_m
    return H

V = -60
dVdt = np.array([0.1])
tau_m = 10
Vt = -58
sigma = 0.2


H = H_function(V, dVdt, tau_m, Vt, sigma)


print (H)