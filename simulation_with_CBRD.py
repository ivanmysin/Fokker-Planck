
import numpy as np
import matplotlib as plt
from scipy.special import erf

class Neuron:

    def __init__(self, params):
        self.Vreset = params["Vreset"]
        self.Vt = params["Vt"]
        self.gl = params["gl"]
        self.El = params["El"]
        self.C = params["C"]
        self.V = self.Vreset
        self.Iext = params["Iext"]


    def update(self, dt):

        dVdt = ( self.gl*(self.El - self.V) + self.Iext ) / self.C

        self.V += dt * dVdt

        return self.V, dVdt

    def reset(self):

        self.V = self.Vreset


class State:

    def __init__(self, neuron, P, Vt, sigma):
        self.P = P
        self.neuron = neuron
        self.Vt = Vt
        self.sigma = sigma

    def H_function(self, dt, V, dVdt):
        tau_m = self.neuron.gl / self.neuron.C
        k = tau_m / dt
        g_tot = self.neuron.gl  # (Cm = 1) !!!!

        T = np.sqrt(0.5 * (1 + k)) * g_tot * (self.Vt - V) / self.sigma

        A_inf = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        A = A_inf * (1 - (1 + k) ** (-0.71 + 0.0825 * (T + 3)))

        dT_dt = -g_tot / self.sigma * np.sqrt(0.5 + 0.5 * k) * dVdt

        dT_dt[dT_dt < 0] = 0

        F_T = np.sqrt(2 / np.pi) * np.exp(-T**2) / (1 + erf(T))

        B = -np.sqrt(2) * tau_m * dT_dt * F_T

        H = A + B

        return H

    def update(self, dt):

        V, dVdt = self.neuron.update(dt)
        H = self.H_function(dt, V, dVdt)
        ph = self.P * H

        self.P -= ph

        return ph


class CBRD:
    def __init__(self, dts, Nts, neuron_params, sigma):
        self.states = []
        for s in range(Nts):
            neuron = Neuron(neuron_params)
            s = State(neuron, 0, neuron_params["Vt"], sigma)
            self.states.append(s)
        self.neuron_params = neuron_params
        self.Vt = self.neuron_params["Vt"]

    def run(self, dt, duration):

        t = 0
        while (t < duration):
            firings = 0
            for s in self.states:
                firings += s.update(dt)

            spre = self.states.pop(-2)

            s.P += spre.P

            self.states.insert(0, State(Neuron(self.neuron_params), firings, self.Vt ) )

            t += dt



###########################################################################








