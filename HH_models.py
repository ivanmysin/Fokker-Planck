# -*- coding: utf-8 -*-
"""
CA1 neuron from
    A Two Compartment Model of a CA1 Pyramidal Neuron
        Katie A. Ferguson∗†and Sue Ann Campbell
        (2009)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import parzen, argrelmax, argrelmin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import erf


exp = np.exp

class CBRD:

    def __init__(self, Nro, dts):
        self.Nro = Nro
        self.dts = dts

        self.t_states = np.linspace(0, self.Nro*self.dts, self.Nro)
        self.ro = np.zeros_like(self.t_states)

        self.ro[-1] = 1 / self.dts
        self.ro_H_integral = 0
        self.ts = 0
        self.max_roH_idx = 0


    def H_function(self, V, dVdt, tau_m, Vt, sigma):

        T = (Vt - V) / sigma / np.sqrt(2)
        A = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        dT_dt = -1.0 / sigma / np.sqrt(2) * dVdt
        dT_dt[dT_dt > 0] = 0
        F_T = np.sqrt(2 / np.pi) * np.exp(-T**2) / (1.000000001 + erf(T))
        B = -np.sqrt(2) * dT_dt * F_T * tau_m
        H = (A + B) / tau_m
        return H


    def update_ro(self, dt, dVdt, tau_m):
        shift = False
        if self.ts >= self.dts:
            self.ro[-1] += self.ro[-2]
            self.ro[:-1] = np.roll(self.ro[:-1], 1)
            self.ro[0] = self.ro_H_integral
            self.ro_H_integral = 0
            self.ts = 0
            shift = True
            # self.ro /= np.sum(self.ro) * self.dts
            # print (np.sum(self.ro) * self.dts)
        H = self.H_function(self.V, dVdt, tau_m, self.Vt, self.sigma)
        H[:self.ref_idx] = 0
        dro = self.ro * (1 - np.exp(-H * dt))  # dt * self.ro * H  #
        self.ro -= dro
        self.ro[self.ro < 0] = 0
        self.max_roH_idx = np.argmax(dro)
        sum_dro = np.sum(dro)
        self.ro_H_integral += sum_dro

        if self.saveCV:
            roH = self.ro * H * self.dts
            isi_mean_of2 = np.sum(self.t_states ** 2 * roH) * np.sum(roH)
            isi_mean = (np.sum(self.t_states * roH))**2  # / sum_roH])**2

            if isi_mean < isi_mean_of2 and isi_mean_of2 > 0:
                CV = np.sqrt(isi_mean_of2 / isi_mean - 1)
            else:
                CV = 0
            # print (CV)
            self.CVhist.append(CV)


        self.ts += dt

        return shift

class FS_neuron(CBRD):
    def __init__(self, params):

        if params["is_use_CBRD"]:
            self.N = params["Nro"]
        else:
            self.N = params["N"]

        self.V = np.zeros(self.N, dtype=float) + params["El"]
        self.Iextmean = params["Iextmean"]
        self.Iextvarience = params["Iextvarience"]
        self.ENa = params["ENa"]
        self.EK = params["EK"]
        self.El = params["El"]
        self.gbarNa = params["gbarNa"]
        self.gbarK = params["gbarK"]
        self.gl = params["gl"]
        self.fi = params["fi"]
        self.Capacity = params["Capacity"]



        self.Vhist = []
        self.mhist = []
        self.nhist = []
        self.hhist = []


        self.firing = []

        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())

        self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
        self.gK = self.gbarK * self.n * self.n * self.n * self.n

        self.Iext = 0
        self.Isyn = 0
        self.countSp = True
        self.th = -20

    def alpha_m(self):
        # double alpha;
        x = -0.1 * (self.V + 33)
        x[x == 0] = 0.000000001

        alpha = x / (np.exp(x) - 1)
        return alpha

    #########
    def beta_m(self):
        beta = 4 * np.exp(- (self.V + 58) / 18)
        return beta

    ########
    def alpha_h(self):

        alpha = self.fi * 0.07 * np.exp(-(self.V + 51) / 10)
        return alpha

    ########
    def beta_h(self):

        beta = self.fi / (np.exp(-0.1 * (self.V + 21)) + 1)
        return beta

    ########
    def alpha_n(self):

        x = -0.1 * (self.V + 38)
        x[x == 0] = 0.00000000001
        alpha = self.fi * 0.1 * x / (np.exp(x) - 1)
        return alpha

    #######np.

    def beta_n(self):

        return (self.fi * 0.125 * np.exp(-(self.V + 48) / 80))

    #######
    def h_integrate(self, dt):

        h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * np.exp(-dt / tau_h)

    #######

    def n_integrate(self, dt):

        n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n())
        tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * np.exp(-dt / tau_n)

    #######
    def update(self, dt, duraction=None):

        if (duraction is None):
            duraction = dt

        t = 0
        i = 0
        while (t < duraction):
            # self.Vhist.append(self.V)
            # self.mhist.append(self.m)
            # self.nhist.append(self.n)
            # self.hhist.append(self.h)
            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience)

            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V) + self.gl * (
            self.El - self.V) - self.Isyn + self.Iext)

            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)

            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
            self.Isyn = 0
            i += 1
            t += dt


            ########

    def checkFired(self, t_):

        if (self.V >= self.th and self.countSp):
            self.firing.append(t_)
            self.countSp = False

        if (self.V < self.th):
            self.countSp = True
            ########

    def getV(self):
        return self.V

    def getVhist(self):
        return self.Vhist

    def setIsyn(self, Isyn):
        self.Isyn += Isyn

    def setIext(self, Iext):
        self.Iext = Iext

    def getFiring(self):
        return self.firing

##############################################


class FS_neuron_Th(FS_neuron):

    def __init__(self, params):

        super(FS_neuron_Th, self).__init__(params)

        self.Vt = params["Vt"]
        self.n_reset = params["n_reset"]
        self.V_reset = params["V_reset"]


        self.is_use_CBRD = params["is_use_CBRD"]
        self.refactory = params["refactory"]
        self.sigma = self.Iextvarience / np.sqrt(2)

        self.firing = [0]
        self.times = [0]

        if self.is_use_CBRD:
            CBRD.__init__(self, params["Nro"], params["dts"])
            self.ref_idx = int (self.refactory / self.dts )
            self.Iext = self.Iextmean
        else:
            self.ts = np.zeros_like(self.V) + 200

    def default(self):

        self.V = self.El
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())

        self.Vhist = []
        self.nhist = []

    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt

        t = 0

        while (t < duration):


            if not self.is_use_CBRD:
                self.Vhist.append(self.V)
                self.nhist.append(self.n)

            dVdt = (self.gK * (self.EK - self.V) + self.gl * (self.El - self.V) - self.Isyn + self.Iext ) / self.Capacity
            self.V = self.V + dt * dVdt


            if (self.is_use_CBRD):
                self.tau_m = self.Capacity / (self.gK + self.gl)
                shift = self.update_ro(dt, dVdt, self.tau_m)

                if shift:
                    self.V[:-1] = np.roll(self.V[:-1], 1)
                    self.n[:-1] = np.roll(self.n[:-1], 1)

                    self.V[0] = self.V_reset
                    self.n[0] = self.n_reset

            else:
                spiking = np.logical_and( (self.V >= self.Vt), (self.ts > self.refactory) )
                self.V[spiking] = self.V_reset
                self.n[spiking] = self.n_reset
                self.ts += dt
                self.ts[spiking] = 0


            self.n = self.n_integrate(dt)
            self.gK = self.gbarK * self.n * self.n * self.n * self.n

            if not self.is_use_CBRD:
                self.Iext = np.random.normal(self.Iextmean, self.Iextvarience, self.V.size)

            self.Isyn = 0
            t += dt

            self.times.append(self.times[-1] + dt)
            if self.is_use_CBRD:
                self.firing.append(1000 * self.ro[0])
            else:
                self.firing.append(1000 * np.mean(spiking) / dt)


        if self.is_use_CBRD:
            return self.t_states, self.ro, self.times, self.firing, self.t_states, self.V
        else:
            return [], [], self.times, self.firing, [], []


    def reset(self):
        self.V = self.V_reset
        self.n = self.n_reset
        self.ts = 0

class ClusterNeuron(FS_neuron):
    def __init__(self, params):

        super(ClusterNeuron, self).__init__(params)

        self.V -= 10
        self.Eh = params["Eh"]
        self.gbarKS = params["gbarKS"]
        self.gbarH = params["gbarH"]
        self.H = 1 / (1 + exp((self.V + 80) / 10))
        self.p = 1 / (1 + exp(-(self.V + 34) / 6.5))
        self.q = 1 / (1 + exp((self.V + 65) / 6.6))
        self.gKS = self.gbarKS * self.p * self.q
        self.gH = self.gbarH * self.H

        self.mhist = []
        self.hhist = []
        self.phist = []
        self.qhist = []
        self.Hhist = []

    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt
        t = 0
        while (t < duration):

            self.Vhist.append(self.V)
            self.nhist.append(self.n)
            self.phist.append(self.p)
            self.qhist.append(self.q)
            self.Hhist.append(self.H)

            self.mhist.append(self.m)
            self.hhist.append(self.h)




            dVdt = self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V)
            dVdt += self.gKS * (self.EK - self.V) + self.gH * (self.Eh - self.V)
            dVdt += self.gl * (self.El - self.V) - self.Isyn + self.Iext
            dVdt /= self.Capacity

            self.V = self.V + dt * dVdt


            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.H = self.H_integrate(dt)
            self.p = self.p_integrate(dt)
            self.q = self.q_integrate(dt)

            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
            self.gH = self.gbarH * self.H
            self.gKS = self.gbarKS * self.p * self.q

            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience)
            self.Isyn = 0
            t += dt


    def H_integrate(self, dt):
        H_0 = 1 / (1 + exp((self.V + 80) / 10))
        tau_H = (200 / (exp((self.V + 70) / 20) + exp(-(self.V + 70) / 20))) + 5

        return H_0 - (H_0 - self.H) * exp(-dt / tau_H)

    def p_integrate(self, dt):
        p_0 = 1 / (1 + exp(-(self.V + 34) / 6.5))
        tau_p = 6
        return p_0 - (p_0 - self.p) * exp(-dt / tau_p)

    def q_integrate(self, dt):
        q_0 = 1 / (1 + exp((self.V + 65) / 6.6))
        tau_q0 = 100
        tau_q = tau_q0 * (1 + (1 / (1 + exp(-(self.V + 50) / 6.8))))
        return q_0 - (q_0 - self.q) * exp(-dt / tau_q)


class ClusterNeuron_Th(ClusterNeuron):

    def __init__(self, params):

        super(ClusterNeuron_Th, self).__init__(params)

        self.Vt = params["Vt"]
        self.n_reset = params["n_reset"]
        self.H_reset = params["H_reset"]
        self.p_reset = params["p_reset"]
        self.q_reset = params["q_reset"]
        self.V_reset = params["V_reset"]

        self.is_use_CBRD = params["is_use_CBRD"]
        self.refactory = params["refactory"]
        self.sigma = self.Iextvarience / np.sqrt(2)

        self.firing = [0]
        self.times = [0]

        if self.is_use_CBRD:
            CBRD.__init__(self, params["Nro"], params["dts"])
            self.ref_idx = int(self.refactory / self.dts)
            self.Iext = self.Iextmean
        else:
            self.ts = np.zeros_like(self.V) + 200



    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt

        t = 0
        while (t < duration):

            if not self.is_use_CBRD:
                self.Vhist.append(self.V)
                self.nhist.append(self.n)
                self.qhist.append(self.q)


            dVdt = self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V)
            dVdt += self.gKS * (self.EK - self.V) + self.gH * (self.Eh - self.V)
            dVdt += self.gl * (self.El - self.V) - self.Isyn + self.Iext
            dVdt /= self.Capacity

            self.V = self.V + dt * dVdt

            if (self.is_use_CBRD):
                self.tau_m = self.Capacity / (self.gK + self.gl + self.gKS + self.gH)
                shift = self.update_ro(dt, dVdt, self.tau_m)

                if shift:
                    self.V[:-1] = np.roll(self.V[:-1], 1)
                    self.n[:-1] = np.roll(self.n[:-1], 1)
                    self.H[:-1] = np.roll(self.H[:-1], 1)
                    self.p[:-1] = np.roll(self.p[:-1], 1)
                    self.q[:-1] = np.roll(self.q[:-1], 1)

                    self.m[:-1] = np.roll(self.m[:-1], 1)
                    self.h[:-1] = np.roll(self.h[:-1], 1)


                    self.V[0] = self.V_reset
                    self.n[0] = self.n_reset
                    self.H[0] += self.H_reset
                    self.p[0] = self.p_reset
                    self.q[0] += self.q_reset

                    self.m[0] = 0
                    self.h[0] = 0.9




            else:
                spiking = np.logical_and( (self.V >= self.Vt), (self.ts > self.refactory) )
                self.V[spiking] = self.V_reset
                self.n[spiking] = self.n_reset
                self.H[spiking] += self.H_reset
                self.p[spiking] = self.p_reset
                self.q[spiking] += self.q_reset

                self.m[spiking] = 0
                self.h[spiking] = 0.9


                self.ts += dt
                self.ts[spiking] = 0


            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.H = self.H_integrate(dt)
            self.p = self.p_integrate(dt)
            self.q = self.q_integrate(dt)

            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
            self.gH = self.gbarH * self.H
            self.gKS = self.gbarKS * self.p * self.q

            if not self.is_use_CBRD:
                self.Iext = np.random.normal(self.Iextmean, self.Iextvarience, self.V.size)

            self.Isyn = 0
            t += dt

            self.times.append(self.times[-1] + dt)
            if self.is_use_CBRD:
                self.firing.append(1000 * self.ro[0])
            else:
                self.firing.append(1000 * np.mean(spiking) / dt)

        if self.is_use_CBRD:
            return self.t_states, self.ro, self.times, self.firing, self.t_states, self.V
        else:
            return [], [], self.times, self.firing, [], []

##################################################################
class Channel:
    def __init__(self, gmax, E, V, x=None, y=None, x_reset=None, y_reset=None):
        self.gmax = gmax
        self.E = E
        self.g = 0
        if not x is None:
            self.x = self.get_x_inf(V)
            self.x_reset = x_reset
        else:
            self.x = None

        if not y is None:
            self.y = self.get_y_inf(V)
            self.y_reset = y_reset
        else:
            self.y = None

    def update(self, dt, V):
        self.g = self.gmax
        if not (self.x is None):
            x_inf = self.get_x_inf(V)
            tau_x = self.get_tau_x(V)
            self.x = x_inf - (x_inf - self.x) * np.exp(-dt / tau_x)
            self.g *= self.x

        if not (self.y is None):
            y_inf = self.get_y_inf(V)
            tau_y = self.get_tau_y(V)
            self.y = y_inf - (y_inf - self.y) * np.exp(-dt / tau_y)
            self.g *= self.y

    def get_I(self, V):
        I = self.g * (V - self.E)
        return I

    def get_g(self):
        return self.g

    def reset(self, spiking):
        if not (self.x is None):
            self.x[spiking] = self.x_reset

        if not (self.y is None):
            self.y[spiking] = self.y_reset

    def get_x_inf(self, V):
        return 0

    def get_y_inf(self, V):
        return 0

    def get_tau_x(self, V):
        return 0

    def get_tau_y(self, V):
        return 0

    def roll(self, max_roH_idx):
        if not (self.x is None):
            tmpx = self.x[max_roH_idx]
            self.x[:-1] = np.roll(self.x[:-1], 1)
            self.x[0] = tmpx

        if not (self.y is None):
            tmpy = self.y[max_roH_idx]
            self.y[:-1] = np.roll(self.y[:-1], 1)
            self.y[0] = tmpy

class KDR_Channel(Channel):

    def get_a(self, V):
        a = 0.17 * np.exp((V + 5) * 0.09)
        return a

    def get_b(self, V):
        b  = 0.17 * np.exp(-(V + 5) * 0.022)
        return b

    def get_tau_x(self, V):
        a = self.get_a(V)
        b = self.get_b(V)

        tau_x = 1 / (a + b) + 0.8

        return tau_x

    def get_x_inf(self, V):
        a = self.get_a(V)
        b = self.get_b(V)
        x_inf = a / (a + b)
        return x_inf

    def get_tau_y(self, V):
        return 300

    def get_y_inf(self, V):
        y_inf = 1 / (1 + np.exp((V + 68) * 0.038) )
        return y_inf

class A_channel(Channel):

    def get_ax(self, V):
        a = 0.08 * np.exp((V+41)*0.089)
        return a

    def get_bx(self, V):
        b = 0.08 * np.exp(-(V+41)*0.016)
        return b

    def get_ay(self, V):
        a = 0.04*np.exp(-(V+49)*0.11)
        return a

    def get_by(self, V):
        b = 0.04
        return b

    def get_tau_x(self, V):
        a = self.get_ax(V)
        b = self.get_bx(V)
        tau_x = 1 / (a + b) + 1
        return tau_x

    def get_x_inf(self, V):
        a = self.get_ax(V)
        b = self.get_bx(V)
        x_inf = a / (a + b)
        return x_inf

    def get_tau_y(self, V):
        a = self.get_ay(V)
        b = self.get_by(V)
        tau_y = 1 / (a + b) + 2
        return tau_y

    def get_y_inf(self, V):
        a = self.get_ay(V)
        b = self.get_by(V)
        y_inf = a  / (a + b)
        return y_inf

    def update(self, dt, V):
        self.g = self.gmax
        if not (self.x is None):
            x_inf = self.get_x_inf(V)
            tau_x = self.get_tau_x(V)
            self.x = x_inf - (x_inf - self.x) * np.exp(-dt / tau_x)
            self.g *= self.x**4

        if not (self.y is None):
            y_inf = self.get_y_inf(V)
            tau_y = self.get_tau_y(V)
            self.y = y_inf - (y_inf - self.y) * np.exp(-dt / tau_y)
            self.g *= self.y**3

class M_channel(Channel):

    def reset(self, spiking):
        self.x[spiking] += self.x_reset * (1 - self.x[spiking])

    def get_a(self, V):
        a = 0.003 * np.exp( (V+45) * 0.135 )
        return a

    def get_b(self, V):
        b = 0.003 * np.exp(-(V+45)*0.09)
        return b

    def get_x_inf(self, V):
        a = self.get_a(V)
        b = self.get_b(V)
        x_inf = a / (a + b)
        return x_inf

    def get_tau_x(self, V):
        a = self.get_a(V)
        b = self.get_b(V)
        tau_x = 1 / (a + b) + 8
        return tau_x

    def update(self, dt, V):
        x_inf = self.get_x_inf(V)
        tau_x = self.get_tau_x(V)
        self.x = x_inf - (x_inf - self.x) * np.exp(-dt / tau_x)
        self.g = self.gmax * self.x**2

class AHP_Channel(Channel):

    def get_tau_x(self, V):
        tau_x = 2000 / (3.3 * np.exp( (V + 35)/20 ) + np.exp(-(V + 35)/20) )
        return tau_x

    def get_x_inf(self, V):
        x_inf = 1 / (1 + np.exp(-(V + 35)/10)) # x_inf = 1 / (1 + np.exp(-(V + 35) / 4))  #
        return x_inf

    def reset(self, spiking):
        self.x[spiking] += self.x_reset * (1 - self.x[spiking])

    def update(self, dt, V):
        x_inf = self.get_x_inf(V)
        tau_x = self.get_tau_x(V)
        self.x = x_inf - (x_inf - self.x) * np.exp(-dt / tau_x)
        self.g = self.gmax * self.x

class HCN_Channel(Channel):

    def get_tau_y(self, V):
        return 180

    def get_y_inf(self, V):
        y_inf = 1.0 / (1 + np.exp((V + 98) * 0.075) )
        return y_inf

    def update(self, dt, V):
        y_inf = self.get_y_inf(V)
        tau_y = self.get_tau_y(V)
        self.y = y_inf - (y_inf - self.y) * np.exp(-dt / tau_y)
        self.g = self.gmax * self.y


class BorgGrahamNeuron(CBRD):
    def __init__(self, params):

        self.is_use_CBRD = params["is_use_CBRD"]

        if self.is_use_CBRD:
            self.V = np.zeros(params["Nro"], dtype=float) - 65
        else:
            self.V = np.zeros(params["N"], dtype=float) - 65

        self.C = params["C"]
        self.V_reset = params["Vreset"]
        self.Vt = params["Vt"]
        self.Iext = params["Iext"]
        self.Iextvarience = params["Iextvarience"]
        self.saveV = params["saveV"]
        self.saveCV = params["saveCV"]
        self.refactory = params["refactory"]

        self.w_in_distr = params["w_in_distr"]
        self.artifitial_generator = False
        self.firing = [0]
        self.times = [0]

        if self.is_use_CBRD:
            CBRD.__init__(self, params["Nro"], params["dts"])
            self.ref_idx = int(self.refactory / self.dts) - 1
        else:
            self.ts = np.zeros_like(self.V) + 200

        leak = Channel(params["leak"]["g"], params["leak"]["E"], self.V)
        dr_current = KDR_Channel(params["dr_current"]["g"], params["dr_current"]["E"], self.V, 1, 1, params["dr_current"]["x_reset"], params["dr_current"]["y_reset"])
        a_current = A_channel(params["a_current"]["g"],params["a_current"]["E"], self.V, 1, 1, params["a_current"]["x_reset"], params["a_current"]["y_reset"])
        m_current = M_channel(params["m_current"]["g"], params["m_current"]["E"], self.V, 1, None, params["m_current"]["x_reset"], params["m_current"]["y_reset"])
        ahp = AHP_Channel(params["ahp"]["g"], params["ahp"]["E"], self.V, 1, None, params["ahp"]["x_reset"], params["ahp"]["y_reset"])
        hcn = HCN_Channel(params["hcn"]["g"], params["hcn"]["E"], self.V, None, 1, params["hcn"]["x_reset"], params["hcn"]["y_reset"])

        self.channels = [leak, dr_current, a_current, m_current, ahp, hcn]

        self.Isyn = 0

        self.tau_m = self.C / params["leak"]["g"]
        self.sigma = self.Iextvarience / params["leak"]["g"] * np.sqrt( 0.5 / self.tau_m)

        if  self.saveV:
            self.Vhist = [self.V[0]]

        if self.saveCV:
            self.CVhist = [0]


    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt

        t = 0
        while(t < duration):

            g_tot = 0
            I = 0
            for ch in self.channels:
                ch.update(dt, self.V)
                I -= ch.get_I(self.V)
                g_tot += ch.get_g()

            # print(self.V)
            I += self.Iext
            I += self.Isyn

            if not self.is_use_CBRD:
                I += np.random.normal(0, self.Iextvarience, self.V.size) / np.sqrt(dt)

            dVdt = I / self.C
            self.V += dt * dVdt

            self.Isyn = 0

            if (self.is_use_CBRD):
                self.tau_m = self.C / g_tot  # 10.4 # 25.4 #
                shift = self.update_ro(dt, dVdt, self.tau_m)
                if shift:
                    self.V[:-1] = np.roll(self.V[:-1], 1)
                    self.V[0] = self.V_reset
                    for ch in self.channels:
                        ch.roll(self.max_roH_idx)
                        ch.reset(0)
            else:
                spiking = np.logical_and( (self.V >= self.Vt), (self.ts > self.refactory) )
                self.ts += dt
                if np.sum(spiking) > 0:
                    self.V[spiking] = self.V_reset
                    self.ts[spiking] = 0
                    for ch in self.channels:
                        ch.reset(spiking)

            if self.saveV:
                if (self.is_use_CBRD):
                    self.Vhist.append( np.sum(self.V * self.ro * self.dts) ) # np.mean(self.V)
                else:
                    self.Vhist.append( self.V[0] )

            t += dt

            self.times.append(self.times[-1] + dt)
            if self.is_use_CBRD:
                self.firing.append(1000 * self.w_in_distr * self.ro[0])
            else:
                self.firing.append(1000 * self.w_in_distr * np.mean(spiking) / dt)


        if self.is_use_CBRD:
            return self.t_states, self.w_in_distr * self.ro, self.times, np.asarray(self.firing) #, self.t_states, self.V
        else:
            return [], [], self.times, np.asarray(self.firing) # , [], []

    def get_flow(self):
        if self.is_use_CBRD:
            return self.w_in_distr  * self.ro[0]
        else:
            return self.w_in_distr * self.firing[-1] * 0.001

    def get_flow_hist(self):
        return self.w_in_distr * np.asarray(self.firing)

    def add_Isyn(self, Isyn):
        self.Isyn += Isyn

    def get_CV(self):
        return self.w_in_distr * np.asarray(self.CVhist)


############
class Optimizer:

    def __init__(self, full_model, optimized_model):

        self.full_model = full_model
        self.optimized_model = optimized_model

        self.dt = 0.1
        self.duration = 1000


    def run_optimization(self):
        self.full_model.update(self.dt, self.duration)
        self.etalonV = np.asarray( self.full_model.getVhist() )


        self.win = parzen(15)


        self.etalonspikes = argrelmax(self.etalonV)[0][1 : 20]

        self.fr_el = 1.0 / np.mean( np.diff(self.etalonspikes) )


        x0 = [0.8, -60, 12.0]

        res = minimize(self.need2optimize, x0, method="Powell" ) # , bounds=[[-61, -50], [-70, 30], [0, 1]]

        print (res.x)

        self.optimized_model.default()

        self.optimized_model.n_reset = res.x[0]
        self.optimized_model.Vt = res.x[1]
        self.optimized_model.V_reset = res.x[2]


        self.optimized_model.update(self.dt, self.duration)

        return res




    def need2optimize(self, x):


        self.optimized_model.default()


        self.optimized_model.n_reset = x[0]
        self.optimized_model.Vt = x[1]
        self.optimized_model.V_reset = x[2]

        self.optimized_model.update(self.dt, self.duration)

        V = np.asarray(self.optimized_model.getVhist())

        time_before_spikes = argrelmax(V)[0][1 : 20]

        if (time_before_spikes.size < 3):
            fr = 0
        else:
            fr = 1.0 / np.mean( np.diff(time_before_spikes) )

        # if time_before_spikes.size == 20:
        metrics = (fr - self.fr_el)**2 # np.sum( (self.etalonspikes - time_before_spikes )**2  )   # np.sum( (V - self.etalonV)**2 )
        # else:
        # metrics = (isi - self.isi)**2


        return metrics

class Animator:
    def __init__(self, model, xlim, ylim):
        self.Fig, self.ax = plt.subplots(nrows=3, ncols=1)
        self.line1, = self.ax[0].plot([], [], 'b', animated=True)
        self.time_text = self.ax[0].text(0.05, 0.9, '', transform=self.ax[0].transAxes)

        self.ax[0].set_xlim(xlim[0], xlim[1])
        self.ax[0].set_ylim(ylim[0], ylim[1])

        self.line2, = self.ax[1].plot([], [], 'b', animated=True)

        self.ax[1].set_xlim(xlim[2], xlim[3])
        self.ax[1].set_ylim(ylim[2], ylim[3])

        self.line3, = self.ax[2].plot([], [], 'b', animated=True)

        self.ax[2].set_xlim(xlim[4], xlim[5])
        self.ax[2].set_ylim(ylim[4], ylim[5])

        self.model = model

    def update_on_plot(self, idx):
        x1, y1, x2, y2, x3, y3 = self.model.update(self.dt)

        self.line1.set_data(x1, y1)
        self.time_text.set_text("simulation time is %.2f in ms" % idx)

        self.line2.set_data(x2, y2)

        self.line3.set_data(x3, y3)

        return [self.line1, self.time_text, self.line2, self.line3]

    def run(self, dt, duration, interval=10):
        self.dt = dt
        self.duration = duration
        ani = FuncAnimation(self.Fig, self.update_on_plot, frames=np.arange(0, self.duration, self.dt),
                                interval=interval, blit=True, repeat=False)

        plt.show()


##############################################
def main_opt():

    params = {

        "Iextmean" : 0.5,
        "Iextvarience" : 0,
        "ENa" : 50.0,
        "EK" : -85.0,
        "El" : -60.0,
        "gbarNa" : 55.0,
        "gbarK" :8.0,
        "gl" : 0.1,
        "fi" : 10,
        "Capacity": 1,

        "Vt" : -58.32,
        "n_reset" : 0.37, # 0.376326530612, # 0.35, # 0.37,  #
        "V_reset" : 26.0,


        "is_use_CBRD" : False,
        "refactory": 5.0,

        # "Nro": 400,
        # "dts": 0.5,
        "N" : 1,

    }

    cluster_neuron_params = params.copy()
    cluster_neuron_params["Iextmean"] = 0.2
    cluster_neuron_params["Iextvarience"] = 0.0
    cluster_neuron_params["fi"] = 5
    cluster_neuron_params["gbarKS"] = 12
    cluster_neuron_params["gbarH"] = 1.0
    cluster_neuron_params["Eh"] = -40.0
    cluster_neuron_params["El"] = -50.0

    cluster_neuron_params["Vt"] = -51.5 # -43.835308
    cluster_neuron_params["V_reset"] = 57.760025
    cluster_neuron_params["n_reset"] = 0.65 # 0.37
    cluster_neuron_params["p_reset"] = 0.028  # 0.100481

    cluster_neuron_params["q_reset"] = 0.008 # 0.012506
    cluster_neuron_params["H_reset"] = 0.008 # 0.008850

    dt = 0.1
    duration = 1000


    neuron = ClusterNeuron(cluster_neuron_params)
    neuron.update(dt, duration)

    V = np.asarray(neuron.getVhist())

    t = np.linspace(0, duration, V.size)

    # H = np.asarray(neuron.Hhist)
    # n = np.asarray(neuron.nhist)
    #
    # p = np.asarray(neuron.phist)
    q = np.asarray(neuron.qhist)
    #
    # m = np.asarray(neuron.mhist)
    # h = np.asarray(neuron.hhist)
    #
    # gNa = cluster_neuron_params["gbarNa"] * m * m * m * h  # * (cluster_neuron_params["ENa"] - V)
    #
    # gNa_level = 0.047 # np.percentile(gNa, 25)
    # Vt_arr = V[gNa < gNa_level]
    #
    # Vt = np.max(Vt_arr)
    # print ("Threshold is %f" % Vt )
    #
    # peaks_idx = argrelmax(V)[0]
    #
    # V_peaks = V[peaks_idx]
    # V_peaks = V_peaks[V_peaks > 0]
    #
    # V_reset = np.mean(V_peaks)
    # print("V reset is %f" % V_reset)
    #
    # n_reset_arr = n[np.abs(V - V_reset) < 2]
    # n_reset = np.mean(n_reset_arr)
    # print("n reset is %f" % n_reset)
    #
    # p_reset_arr = p[np.abs(V - V_reset) < 2]
    # p_reset = np.mean(p_reset_arr)
    # print("p reset is %f" % p_reset)
    #
    #
    # q_reset_arr = q[argrelmin(q)[0]] - q[(V <= V_reset) & ( (V - V_reset) < -0.5)]
    # q_reset = np.mean(q_reset_arr)
    # print("q reset is %f" % q_reset)
    #
    # H_reset_arr = H[argrelmin(H)[0]] - H[(V <= V_reset) & ( (V - V_reset) < -0.5)]
    # H_reset = np.mean(H_reset_arr)
    # print("H reset is %f" % H_reset)
    #
    #
    # # self.gK = self.gbarK * self.n * self.n * self.n * self.n
    # # self.gH = self.gbarH * self.H
    # # self.gKS = self.gbarKS * self.p * self.q
    #
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    #
    # ax[0].plot(t, V)
    # ax[0].plot([0, duration], [Vt, Vt], color="red")
    #
    # # ax[1].plot(t, neuron.mhist, color="blue", label="m")
    #
    # # ax[1].plot(t, n, color="blue", label="n")
    # ax[1].plot(t, H, color="green", label="H")
    # # ax[1].plot(t, p, color="black", label="p")
    # # ax[1].plot(t, q, color="red", label="q")
    # # ax[1].plot(t, m, color="m", label="m")
    # # ax[1].plot(t, h, color="c", label="h")
    #
    # ax[1].legend()
    #
    # ax[2].plot(t, gNa, color="m", label="INa")
    # ax[2].plot([0, duration], [gNa_level, gNa_level], color="red")
    # ax[2].legend()
    #
    #
    # plt.show()

    neuron_th = ClusterNeuron_Th(cluster_neuron_params)

    # time_before_spike2 = argrelmax(V2)[0][1 : 50]





    neuron_th.update(dt, duration)
    V1 = np.asarray(neuron_th.Vhist)




    # opt = Optimizer(neuron, neuron_th)
    #
    # res = opt.run_optimization()

    # neuron_th.default()
    # neuron_th.Vt = res.x[0]
    # neuron_th.V_reset = res.x[1]
    # neuron_th.n_reset = res.x[2]

    # n_reset = np.linspace(0.1, 0.9, 1000)
    #
    # metrics = np.zeros_like(n_reset)
    #
    # isi1_arr = []
    #
    # for idx, n in enumerate(n_reset):
    #     neuron_th.default()
    #     # neuron_th.Vt = res.x[0]
    #     # neuron_th.V_reset = res.x[1]
    #     neuron_th.n_reset = n
    #
    #     neuron_th.update(dt, duration)
    #
    #     V1 = np.asarray(neuron_th.Vhist)
    #
    #     time_before_spike1 = argrelmax(V1)[0][1 : 20] * dt
    #
    #     isi1 = np.mean(np.diff(time_before_spike1))
    #     isi1_arr.append(isi1)
    #
    #     metrics[idx] =  np.sum( (time_before_spike1 - time_before_spike2)**2 ) # np.sum( (V1 - V2)**2 ) # np.sum( (isi1 - isi2)**2 ) #
    #
    # opt_n = n_reset[np.argmin(metrics)]
    #
    # print (isi1_arr[np.argmin(metrics)], isi2)
    # print (opt_n, metrics.min() )
    #
    # neuron_th.default()
    # neuron_th.n_reset = opt_n
    #neuron_th.update(dt, duration)
    # opt_V = np.asarray(neuron_th.Vhist)
    #
    # t = np.linspace(0, duration / 1000, V2.size)
    #
    #
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)


    ax[0].plot(t, neuron_th.qhist, color="red")
    ax[0].plot(t, q, color="green")



    ax[1].plot(t, V, color="green")
    ax[1].plot(t, V1, color="red")



    plt.show()

def main_CBRD_animation():

    neuron_params = {

        "Iextmean" : 0.5,
        "Iextvarience" : 0.2,
        "ENa" : 50.0,
        "EK" : -90.0,
        "El" : -60.0,
        "gbarNa" : 55.0,
        "gbarK" :8.0,
        "gl" : 0.1,
        "fi" : 10,
        "Capacity" : 1,

        "is_use_CBRD" : True,
        "refactory": 5.0,

        "Vt" : -58.32,
        "n_reset" : 0.37, # 0.376326530612, # 0.35, # 0.37,  #
        "V_reset" : 26.0,

        "Nro": 400,
        "dts": 0.5,

        "N" : 100, }

    cluster_neuron_params = neuron_params.copy()
    cluster_neuron_params["Iextmean"] = 2.2
    cluster_neuron_params["Iextvarience"] = 1.5
    cluster_neuron_params["fi"] = 5
    cluster_neuron_params["gbarKS"] = 12
    cluster_neuron_params["gbarH"] = 1.0
    cluster_neuron_params["Eh"] = -40.0
    cluster_neuron_params["El"] = -50.0

    cluster_neuron_params["Vt"] = -51.5 # -43.835308
    cluster_neuron_params["V_reset"] = 57.760025
    cluster_neuron_params["n_reset"] = 0.65 # 0.37
    cluster_neuron_params["p_reset"] = 0.028  # 0.100481

    cluster_neuron_params["q_reset"] = 0.008 # 0.012506
    cluster_neuron_params["H_reset"] = 0.008 # 0.008850


    dt = 0.1
    duration = 1500

    cbrd = ClusterNeuron_Th(cluster_neuron_params) # FS_neuron_Th(neuron_params)

    neuron_params["is_use_CBRD"] = False
    monte_carlo = ClusterNeuron_Th(cluster_neuron_params)
    # # animator = Animator(cbrd, [0, 200, 0, duration, 0, 200], [0, 1, 0, 400, -85, -55])
    # # animator.run(dt, duration, 0)
    #
    cbrd.update(dt, duration)
    monte_carlo.update(dt, duration)

    plt.subplot(211)
    plt.plot(cbrd.times, cbrd.firing, color="green", label='CBRD')
    plt.subplot(212)
    plt.plot(monte_carlo.times, monte_carlo.firing, color="blue", label='Monte-Carlo')
    plt.legend()
    plt.show()



def  main_Graham_neuron():

    neuron_params = {
        "C" : 0.7, # mkF / cm^2
        "Vreset" : -40,
        "Vt" : -55,
        "Iext" : 0.3, # nA / cm^2
        "saveV": False,
        "refactory" : 2.5,
        "Iextvarience" : 0.1,

        "is_use_CBRD": True,

        "Nro": 400,
        "dts": 0.5,
        "N": 1500,

        "leak"  : {"E" : -61.22, "g" : 0.025 },
        "dr_current" : {"E" : -70, "g" : 0.76, "x" : 1, "y" : 1, "x_reset" : 0.26, "y_reset" : 0.47},  # "g" : 0.76
        "a_current": {"E": -70, "g": 2.3, "x": 1, "y": 1, "x_reset" : 0.74,  "y_reset" : 0.69}, # 2.3 "g": 4.36,
        "m_current": {"E": -80, "g": 0.4, "x": 1, "y": None, "x_reset" : 0.18, "y_reset" : None }, # 0.4 "g": 0.76,
        "ahp": {"E": -70, "g": 0.32, "x": 1, "y": None, "x_reset" : 0.018, "y_reset" : None}, # 0.32 "g": 0.6,
        "hcn" : { "E": -17, "g": 0.003, "x": None, "y": 1, "x_reset" : None, "y_reset" : 0.002 }, # 0.003
    }

    dt = 0.1
    duration = 500
    cbrd = BorgGrahamNeuron(neuron_params)

    # assert(False)

    cbrd.update(dt, duration)

    # animator = Animator(cbrd, [0, 200, 0, duration, 0, 200], [0, 1, 0, 1000, 0, 20])
    # animator.run(dt, duration, 0)

    # neuron_params["is_use_CBRD"] = False
    neuron_params["is_use_CBRD"] = False
    monte_carlo = BorgGrahamNeuron(neuron_params)

    monte_carlo.update(dt, duration)

    mc_firing = np.asarray(monte_carlo.firing)
    win = np.ones(10)  # parzen(15)
    win /= np.sum(win)
    mc_firing = np.convolve(mc_firing, win, mode="same")

    t = np.linspace(0, duration, len(cbrd.firing) )
    plt.figure()
    plt.plot(t, cbrd.firing, color="green", label="CBRD", linewidth=4)
    plt.plot(t, mc_firing, color="blue", label="Monte-Carlo")
    plt.xlim(0, duration)
    plt.ylim(0, 1000)
    plt.legend()

    # plt.show(block=False)
    # monte_carlo_V = np.asarray(monte_carlo.Vhist)
    # t = np.linspace(0, duration, monte_carlo_V.size)
    #
    #
    # plt.figure()
    # plt.plot(t, monte_carlo_V, color="blue", label="Monte-Carlo")
    # plt.plot(t, cbrd.Vhist, color="green", label="CBRD")
    # plt.xlim(0, duration)
    # plt.ylim(-90, 0)
    # plt.legend()
    #
    #
    # plt.figure()
    # plt.plot(cbrd.V, color="green", label="CBRD")
    # plt.xlim(0, 200)
    # plt.ylim(-90, 0)
    # plt.legend()


    plt.show()


if __name__ == "__main__":
    # main_CBRD_animation()
    # main_opt()

    main_Graham_neuron()

