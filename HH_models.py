# -*- coding: utf-8 -*-
"""
CA1 neuron from
    A Two Compartment Model of a CA1 Pyramidal Neuron
        Katie A. Ferguson∗†and Sue Ann Campbell
        (2009)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import parzen, argrelmax
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
        dro = self.ro * (1 - np.exp(-H * dt))  # dt * self.ro * H  #
        dro[:self.ref_idx] = 0
        self.ro -= dro
        self.ro[self.ro < 0] = 0
        self.ro_H_integral += np.sum(dro)
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

                    self.V[0] = self.V_reset
                    self.n[0] = self.n_reset
                    self.H[0] = self.H_reset
                    self.p[0] = self.p_reset
                    self.q[0] = self.q_reset

            else:
                spiking = np.logical_and( (self.V >= self.Vt), (self.ts > self.refactory) )
                self.V[spiking] = self.V_reset
                self.n[spiking] = self.n_reset
                self.ts += dt
                self.ts[spiking] = 0

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

        "Nro": 400,
        "dts": 0.5,
        "N" : 1,

    }

    cluster_neuron_params = params.copy()
    cluster_neuron_params["Iextmean"] = 0.5
    cluster_neuron_params["Iextvarience"] = 0.0
    cluster_neuron_params["fi"] = 5
    cluster_neuron_params["gbarKS"] = 12
    cluster_neuron_params["gbarH"] = 1.0
    cluster_neuron_params["Eh"] = -40.0
    cluster_neuron_params["El"] = -50.0

    cluster_neuron_params["Vt"] = -50.0
    cluster_neuron_params["V_reset"] = -50.0
    cluster_neuron_params["n_reset"] = 0.37
    cluster_neuron_params["p_reset"] = 0.37
    cluster_neuron_params["q_reset"] = 0.37
    cluster_neuron_params["H_reset"] = 0.37

    dt = 0.1
    duration = 1000


    neuron = ClusterNeuron(cluster_neuron_params)
    neuron.update(dt, duration)

    V = np.asarray(neuron.getVhist())

    t = np.linspace(0, duration, V.size)

    H = np.asarray(neuron.Hhist)
    n = np.asarray(neuron.nhist)

    p = np.asarray(neuron.phist)
    q = np.asarray(neuron.qhist)

    m = np.asarray(neuron.mhist)
    h = np.asarray(neuron.hhist)

    gNa = cluster_neuron_params["gbarNa"] * m * m * m * h  # * (cluster_neuron_params["ENa"] - V)

    Vt = V[gNa < np.percentile(gNa, 20)]
    print (Vt.size)
    print ( np.max(Vt) )

    # self.gK = self.gbarK * self.n * self.n * self.n * self.n
    # self.gH = self.gbarH * self.H
    # self.gKS = self.gbarKS * self.p * self.q

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

    ax[0].plot(t, V)

    # ax[1].plot(t, neuron.mhist, color="blue", label="m")

    ax[1].plot(t, n, color="blue", label="n")
    ax[1].plot(t, H, color="green", label="H")
    ax[1].plot(t, p, color="black", label="p")
    ax[1].plot(t, q, color="red", label="q")

    ax[1].plot(t, m, color="m", label="m")
    ax[1].plot(t, h, color="c", label="h")

    ax[2].plot(t, gNa, color="m", label="INa")
    plt.legend()


    fig = plt.figure()

    plt.hist(gNa, bins=500)




    plt.show()


    # neuron_th = FS_neuron_Th(params)
    #neuron = FS_neuron(params)

    # neuron.update(dt, duration)
    # V2 = np.asarray(neuron.Vhist)

    # time_before_spike2 = argrelmax(V2)[0][1 : 50]





    # neuron_th.update(dt, duration)
    #
    # V1 = np.asarray(neuron_th.Vhist)




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
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)
    #
    # # ax[0].plot(n_reset, metrics)
    #
    # # ax[0].plot(t, V1, color="green")
    # # ax[0].plot(t, V2, color="red")
    #
    # ax[1].plot(t, opt_V, color="green")
    # ax[1].plot(t, V2, color="red")
    #
    #
    #
    # plt.show()

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

        "N" : 4500, }




    dt = 0.1
    duration = 1500

    # cbrd = FS_neuron_Th(neuron_params)
    #
    # neuron_params["is_use_CBRD"] = False
    # monte_carlo = FS_neuron_Th(neuron_params)
    # # animator = Animator(cbrd, [0, 200, 0, duration, 0, 200], [0, 1, 0, 400, -85, -55])
    # # animator.run(dt, duration, 0)
    #
    # cbrd.update(dt, duration)
    # monte_carlo.update(dt, duration)
    #
    # plt.plot(cbrd.times, cbrd.firing, color="green", label='CBRD')
    # plt.plot(monte_carlo.times, monte_carlo.firing, color="blue", label='Monte-Carlo')
    # plt.legend()
    # plt.show()






if __name__ == "__main__":
    # main_CBRD_animation()
    main_opt()

