
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import erf

import Models4CBRD as Mods

class LIF_Neuron:

    def __init__(self, params):
        self.Vreset = params["Vreset"]
        self.Vt = params["Vt"]
        self.gl = params["gl"]
        self.El = params["El"]
        self.C = params["C"]
        self.V = self.Vreset
        self.Iext = params["Iext"]
        self.g_tot = self.gl

        self.Isyn = 0

    def add_Isyn(self, Isyn):
        self.Isyn += Isyn

    def update(self, dt):

        dVdt = ( self.gl*(self.El - self.V) + self.Iext + self.Isyn) / self.C
        # print (self.Isyn)
        self.V += dt * dVdt
        self.Isyn = 0
        return self.V, dVdt, self.g_tot, self.C

    def reset(self, another_neuron):

        self.V = self.Vreset


class State:

    def __init__(self, neuron, P, Vt, refactory, sigma, ts, max_ts):
        self.P = P
        self.neuron = neuron
        self.Vt = Vt
        self.sigma = sigma
        self.ts = ts
        self.max_ts = max_ts
        self.isnotlast = True
        self.refactory = refactory

        self.tmp_var = 0

    def H_function(self, dt, V, dVdt, g_tot, C):

        # print (dt, V, dVdt, g_tot, C)

        tau_m = C / g_tot
         # k = tau_m / dt

        T = (self.Vt - V) / self.sigma / np.sqrt(2) # np.sqrt(0.5 * (1 + k)) * g_tot *

        A = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        # A = A_inf * (1 - (1 + k)**(-0.71 + 0.0825 * (T + 3)))



        dT_dt = -1.0 / self.sigma / np.sqrt(2) * dVdt  # -g_tot / self.sigma * np.sqrt(0.5 + 0.5 * k) * dVdt

        if dT_dt > 0:
            dT_dt = 0
        # dT_dt[dT_dt > 0] = 0


        F_T = np.sqrt(2 / np.pi) * np.exp(-T**2) / (1.000000001 + erf(T))

        B = -np.sqrt(2) * dT_dt * F_T * tau_m

        self.tmp_var = B


        H = (A + B) / tau_m

        # if (H > 1):
        #     H = 1
        #
        # if (H < 0):
        #     H = 0
        # print(H)
        # print("###################")
        return H

    def reset(self, p, another_state):
        self.P = p
        self.neuron.reset(another_state.neuron)

    def update(self, dt):

        V, dVdt, g_tot, C = self.neuron.update(dt)

        ph = 0
        if self.ts > self.refactory and self.P > 0.000001:

            H = self.H_function(dt, V, dVdt, g_tot, C)
            ph = self.P * H

            self.P -= dt * ph

        if self.isnotlast:
            self.ts += dt


        # self.Vt = max([-50, (-85 + 400 / self.ts) ])
        if self.isnotlast and self.ts >= self.max_ts:
            self.ts = 0

        return ph


class CBRD:
    def __init__(self, dts, Nts, neuron_params, sigma, Neuron_class):
        self.states = []
        self.dts = dts
        self.Nts = Nts

        self.neuron_params = neuron_params
        self.Vt = self.neuron_params["Vt"]
        self.sigma = sigma

        self.ts_states = np.linspace(0, self.Nts*self.dts, self.Nts)

        self.Pts = np.zeros_like(self.ts_states)

        self.flow_x = []
        self.flow_y = []
        self.flow = 0

        self.t = 0
        self.firings = 0

        for idx in range(Nts):
            neuron = Neuron_class(neuron_params)
            s = State(neuron, 0, neuron_params["Vt"], neuron_params["refactory"], sigma, self.ts_states[idx], self.ts_states[-1])
            self.states.append(s)

        self.states[-1].isnotlast = False
        self.states[-1].P = 1 / self.dts

        self.Pts[-1] = self.states[-1].P

    def update(self, dt):
        self.flow = 0

        tmp_ph = 0
        s4reset = None
        max_ph_state = None

        # ro_h = []

        for s in self.states:
            ph = s.update(dt)

            # ro_h.append(s.tmp_var) #s.neuron.V

            self.flow += ph * dt

            if ph >= tmp_ph:
                max_ph_state = s

            if s.ts == 0:
                s4reset = s

        self.firings += self.flow

        if not s4reset is None:
            self.states[-1].P += s4reset.P
            s4reset.reset(self.firings, max_ph_state)
            self.firings = 0


        for idx, s in enumerate(self.states):
            self.ts_states[idx] = s.ts
            self.Pts[idx] = s.P

        tmp = np.argsort(self.ts_states)
        self.ts_states = self.ts_states[tmp]
        self.Pts = self.Pts[tmp]

        # ro_h = np.asarray(ro_h)[tmp]
        # print ( np.sum(self.Pts) * self.dts )
        # self.states[-1].tmp_var
        self.flow_x.append(self.t)
        self.flow_y.append( 1000 * self.Pts[0]  ) #self.states[-50].neuron.V)
        self.t += dt

        return self.ts_states, self.Pts, self.flow_x, np.asarray(self.flow_y)

    def get_flow(self):
        return self.flow

    def add_Isyn(self, Isyn):
        for s in self.states:
            s.neuron.add_Isyn(Isyn)

###########################################################################
class Animator:
    def __init__(self, model, xlim, ylim):

        self.Fig, self.ax = plt.subplots(nrows=2, ncols=1)
        self.line1, = self.ax[0].plot([], [], 'b', animated=True)
        self.time_text = self.ax[0].text(0.05, 0.9, '', transform=self.ax[0].transAxes)

        self.ax[0].set_xlim(xlim[0], xlim[1])
        self.ax[0].set_ylim(ylim[0], ylim[1])


        self.line2, = self.ax[1].plot([], [], 'b', animated=True)

        self.ax[1].set_xlim(xlim[2], xlim[3])
        self.ax[1].set_ylim(ylim[2], ylim[3])

        self.model = model


    def update_on_plot(self, idx):

        x1, y1, x2, y2 = self.model.update(self.dt)


        #vself.ax[0].set_xlim(x1.min(), x1.max())
        # self.ax[0].set_ylim(0.8 * y1.min(), 1.2 * y1.max())
        #
        # self.ax[1].set_ylim(0.8 * y2.min(), 1.2 * y2.max())

        self.line1.set_data(x1, y1)
        self.time_text.set_text("simulation time is %.2f in ms" % idx)

        self.line2.set_data(x2, y2)

        return [self.line1, self.time_text, self.line2]

    def run(self, dt, duration, interval=10):

        self.dt = dt
        self.duration = duration
        ani = FuncAnimation(self.Fig, self.update_on_plot, frames=np.arange(0, self.duration, self.dt), interval=interval, blit=True, repeat=False)

        plt.show()
###########################################################################
class Synapse:
    def __init__(self, params):
        self.w = params["w"]
        self.pre = params["pre"]
        self.post = params["post"]

    def update(self, dt):

        pre_flow = self.pre.get_flow()
        self.post.add_Isyn(pre_flow * self.w)

        return

class Network:
    def __init__(self, list_of_cbrd, synapses):
        self.list_of_cbrd = list_of_cbrd
        self.synapses = synapses

    def update(self, dt):

        sum_Pts = 0
        sum_flow_y = 0

        for cbrd in self.list_of_cbrd:
            ts_states, Pts, flow_x, flow_y = cbrd.update(dt)

            sum_Pts += Pts
            sum_flow_y += flow_y

        sum_Pts /= len(self.list_of_cbrd)
        sum_flow_y /= len(self.list_of_cbrd)

        for synapse in self.synapses:
            synapse.update(dt)

        return ts_states, sum_Pts, flow_x, sum_flow_y



###########################################################################
def main():
    N_cbrd = 1

    neuron_params = {
        "Vreset" : 0, # -60,
        "Vt" : 10, # -50,
        "gl" : 0.01,  # 0.02,
        "El" : 0, # -60
        "C" : 0.1,
        "Iext" : 0.15,
        "refactory": 0,
    }
    #
    # neuron_params = {
    #     "V0" : -65,
    #     "C" : 0.7,
    #     "Vreset" : -40,
    #     "Vt" : -50,
    #     "Iext" : 3.15,
    #     "refactory" : 1.5,
    #     "saveV"  : False,
    #     "leak"  : {"E" : -65, "g" : 0.07},
    #     "dr_current" : {"E" : -70, "g" : 0.76, "x" : 1, "y" : 1},
    #     "a_current": {"E": -70, "g": 4.36, "x": 1, "y": 1},
    #     "m_current": {"E": -80, "g": 0.76, "x": 1, "y": None},
    #     "ahp": {"E": -70, "g": 0.6, "x": 1, "y": None},
    # }

    N_syns = 0
    synapse_params = {
        "w" : 200.2,
        "pre_idx": 0,
        "post_idx": 0,

    }

    dts = 0.5
    Nts = 400
    sigma = 2.0

    dt = 0.1
    duration = 150

    list_of_cdrd = []
    synapses = []

    for idx in range(N_cbrd):

        params_tmp = neuron_params.copy()

        #params_tmp["Iext"] += 0.5 * np.random.randn()

        cbrd = CBRD(dts, Nts, params_tmp, sigma, LIF_Neuron) #     Mods.BorgGrahamNeuron
        list_of_cdrd.append(cbrd)

    for idx in range(N_syns):

        params_tmp = synapse_params.copy()

        params_tmp["pre"] = list_of_cdrd[params_tmp["pre_idx"]]

        params_tmp["post"] = list_of_cdrd[params_tmp["post_idx"]]
        synapse = Synapse(params_tmp)

        synapses.append(synapse)

    net = Network(list_of_cdrd, synapses)

    animator = Animator(net, [0, 200, 0, duration], [0, 0.2, 0, 200])
    animator.run(dt, duration, 0)

    # cbrd.run(dt, duration)

main()





