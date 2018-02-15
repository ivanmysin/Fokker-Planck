
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
        self.g_tot = self.gl


    def update(self, dt):

        dVdt = ( self.gl*(self.El - self.V) + self.Iext ) / self.C

        self.V += dt * dVdt

        return self.V, dVdt, self.g_tot, self.C

    def reset(self, another_neuron):

        self.V = self.Vreset


class State:

    def __init__(self, neuron, P, Vt, sigma, ts, max_ts):
        self.P = P
        self.neuron = neuron
        self.Vt = Vt
        self.sigma = sigma
        self.ts = ts
        self.max_ts = max_ts
        self.isnotlast = True

    def H_function(self, dt, V, dVdt, g_tot, C):
        tau_m = C / g_tot
        k = tau_m / dt


        T = np.sqrt(0.5 * (1 + k)) * g_tot * (self.Vt - V) / self.sigma


        A_inf = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        A = A_inf * (1 - (1 + k)**(-0.71 + 0.0825 * (T + 3)))

        dT_dt = -g_tot / self.sigma * np.sqrt(0.5 + 0.5 * k) * dVdt

        if dT_dt < 0:
            dT_dt = 0
        # dT_dt[dT_dt < 0] = 0


        F_T = np.sqrt(2 / np.pi) * np.exp(-T**2) / (1.000000001 + erf(T))

        B = -np.sqrt(2) * tau_m * dT_dt * F_T

        H = (A + B) / tau_m

        # if (H > 1):
        #     H = 1
        #
        # if (H < 0):
        #     H = 0

        return H

    def reset(self, p, another_state):
        self.P = p
        self.neuron.reset(another_state)

    def update(self, dt):

        V, dVdt, g_tot, C = self.neuron.update(dt)
        H = self.H_function(dt, V, dVdt, g_tot, C)
        ph = self.P * H

        self.P -= ph

        if self.isnotlast:
            self.ts += dt

        if self.isnotlast and self.ts >= self.max_ts:
            self.ts = 0

        return ph


class CBRD:
    def __init__(self, dts, Nts, neuron_params, sigma):
        self.states = []
        self.dts = dts
        self.Nts = Nts

        self.neuron_params = neuron_params
        self.Vt = self.neuron_params["Vt"]
        self.sigma = sigma

        self.ts_states = np.linspace(0, self.Nts*self.dts, self.Nts)

        self.Pts = np.zeros_like(self.ts_states)
        self.Pts[-1] = 1
        self.flow_x = []
        self.flow_y = []

        self.t = 0
        self.firings = 0

        for idx in range(Nts):
            neuron = Neuron(neuron_params)
            s = State(neuron, 0, neuron_params["Vt"], sigma, self.ts_states[idx], self.ts_states[-1])
            self.states.append(s)

        self.states[-1].isnotlast = False
        self.states[-1].P = 1


    def update(self, dt):
        firing = 0

        tmp_p = 0
        s4reset = None
        max_p_state = None
        for s in self.states:
            firing += s.update(dt)

            if s.P > tmp_p:
                max_p_state = s

            if s.ts == 0:
                s4reset = s

        self.firings += firing


        if not s4reset is None:
            self.states[-1].P += s4reset.P
            s4reset.reset(self.firings, max_p_state)
            self.firings = 0


        for idx, s in enumerate(self.states):
            self.ts_states[idx] = s.ts
            self.Pts[idx] = s.P

        tmp = np.argsort(self.ts_states)
        self.ts_states = self.ts_states[tmp]
        self.Pts = self.Pts[tmp]

        # print ( np.sum(self.Pts) )

        self.flow_x.append(self.t)
        self.flow_y.append( 1000 * firing / dt ) #self.states[-50].neuron.V)
        self.t += dt

        return self.ts_states, self.Pts, self.flow_x, self.flow_y

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

def main():

    neuron_params = {
        "Vreset" : -60,
        "Vt" : -50,
        "gl" : 0.1,
        "El" : -60,
        "C" : 1,
        "Iext" : 1.5,
    }

    dts = 0.5
    Nts = 400
    sigma = 1.5

    dt = 0.1
    duration = 100
    cbrd = CBRD(dts, Nts, neuron_params, sigma)

    animator = Animator(cbrd, [0, 200, 0, 100], [0, 1.2, 0, 1000])
    animator.run(dt, duration, 1)

    # cbrd.run(dt, duration)

main()





