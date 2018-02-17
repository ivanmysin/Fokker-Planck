import numpy as np
import matplotlib.pyplot as plt

class Channel:
    def __init__(self, gmax, E, V, x=None, y=None):
        self.gmax = gmax
        self.E = E
        self.g = 0
        if not x is None:
            self.x = self.get_x_inf(V)
        else:
            self.x = None

        if not y is None:
            self.y = self.get_y_inf(V)
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

    def reset(self):
        pass


    def get_x_inf(self, V):
        return 0

    def get_y_inf(self, V):
        return 0

    def get_tau_x(self, V):
        return 0

    def get_tau_y(self, V):
        return 0

class KDR_Channel(Channel):

    def reset(self):
        self.x = 0.262
        self.y = 0.473


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

    def reset(self):
        self.x = 0.743
        self.y = 0.691

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

    def reset(self):
        self.x += 0.175*(1 - self.x)

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
        tau_x = 2000 / (3.3 * np.exp( (V + 35)/20 ) ) + np.exp(-(V + 35)/20)
        return tau_x

    def get_x_inf(self, V):
        x_inf = 1 / (1 + np.exp(-(V + 35)/10))
        return x_inf

    def reset(self):
        self.x += 0.018*(1 - self.x)

class BorgGrahamNeuron:
    def __init__(self, params):
        self.V = params["V0"]
        self.C = params["C"]
        self.Vreset = params["Vreset"]
        self.Vth = params["Vt"]
        self.Iext = params["Iext"]
        self.saveV = params["saveV"]

        leak = Channel(params["leak"]["g"], params["leak"]["E"], self.V)
        dr_current = KDR_Channel(params["dr_current"]["g"], params["dr_current"]["E"], self.V, 1, 1)
        a_current = A_channel(params["a_current"]["g"],params["a_current"]["E"], self.V, 1, 1)
        m_current = M_channel(params["m_current"]["g"], params["m_current"]["E"], self.V, 1)
        ahp = AHP_Channel(params["ahp"]["g"], params["ahp"]["E"], self.V, 1)

        self.channels = [leak, dr_current, a_current, m_current, ahp]

        if  self.saveV:
            self.Vhist = [self.V]

        self.t = 0
        self.ts = 200

        self.refactory = 1.5


    def reset(self, another_neuron = None):
        self.V = self.Vreset
        self.ts = 0


        for idx, ch in enumerate(self.channels):
            if not another_neuron is None:
                ch.x = another_neuron.channels[idx].x
                ch.y = another_neuron.channels[idx].y
            ch.reset()

    def update(self, dt):

        I = 0
        g_tot = 0
        for ch in self.channels:
            ch.update(dt, self.V)
            I -= ch.get_I(self.V)
            g_tot += ch.get_g()

        I += self.Iext
        dVdt = I / self.C
        self.V += dt * dVdt

        if self.saveV:
            self.Vth = max(-50, (-85 + 400 / self.ts))
            if self.V > self.Vth and self.ts > self.refactory:
                self.reset()

        self.t += dt
        self.ts += dt

        if self.saveV:
            self.Vhist.append(self.V)

        return self.V, dVdt, g_tot, self.C

###################################################################
def main():
    neuron_params = {
        "V0" : -65,
        "C" : 0.7,
        "Vreset" : -40,
        "Vt" : -50,
        "Iext" : 5.15,
        "saveV": True,
        "leak"  : {"E" : -65, "g" : 0.07},
        "dr_current" : {"E" : -70, "g" : 0.76, "x" : 1, "y" : 1},
        "a_current": {"E": -70, "g": 4.36, "x": 1, "y": 1},
        "m_current": {"E": -80, "g": 0.76, "x": 1, "y": None},
        "ahp": {"E": -70, "g": 0.6, "x": 1, "y": None},
    }


    neuron = BorgGrahamNeuron(neuron_params)



    for _ in range(1000):
        neuron.update(0.1)

    t = np.linspace(0, 100, len(neuron.Vhist) )
    plt.plot(t, neuron.Vhist)
    plt.show()

if __name__ == "__main__":
    main()
