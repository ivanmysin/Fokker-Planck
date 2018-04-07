import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import erf
from scipy.stats import pearsonr
from scipy.signal import argrelextrema, parzen
# from scipy.signal.parzen

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

        dro = dt * self.ro * H  # -self.ro * np.exp(-H * dt)  #

        self.ro -= dro


        self.ro[self.ro < 0] = 0

        self.ro_H_integral += np.sum(dro)



        self.ts += dt

        return shift


class Neuron(CBRD):

    def __init__(self, params):
        self.V = params["V0"]
        self.Vreset = params["Vreset"]

    def add_Isyn(self, Isyn):
        pass

    def update(self, dt):
        return 0, 0, 0, 0

    def reset(self):
        self.V = self.Vreset

    def get_flow(self):
        return 0

class LIF_Neuron(Neuron):

    def __init__(self, params):
        self.Vreset = params["Vreset"]
        self.Vt = params["Vt"]
        self.gl = params["gl"]
        self.El = params["El"]
        self.C = params["C"]
        self.sigma = params["sigma"]

        self.refactory = params["refactory"]

        self.Iext = params["Iext"]
        self.g_tot = self.gl

        self.internal_reset =  params["internal_reset"]

        if self.internal_reset:
            self.V = np.zeros(params["Nn"]) + self.Vreset
        else:
            CBRD.__init__(self, params["Nro"], params["dts"])
            self.V = np.zeros(self.Nro) + self.Vreset

            self.ref_idx = int (self.refactory // self.dts)

            if not params["tau_t"] is None:
                self.Vt -= np.exp(-params["tau_t"] / (self.t_states + 0.000001) )


        if self.internal_reset:
            self.Vhist = []


        self.tau_m = self.C / self.g_tot
        self.Isyn = 0
        self.t = 0

        self.firings = []
        self.time = []

        self.artifitial_generator = False

    def add_Isyn(self, Isyn):
        self.Isyn += Isyn

    def update(self, dt):

        # print(self.Iext, self.Isyn)

        dVdt = -self.V/self.tau_m + self.Iext/self.tau_m + self.Isyn

        #(self.gl * (self.El - self.V) + self.Iext + self.Isyn) / self.C
        dVdt[:self.ref_idx] = 0

        self.V += dt * dVdt
        self.Isyn = 0
        self.t += dt

        if self.internal_reset:
            self.V[self.V >= self.Vt] = self.Vreset
        else:
            shift = self.update_ro(dt, dVdt, self.tau_m)

            if shift:
                self.V[:-1] = np.roll(self.V[:-1], 1)
                self.V[0] = self.Vreset

        self.firings.append(1000 * self.ro[0])
        self.time.append(self.t)

        return self.t_states, self.ro, self.time, np.asarray(self.firings)

    def get_flow(self):
        return self.ro[0]



class  SineGenerator(Neuron):
    def __init__(self, params):

        self.fr = params["fr"]
        self.phase = params["phase"]
        self.amp_max = params["amp_max"]
        self.amp_min = params["amp_min"]
        self.flow = 0
        self.t = 0
        self.hist = []

        self.artifitial_generator = True

    def update(self, dt):
        self.flow = 0.5 * (np.cos(2*np.pi*self.fr*self.t + self.phase) + 1) * (self.amp_max - self.amp_min) + self.amp_min
        self.t += 0.001 * dt
        self.hist.append(self.flow)
        return 0, 0, self.flow, 0

    def get_flow(self):
        return self.flow

    def add_Isyn(self, Isyn):
        pass

    def get_hist(self):
        return np.asarray(self.hist)

class PoissonGenerator:
    def __init__(self, params):

        self.fr = params["fr"]
        self.w = params["w"]
        self.refactory = params["refactory"]
        self.length = params["length"]

        self.flow = 0
        self.previos_t = params["refactory"] + 10
        self.start = self.length + 1

        self.hist = []
        self.artifitial_generator = True


    def update(self, dt):

        if self.flow == self.w:
            self.start += dt
            if self.start >= self.length:
                self.flow = 0
                self.previos_t = 0

        elif self.flow == 0:

            if self.previos_t > self.refactory:
                r = 1000 * np.random.rand() / dt
                if (r < self.fr):
                    self.flow = self.w
                    self.start = 0



        self.previos_t += dt
        self.hist.append(self.flow)
        return 0, 0, self.flow, 0


    def get_hist(self):
        return np.asarray(self.hist)

    def get_flow(self):
        return self.flow

    def add_Isyn(self, Isyn):
        pass

class Network:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses

        self.Nn = 0
        for neuron in self.neurons:
            if not (neuron.artifitial_generator):
                self.Nn += 1

    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt


        t = 0
        while(t < duration):

            sum_Pts = 0
            sum_flow = 0

            for neuron in self.neurons:
                ts_states, Pts, times, flow  = neuron.update(dt)

                if not (neuron.artifitial_generator):
                    sum_Pts += Pts
                    sum_flow += flow

            sum_Pts /= self.Nn
            sum_flow /= self.Nn

            for synapse in self.synapses:
                synapse.update(dt)

            t += dt

        generators_signal = []

        for neuron in self.neurons:
            if (neuron.artifitial_generator):
                generators_signal.append(np.asarray( neuron.hist) )

        return ts_states, sum_Pts, times, sum_flow, generators_signal

    def getflow(self):
        flow = 0
        for neuron in self.neurons:
            flow += np.asarray( neuron.firings )

        flow /= len(self.neurons)

        return flow

class Synapse:
    def __init__(self, params):
        self.w = params["w"]
        self.delay = params["delay"]
        self.pre = params["pre"]
        self.post = params["post"]

        self.pre_hist = []

    def update(self, dt):

        if (len(self.pre_hist) == 0)  and (self.delay > 0):

            for _ in range(int(self.delay/dt)):
                self.pre_hist.append(0)

        pre_flow = self.pre.get_flow()

        if (self.delay > 0):
            self.pre_hist.append(pre_flow)
            pre_flow = self.pre_hist.pop(0)

        self.post.add_Isyn(pre_flow * self.w)

        return

############################################

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


        # self.line3, = self.ax[2].plot([], [], 'b', animated=True)
        #
        # self.ax[2].set_xlim(xlim[4], xlim[5])
        # self.ax[2].set_ylim(ylim[4], ylim[5])


        self.model = model


    def update_on_plot(self, idx):

        x1, y1, x2, y2 = self.model.update(self.dt)


        self.line1.set_data(x1, y1)
        self.time_text.set_text("simulation time is %.2f in ms" % idx)

        self.line2.set_data(x2, y2)

       # self.line3.set_data(x3, y3)

        return [self.line1, self.time_text, self.line2] # , self.line3


    def run(self, dt, duration, interval=10):

        self.dt = dt
        self.duration = duration
        ani = FuncAnimation(self.Fig, self.update_on_plot, frames=np.arange(0, self.duration, self.dt), interval=interval, blit=True, repeat=False)

        plt.show()



#############################################

def run_simulation(Nn=15, std_of_Vt=3.0, iext=12, ntimesinputs=0, nspatialinputs=0):

    if std_of_Vt == 0:
        Nn = 1

    neuron_params = {
        "Vreset" : -10, # -60,
        "Vt" : 20, # -50,
        "gl" : 0.01,  # 0.02,
        "El" : 0, # -60
        "C" : 0.2,
        "Iext" : iext, # 0.14, # 0.15,
        "sigma" : 3.0 / np.sqrt(2),
        "refactory": 5.0,
        "internal_reset" : False,
        "Nro" : 400,
        "dts" : 0.5,
        "tau_t" : 0,
    }

    sine_generator_params = {
        "fr" : 2,
        "phase" : 0,
        "amp_max" : 0.025 * Nn,
        "amp_min" : 0,
    }

    poisson_generator_params = {

        "fr" : 25, # 3,
        "w" : 0.05 * Nn,
        "refactory" : 10,
        "length" : 5,
    }

    synapse_params = {
        "w" : 20.0,
        "delay" : 2,
        "pre" : 0,
        "post" : 0,
    }

    dt = 0.1
    duration = 900

    neurons = []
    synapses = []
    synapse_params["w"] /= Nn

    for _ in range(ntimesinputs):
        neuron = SineGenerator(sine_generator_params)
        neurons.append(neuron)

    for _ in range(nspatialinputs):
        neuron = PoissonGenerator(poisson_generator_params)
        neurons.append(neuron)

    for _ in range(Nn):
        neuron_params_tmp = neuron_params.copy()

        if Nn > 1:
            neuron_params_tmp["Vt"] += std_of_Vt * np.random.standard_normal()

        neuron = LIF_Neuron(neuron_params_tmp)
        neurons.append(neuron)


    for pre_idx in range(Nn):
        for post_idx in range(Nn):
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = Synapse(synapse_params_tmp)
            synapses.append(synapse)

    net = Network(neurons, synapses)
    # animator = Animator(net, [0, 200, 0, duration, 0, 200], [0, 1, 0, 1000, 0, 20])
    # animator.run(dt, duration, 0)
    # flow = net.getflow()

    ts_states, sum_Pts, times, flow, generators_signal = net.update(dt, duration)

    return flow, generators_signal

def synchronization_regimes():

    std_vt_arr = np.linspace(0, 3, 12)
    iext_arr = np.linspace(12, 15, 5)

    sych_coeff_arr = np.empty([std_vt_arr.size, iext_arr.size], dtype=np.float)

    for idx1, val1 in enumerate(std_vt_arr):
        for idx2, val2 in enumerate(iext_arr):
            flow, _ = run_simulation(std_of_Vt=val1, iext=val2)

            synch_coeff = np.std(flow[-1000:])

            sych_coeff_arr[idx1, idx2] = synch_coeff

            print (idx1, idx2)

    np.save("synch_coeffs", sych_coeff_arr)
    fig, ax = plt.subplots()
    ax.pcolor(std_vt_arr, iext_arr, sych_coeff_arr.T, cmap='RdGy')

    # plt.colorbar()
    plt.show()
    plt.savefig("sinch_regimes", dpi=500)

def answer_2_inputs():

    std_vt_arr = np.linspace(0, 5, 2) # 10
    iext = 14
    flow_spatial_arr = []
    ppv_arr = []

    win = parzen(15)
    for stdvt in std_vt_arr:
        flow_spatial, generators_signal = run_simulation(std_of_Vt=stdvt, iext=iext, nspatialinputs=1)
        generators_signal = generators_signal[0]

        flow_spatial = np.convolve(flow_spatial, win, mode="same")

        flow_spatial_arr.append(flow_spatial)
        t = np.linspace(0, 900, flow_spatial.size)

        answs_idx = argrelextrema(flow_spatial, np.greater, order=150)[0]
        input_idx = np.argwhere(np.diff(generators_signal) < 0)

        answs_idx = answs_idx[ flow_spatial[answs_idx] > np.percentile(flow_spatial, 80) ]

        print(answs_idx)

        ppv = calculate_ppv(input_idx, answs_idx)
        ppv_arr.append(ppv)

        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.scatter(t[input_idx], np.zeros_like(t[input_idx]) + np.max(flow_spatial))

        axs.scatter(t[answs_idx], flow_spatial[answs_idx], color="black")
        axs.plot(t, flow_spatial, color="red")

        plt.show(block=False)



    flow_spatial_arr = np.asarray(flow_spatial_arr)
    np.save("spatial_input", flow_spatial_arr)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(std_vt_arr, ppv_arr)
    axs.set_title("PPV")
    plt.show(block=False)


    # std_vt_arr = np.linspace(0, 12, 2) # 24
    # flow_time_arr = []
    # corr_arr = []
    # for stdvt in std_vt_arr:
    #     flow_time, generators_signal = run_simulation(std_of_Vt=stdvt, iext=iext, nspatialinputs=0, ntimesinputs=1)
    #     t = np.linspace(0, 900, flow_time.size)
    #     generators_signal = generators_signal[0]
    #
    #     flow_time_arr.append(flow_time)
    #     R, p = pearsonr(flow_time, generators_signal)
    #     corr_arr.append(R)
    #
    #
    #     fig, axs = plt.subplots(nrows=1, ncols=1)
    #     sine = generators_signal * np.std(flow_time) + np.mean(flow_time)
    #     axs.plot(t, sine, color="blue")
    #     axs.plot(t, flow_time, color="red" )
    #     plt.show(block=False)
    #
    #
    # flow_time_arr = np.asarray(flow_time_arr)
    # np.save("time_input", flow_time_arr)
    #
    # fig, axs = plt.subplots(nrows=1, ncols=1)
    # axs.plot(std_vt_arr, corr_arr)
    # axs.set_title("I/O correlation")
    plt.show()

def calculate_ppv(input, output):
    tp = 0
    fp = 0

    if output.size == 0:
        return 0

    for ans in output:
        n = np.sum( np.abs(ans - input) < 50 )
        if n == 0:
            fp += 1
        else:
            tp += 1
    ppv = tp / (tp + fp)

    return ppv


def main():
    # synchronization_regimes()
    answer_2_inputs()


if __name__ == "__main__":
    main()

    # t = np.linspace(0, 1, 100)
    #
    # input = np.zeros(100)
    # output = np.zeros(100)
    #
    # input[0::10] = 1
    # input[1::10] = 1
    # input[2::10] = 1
    # input[3::10] = 1
    #
    # output[2::10] = 1
    #
    # answs_idx = argrelextrema(output, np.greater, order=4)[0]
    # input_idx = np.argwhere( np.diff(input) < 0 )
    #
    #
    #
    # print(tp, fp)
    #
    # plt.plot(t, input)
    # plt.plot(t, output)
    #
    # plt.scatter(t[input_idx], input[input_idx] + 0.0001 )
    #
    #
    #
    # plt.show()



