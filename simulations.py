
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import CBRDlib as lib
from scipy.signal import parzen


class Animator:
    def __init__(self, model, xlim, ylim, path=None):

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
        self.path = path

        self.disridution_y = np.empty((100, 0), dtype=np.float)
        self.disridution_x = np.empty(100, dtype=np.float)



    def plot_distribution_as_pcolor(self):

        plt.pcolor( np.linspace(0, self.disridution_y.shape[1]*self.dt,  self.disridution_y.shape[1]), self.disridution_x, self.disridution_y, cmap='gray_r', vmin=0, vmax=0.5)
        plt.show()


    def update_on_plot(self, idx):


        x1, y1, x2, y2, y3 = self.model.update(self.dt)

        # y3 = self.model.get_artificial_signals()
        V = np.empty(0, dtype=float)
        for n in self.model.neurons:
            V = np.append(V, n.channels[4].get_g())

        disridution_y, disridution_x = np.histogram(V, bins=100, range=[0, 0.5], density=True )
        disridution_x = disridution_x[:-1]
        self.disridution_x = disridution_x

        disridution_y = np.convolve(disridution_y, parzen(7), mode="same")

        self.disridution_y = np.append(self.disridution_y, disridution_y.reshape(-1, 1), axis=1)

        self.line1.set_data(disridution_x, disridution_y)



        self.time_text.set_text("simulation time is %.2f in ms" % idx)



        self.line2.set_data(x2, y2)

        if len(y3) > 0:
            # print(len(x2), len(y3) )
            # self.line3.set_data(x2, y3)
            pass

        return [self.line1, self.time_text, self.line2, self.line3]


    def run(self, dt, duration, interval=1):

        self.dt = dt
        self.duration = duration

        if interval==0:
            interval = 1
        ani = animation.FuncAnimation(self.Fig, self.update_on_plot, frames=np.arange(0, self.duration, self.dt), interval=interval, blit=True, repeat=False)


        plt.show(block=False)

        if not (self.path is None):
            mywriter = animation.FFMpegWriter(fps=100)
            ani.save(self.path+".mp4", writer=mywriter)

        plt.close(self.Fig)

def run_simulation_HH(sim_params):

    if sim_params["std_of_Vt"] == 0 and sim_params["std_of_Iext"] == 0:
        sim_params["Nn"] = 1


    Nn = sim_params["Nn"]

    neuron_params = {
        "C" : 0.3, # mkF / cm^2
        "Vreset" : -40,
        "Vt" : sim_params["Vt"], # -55,
        "Iext" : sim_params["Iext"], # 0.3, # nA / cm^2
        "saveV": False,
        "ref_dvdt" : 1.5,
        "refactory" : 5.5,

        "sigma" : sim_params["sigma"],

        "use_CBRD": sim_params["use_CBRD"],

        "w_in_distr" : 1.0,
        "saveCV" : True,

        "N": sim_params["Nstates"], #400,
        "dts": sim_params["dts"], #0.5,

        "gl" : 0.025,
        "El" : -61.22,

        "leak"  : {"E" : -61.22, "g" : 0.025 },
        "dr_current" : {"E" : -70, "g" : 0.76, "x" : 1, "y" : 1, "x_reset" : 0.26, "y_reset" : 0.47},  # "g" : 0.76
        "a_current": {"E": -70, "g": 2.3, "x": 1, "y": 1, "x_reset" : 0.74,  "y_reset" : 0.69}, # 2.3 "g": 4.36,
        "m_current": {"E": -80, "g": 0.4, "x": 1, "y": None, "x_reset" : 0.18, "y_reset" : None }, # 0.4 "g": 0.76,
        "ahp": {"E": -70, "g": 0.32, "x": 1, "y": None, "x_reset" : 0.018, "y_reset" : None}, # 0.32 "g": 0.6,
        "hcn" : { "E": -17, "g": 0.003, "x": None, "y": 1, "x_reset" : None, "y_reset" : 0.002 }, # 0.003
    }


    if not sim_params["use_CBRD"]:
        neuron_params["N"] = 4000
    else:
        neuron_params["ahp"]["g"] = 0.32 #  -73 # ,



    sine_generator_params = {
        "fr" : sim_params["fsine"],
        "phase" : 0,
        "amp_max" : 0.5,
        "amp_min" : 0,
    }

    poisson_generator_params = {
        "fr" : sim_params["fpoisson"],
        "w" : 0.5,
        "refactory" : 10,
        "length" : 5,
    }

    synapse_params = {
        "w" : sim_params["ext_sinapse_w"], #0.0003
        "delay" : sim_params["sinaptic_delay"],
        "pre" : 0,
        "post" : 0,

        "tau_s" : 5.4,
        "tau_a" : 1.0,
        "gbarS" : 1.0,
        "Erev" : 0,
    }

    dt = 0.1
    duration = sim_params["duration"]

    neurons = []
    synapses = []

    artgen_idxs = []
    neurons_idxs = []


    for _ in range(sim_params["ntimesinputs"]):
        neuron = lib.SineGenerator(sine_generator_params)
        artgen_idxs.append(len(neurons))
        neurons.append(neuron)


    for _ in range(sim_params["nspatialinputs"]):
        neuron = lib.PoissonGenerator(poisson_generator_params)
        artgen_idxs.append(len(neurons))
        neurons.append(neuron)


    std_of_Iext = sim_params["std_of_Iext"]
    if std_of_Iext > 0:
        iext_min = neuron_params["Iext"] - 3 * std_of_Iext
        iext_max = neuron_params["Iext"] + 3 * std_of_Iext
        iext_arr = np.linspace(iext_min, iext_max, Nn)
        i_ext_int = (iext_max - iext_min) / Nn
        p_ext = 1 / (std_of_Iext * np.sqrt(2 * np.pi)) * np.exp(-(neuron_params["Iext"] - iext_arr) ** 2 / (2 * std_of_Iext** 2))
        p_ext = p_ext * i_ext_int
        p_ext /= np.sum(p_ext)

    std_of_Vt = sim_params["std_of_Vt"]
    if std_of_Vt > 0:
        vt_min = neuron_params["Vt"] - 3 * std_of_Vt
        vt_max = neuron_params["Vt"] + 3 * std_of_Vt
        vt_arr = np.linspace(vt_min, vt_max, Nn)
        vt_int = (vt_max - vt_min) / Nn
        p_vt = 1 / (std_of_Vt * np.sqrt(2 * np.pi)) * np.exp(-(neuron_params["Vt"] - vt_arr)**2 / (2 * std_of_Vt** 2))
        p_vt = p_vt * vt_int
        p_vt /= np.sum(p_vt)

    for idx in range(Nn):
        neuron_params_tmp = neuron_params.copy()
        neuron_params_tmp["w_in_distr"] = 1 / Nn

        if std_of_Vt > 0:
            neuron_params_tmp["Vt"] = vt_arr[idx]
            neuron_params_tmp["w_in_distr"] = p_vt[idx]

        if std_of_Iext > 0:
            neuron_params_tmp["Iext"] = iext_arr[idx]
            neuron_params_tmp["w_in_distr"] = p_ext[idx]

        neuron = lib.BorgGrahamNeuron(neuron_params_tmp)


        neurons_idxs.append(len(neurons))
        neurons.append(neuron)


    # set synapses from generators to neurons
    for pre_idx in artgen_idxs:
        for post_idx in neurons_idxs:
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["delay"] = 0
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.SimlestSinapse(synapse_params_tmp)
            synapses.append(synapse)


    # set synapses from neurons to neurons
    for pre_idx in neurons_idxs:
        for post_idx in neurons_idxs:
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]

            if sim_params["synapse_type"] == "two_exponentials":
                synapse = lib.Synapse(synapse_params_tmp)
            elif sim_params["synapse_type"] == "simplest":
                synapse = lib.SimlestSinapse(synapse_params_tmp)

            synapses.append(synapse)

    net = lib.Network(neurons, synapses, sim_params["is_save_distrib"])

    if sim_params["show_animation"]:
        animator = Animator(net, [0, 0.5, 0, duration, 0, duration], [0, 150, 0, 1000, 0, 5.2])
        animator.run(dt, duration, 0)
    else:
        ts_states, sum_Pts, times, flow, artsignals = net.update(dt, duration)

    flow = net.getflow()
    artsignals = net.neurons[0].Vhist   # net.get_artificial_signals()


    if sim_params["is_save_distrib"]:
        x4p, Pxvar = net.get_distrib()

        t = np.linspace(0, flow.size*dt, flow.size)

        plt.pcolor(t, x4p, Pxvar, cmap='gray_r', vmin=0, vmax=0.5)
        plt.show()



    return flow, artsignals

def run_simulation_LIF(sim_params):

    if sim_params["std_of_Vt"] == 0 and sim_params["std_of_Iext"] == 0:
        sim_params["Nn"] = 1


    Nn = sim_params["Nn"]
    neuron_params = {
        "Vreset" : -10, # -60,
        "Vt" : 20, # -50,
        "gl" : 0.01,  # 0.02,
        "El" : 0, # -60
        "C" : 0.2, # 0.2,
        "Iext" : sim_params["Iext"],
        "sigma" : sim_params["sigma"], # 0.2,
        "refactory": sim_params["neuron_refractory"], #3.0,
        "ref_dvdt" : sim_params["neuron_refractory"], # 3.0,

        "N" : 400,
        "dts" : 0.5,
        "use_CBRD" : sim_params["use_CBRD"],

        "w_in_distr" : 1.0,
    }

    if not sim_params["use_CBRD"]:
        neuron_params["N"] = 10000

    #print(neuron_params)

    sine_generator_params = {
        "fr" : sim_params["fsine"],
        "phase" : 0,
        "amp_max" : 0.5,
        "amp_min" : 0,
    }

    poisson_generator_params = {
        "fr" : sim_params["fpoisson"],
        "w" : 0.5,
        "refactory" : 10,
        "length" : 5,
    }

    synapse_params = {
        "w" : sim_params["ext_sinapse_w"], #0.0003
        "delay" : sim_params["sinaptic_delay"],
        "pre" : 0,
        "post" : 0,

        "tau_s" : 5.4,
        "tau_a" : 1.0,
        "gbarS" : 1.0,
        "Erev" : 50,
    }

    dt = 0.1
    duration = sim_params["duration"]

    neurons = []
    synapses = []

    artgen_idxs = []
    neurons_idxs = []


    for _ in range(sim_params["ntimesinputs"]):
        neuron = lib.SineGenerator(sine_generator_params)
        artgen_idxs.append(len(neurons))
        neurons.append(neuron)


    for _ in range(sim_params["nspatialinputs"]):
        neuron = lib.PoissonGenerator(poisson_generator_params)
        artgen_idxs.append(len(neurons))
        neurons.append(neuron)


    std_of_Iext = sim_params["std_of_Iext"]
    if std_of_Iext > 0:
        iext_min = neuron_params["Iext"] - 3 * std_of_Iext
        iext_max = neuron_params["Iext"] + 3 * std_of_Iext
        iext_arr = np.linspace(iext_min, iext_max, Nn)
        i_ext_int = (iext_max - iext_min) / Nn
        p_ext = 1 / (std_of_Iext * np.sqrt(2 * np.pi)) * np.exp(-(neuron_params["Iext"] - iext_arr) ** 2 / (2 * std_of_Iext** 2))
        p_ext = p_ext * i_ext_int
        p_ext /= np.sum(p_ext)

    std_of_Vt = sim_params["std_of_Vt"]
    if std_of_Vt > 0:
        vt_min = neuron_params["Vt"] - 3 * std_of_Vt
        vt_max = neuron_params["Vt"] + 3 * std_of_Vt
        vt_arr = np.linspace(vt_min, vt_max, Nn)
        vt_int = (vt_max - vt_min) / Nn
        p_vt = 1 / (std_of_Vt * np.sqrt(2 * np.pi)) * np.exp(-(neuron_params["Vt"] - vt_arr)**2 / (2 * std_of_Vt** 2))
        p_vt = p_vt * vt_int
        p_vt /= np.sum(p_vt)


    for idx in range(Nn):
        neuron_params_tmp = neuron_params.copy()
        # neuron_params_tmp["w_in_distr"] = 1 / Nn

        if std_of_Vt > 0:
            neuron_params_tmp["Vt"] = vt_arr[idx]
            neuron_params_tmp["w_in_distr"] = p_vt[idx]

        if std_of_Iext > 0:
            neuron_params_tmp["Iext"] = iext_arr[idx]
            neuron_params_tmp["w_in_distr"] = p_ext[idx]

        neuron = lib.LIF_Neuron(neuron_params_tmp)
        neurons_idxs.append(len(neurons))
        neurons.append(neuron)


    # set synapses from generators to neurons
    for pre_idx in artgen_idxs:
        for post_idx in neurons_idxs:
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["delay"] = 0
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.SimlestSinapse(synapse_params_tmp)
            synapses.append(synapse)


    # set synapses from neurons to neurons
    for pre_idx in neurons_idxs:
        for post_idx in neurons_idxs:
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]

            if sim_params["synapse_type"] == "two_exponentials":
                synapse = lib.Synapse(synapse_params_tmp)
            elif sim_params["synapse_type"] == "simplest":
                synapse = lib.SimlestSinapse(synapse_params_tmp)

            synapses.append(synapse)

    net = lib.Network(neurons, synapses, sim_params["is_save_distrib"])

    if sim_params["show_animation"]:
        animator = Animator(net, [-15, 21, 0, duration, 0, duration], [0, 3.0, 0, 1000, 0, 5.2])
        animator.run(dt, duration)
        animator.plot_distribution_as_pcolor()

        flow = net.getflow()
    else:
        ts_states, sum_Pts, times, flow, artsignals = net.update(dt, duration)

        if sim_params["is_save_distrib"]:
            x4p, Pxvar = net.get_distrib()

            t = np.linspace(0, flow.size*dt, flow.size)

            plt.pcolor(t, x4p, Pxvar, cmap='gray_r', vmin=0, vmax=0.5)
            plt.show()


    artsignals = net.neurons[0].Vhist # get_artificial_signals()

    return flow, artsignals


########################################################################################################################
def get_default_simulation_params4LIF():
    default_simulation_params4LIF = {
        "Nn": 15,
        "Iext": 0.2,
        "sigma" : 0.3,
        "ntimesinputs": 0,
        "nspatialinputs": 0,
        "std_of_Vt": 0,
        "std_of_Iext": 0,
        "use_CBRD" : True,
        "neuron_refractory" : 3.0,

        "ext_sinapse_w" : 0.001,
        "sinaptic_delay" : 1.0,


        "fpoisson" : 25,
        "fsine"    : 8,

        "duration" : 500,
        "show_animation" : False,
        "synapse_type" : "two_exponentials",

        "is_save_distrib" : False,

        "Nstates" : 400,
        "dts" : 0.5,
    }

    return default_simulation_params4LIF

def get_default_simulation_params4HH():
    params = {
        "Nn": 15,
        "Iext": 0.2,
        "Vt" : -55,
        "sigma" : 0.3,
        "ntimesinputs": 0,
        "nspatialinputs": 0,
        "std_of_Vt": 0,
        "std_of_Iext": 0,
        "use_CBRD" : True,
        "neuron_refractory" : 5.0,

        "ext_sinapse_w" : 0.001,
        "sinaptic_delay" : 1.0,


        "fpoisson" : 25,
        "fsine"    : 8,

        "duration" : 500,
        "show_animation" : False,
        "synapse_type" : "two_exponentials",

        "Nstates": 400,
        "dts": 0.5,
        "is_save_distrib" : False,
    }

    return params
########################################################################################################################
def variate_Iext_at_different_std_and_w(Iext, variateVt, path, useLIF=True):

    if useLIF:
        simulation_params = get_default_simulation_params4LIF()
    else:
        simulation_params = get_default_simulation_params4HH()
        simulation_params["duration"] = 1500

    simulation_params["Iext"] = Iext

    synaptic_w = np.linspace(0.0001, 0.001, 10)
    if variateVt:
        std_Vt_arr = np.linspace(0, 6, 10)
        std_arr = std_Vt_arr
        synch_coeff_arr = np.empty((synaptic_w.size, std_Vt_arr.size), dtype=np.float)
    else:
        std_Iext_arr = np.linspace(0, 0.2, 10)
        std_arr = std_Iext_arr
        synch_coeff_arr = np.empty((synaptic_w.size, std_Iext_arr.size), dtype=np.float)


    cnt = 1

    for idx1, synw in enumerate(synaptic_w):
        for idx2, parm in enumerate(std_arr):

            params_tmp = simulation_params.copy()

            params_tmp["ext_sinapse_w"] = synw

            if variateVt:
                params_tmp["std_of_Vt"] = parm
            else:
                params_tmp["std_of_Iext"] = parm

            if useLIF:
                flow, _ = run_simulation_LIF(params_tmp)
            else:
                flow, _ = run_simulation_HH(params_tmp)
            # flow = np.random.rand(1002)

            max_of_flow = np.max(flow)
            outfile = path + str(cnt)

            if variateVt:
                np.savez(outfile, flow=flow, synw=synw, std_Vt=parm)
                string_params = "synaptic weight: {:.2}, std of Vt: {:.2}".format(synw, parm)
            else:
                np.savez(outfile, flow=flow, synw=synw, std_Iext=parm)
                string_params = "synaptic weight: {:.2}, std of Iext: {:.2}".format(synw, parm)


            print(outfile)

            fig, ax = plt.subplots()
            ax.plot(np.linspace(0, simulation_params["duration"], flow.size), flow)
            ax.set_xlabel("time, ms")
            ax.set_ylabel("population frequency, Hz")
            ax.text(10, 1.2 * max_of_flow, string_params, fontsize=12)
            ax.set_ylim(0, 1.5 * max_of_flow)
            ax.set_xlim(0, simulation_params["duration"])
            # plt.show()
            fig.savefig(outfile + ".png")

            plt.close(fig)

            # assert(False)
            cnt += 1

            synch_coeff = np.std(flow[-5000:])

            if synch_coeff >= 3:
                synch_coeff = 1
            else:
                synch_coeff = 0

            synch_coeff_arr[idx1, idx2] = synch_coeff

    fig, ax = plt.subplots()
    cax = ax.pcolor(std_arr, synaptic_w, synch_coeff_arr, cmap='gray', vmin=0, vmax=1)
    if variateVt:
        ax.set_xlabel("std of Vt")
    else:
        ax.set_xlabel("std of Iext")
    ax.set_ylabel("synaptic w")
    ax.set_title( "Iext : {}".format(Iext) )
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['asynchronious', 'synchronious'])
    fig.savefig(path + "synch_areas")
    plt.show(block=False)

def synchronization_regimesLIF(path):


    iext_arr = [0.1, 0.2, 0.3] #[0.005, 0.007, 0.009] # np.around(((0.11 * (np.logspace(0, 1, 10, endpoint=True) - 1)) ) * 0.29 + 0.01 , decimals=2 )
    # array([0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.13, 0.17, 0.23, 0.3 ])
    # np.linspace(0.05, 0.3, 5)


    # variateVt = True
    #
    # for iext in iext_arr:
    #     path_tmp = path + "variateVt_iext={}".format(iext) + "/"
    #     if not (os.path.isdir(path_tmp)):
    #         os.mkdir(path_tmp)
    #
    #     variate_Iext_at_different_std_and_w(iext, variateVt, path_tmp)


    variateVt = False
    for iext in iext_arr:
        path_tmp = path + "variateIext_iext={}".format(iext) + "/"
        if not (os.path.isdir(path_tmp)):
            os.mkdir(path_tmp)

        variate_Iext_at_different_std_and_w(iext, variateVt, path_tmp)


def synchronization_regimesHH(path, variateVt = False):


    if not (os.path.isdir(path)):
        os.mkdir(path)

    iext_arr = [0.6, 0.7, 0.8, 0.9] # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


    for iext in iext_arr:

        if variateVt:
            path_tmp = path + "variateVt_iext={}".format(iext) + "/"
        else:
            path_tmp = path + "variateIext_iext={}".format(iext) + "/"

        if not (os.path.isdir(path_tmp)):
            os.mkdir(path_tmp)

        variate_Iext_at_different_std_and_w(iext, variateVt, path_tmp, useLIF=False)


def compare_with_article(path):
    simulation_params = get_default_simulation_params4LIF()

    simulation_params["duration"] = 1500
    simulation_params["synapse_type"] = "two_exponentials" # "simplest"
    simulation_params["ext_sinapse_w"] = 0.0015 #  0.02
    simulation_params["sigma"] = 0.01  #* np.sqrt(20)
    simulation_params["std_of_Vt"] = 0

    simulation_params["neuron_refractory"] = 5.0
    simulation_params["sinaptic_delay"] = 2.0

    iext_arr = np.linspace(0.012, 0.015,10)
    std_Vt_arr = np.linspace(0, 3, 10)


    synch_coeff_arr = np.empty((iext_arr.size, std_Vt_arr.size), dtype=np.float)

    simulation_params["show_animation"] = True
    simulation_params["Iext"] = 0.15

    flow, _ = run_simulation_LIF(simulation_params)

    assert(False)

    cnt = 1
    for idx1, Iext in enumerate(iext_arr):
        for idx2, stdVt in enumerate(std_Vt_arr):
            params_tmp = simulation_params.copy()

            params_tmp["Iext"] = Iext

            params_tmp["std_of_Vt"] = stdVt


            flow, _ = run_simulation_LIF(params_tmp)
            # flow = np.random.rand(1002)

            max_of_flow = np.max(flow)
            outfile = path + str(cnt)


            np.savez(outfile, flow=flow, iext=Iext, std_Vt=stdVt)
            string_params = "Iext: {:.2}, std of Vt: {:.2}".format(Iext, stdVt)


            print(outfile)

            fig, ax = plt.subplots()
            ax.plot(np.linspace(0, simulation_params["duration"], flow.size), flow)
            ax.set_xlabel("time, ms")
            ax.set_ylabel("population frequency, Hz")
            ax.text(10, 1.2 * max_of_flow, string_params, fontsize=12)
            ax.set_ylim(0, 1.5 * max_of_flow)
            ax.set_xlim(0, simulation_params["duration"])
            # plt.show()
            fig.savefig(outfile + ".png")

            plt.close(fig)

            # assert(False)
            cnt += 1

            synch_coeff = np.std(flow[-1000:])

            if synch_coeff >= 3:
                synch_coeff = 1
            else:
                synch_coeff = 0

            synch_coeff_arr[idx1, idx2] = synch_coeff


    fig, ax = plt.subplots()
    cax = ax.pcolor(std_Vt_arr, iext_arr, synch_coeff_arr, cmap='gray', vmin=0, vmax=1)

    ax.set_ylabel("synaptic w")
    # ax.set_title("Iext : {}".format(Iext))
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['asynchronious', 'synchronious'])
    fig.savefig(path + "synch_areas")
    plt.show(block=False)


def generateMCvsCBRD(path):
    paramsofplot = {'legend.fontsize': '18',
              # 'figure.figsize': (15, 5),
              'axes.labelsize': '18',
              'axes.titlesize': '18',
              'xtick.labelsize': '18',
              'ytick.labelsize': '18'}
    plt.rcParams.update(paramsofplot)

    win = parzen(15)
    win /= np.sum(win)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True) #, figsize=(10, 10)


    # sim_params = get_default_simulation_params4LIF()
    #
    # sim_params["duration"] = 500
    # sim_params["Iext"] = 0.3
    # sim_params["ext_sinapse_w"] = 0.0 # 0.002
    # sim_params["use_CBRD"] = False
    # flow_mc, Vhistmc = run_simulation_LIF(sim_params)
    # flow_mc = np.convolve(flow_mc, win, mode="same")
    # t_mc = np.linspace(0, flow_mc.size * 0.1, flow_mc.size)
    #
    # sim_params["use_CBRD"] = True
    # flow_cbrd, Vhistcbrd = run_simulation_LIF(sim_params)
    # t_cbrd = np.linspace(0, flow_cbrd.size * 0.1, flow_cbrd.size)
    #
    # # ax[0].plot(t_mc, flow_mc, "r", label="Монте-Карло", color="k", linestyle=":")
    # ax[0].plot(t_cbrd, flow_cbrd, "b", label="РРП", color="k", linestyle="-")
    # ax[0].set_ylim(0, 1.5 * flow_cbrd.max())
    # ax[0].set_xlabel("время, мс")
    # ax[0].set_ylabel("частота, Гц")
    # ax[0].legend()

    # ax[1].plot(t_cbrd, Vhistcbrd)



    sim_params = get_default_simulation_params4HH()


    sim_params["duration"] = 500
    sim_params["sigma"] = 0.3
    sim_params["Iext"] = 0.9
    sim_params["Vt"] = -55.0
    sim_params["std_of_Iext"] = 0.2

    sim_params["Nstates"] = 800
    sim_params["dts"] = 0.5

    sim_params["ext_sinapse_w"] = 0.0
    sim_params["use_CBRD"] = False

    flow_mc, Vhistmc = run_simulation_HH(sim_params)
    flow_mc = np.convolve(flow_mc, win, mode="same")
    t_mc = np.linspace(0, flow_mc.size * 0.1, flow_mc.size)

    sim_params["show_animation"] = False
    sim_params["use_CBRD"] = True
    #sim_params["sigma"] = 0.165


    flow_cbrd, Vhistcbrd = run_simulation_HH(sim_params)
    t_cbrd = np.linspace(0, flow_cbrd.size * 0.1, flow_cbrd.size)




    ax[1].plot(t_mc, flow_mc, "r", label="Монте-Карло", color="r", linestyle=":")
    ax[1].plot(t_cbrd, flow_cbrd, "b", label="РРП", color="g", linestyle="-")
    ax[1].set_xlim(0, sim_params["duration"])
    ax[1].set_ylim(0, 1.5*flow_cbrd.max())
    ax[1].set_xlabel("время, мс")
    ax[1].set_ylabel("частота, Гц")
    ax[1].legend()


    ax[0].plot(t_mc, Vhistmc, "r", label="Монте-Карло", color="k", linestyle=":")

    ax[0].plot(t_cbrd, Vhistcbrd, "b", label="РРП", color="k", linestyle="-")
    ax[0].set_ylim(-80, 5)
    ax[0].set_xlabel("время, мс")
    ax[0].set_ylabel("Потенциал, мВ")
    ax[0].legend()




    fig.savefig(path+"fig.png")
    plt.show()


def plot_distribution_of_vars(path):
    paramsofplot = {'legend.fontsize': '14',
              # 'figure.figsize': (15, 5),
              'axes.labelsize': '14',
              'axes.titlesize': '14',
              'xtick.labelsize': '14',
              'ytick.labelsize': '14'}
    plt.rcParams.update(paramsofplot)

    sim_params = get_default_simulation_params4HH() #get_default_simulation_params4LIF()

    sim_params["duration"] = 1000
    sim_params["Iext"] = 0.5
    sim_params["sigma"] = 0.3
    sim_params["ext_sinapse_w"] = 0.001 # 0.002
    sim_params["use_CBRD"] = True

    sim_params["std_of_Iext"] = 0.0

    sim_params["show_animation"] = True
    sim_params["is_save_distrib"] = True

    sim_params["Nstates"] = 800
    sim_params["dts"] = 1.5

    flow_mc, Vhistmc = run_simulation_HH(sim_params)  #run_simulation_LIF(sim_params)




########################################################################################################################
if __name__ == "__main__":
    # path ="/home/ivan/Data/CBRD/synchronization_LIF/"
    # synchronization_regimesLIF(path)

    # path ="/home/ivan/Data/CBRD/synchronization_HH/"
    # synchronization_regimesHH(path)

    # path = "/home/ivan/Data/CBRD/synchronization_LIF/compare_with_article/"
    # compare_with_article(path)

    path = "/home/ivan/Data/CBRD/MCvsCBRD/"
    generateMCvsCBRD(path)

    # path = "/home/ivan/Data/CBRD/distribution_of_vars/"
    # plot_distribution_of_vars(path)







# flow, _ = run_simulation_LIF(simulation_params)


# neuron_params = {
#     "C" : 0.3, # mkF / cm^2
#     "Vreset" : -40,
#     "Vt" : -55,
#     "Iext" : 0.9, # nA / cm^2
#     "saveV": False,
#     "refactory" : 5.5,
#     "ref_dvdt": 0.5,
#     "sigma" : 0.1,
#     "El" : -61.22,
#     "gl" : 0.025,
#
#     "use_CBRD": True,
#
#     "w_in_distr" : 1.0,
#
#     "N": 400,
#     "dts": 0.5,
#
#
#     # "leak"  : {"E" : -61.22, "g" : 0.025 },
#     "dr_current" : {"E" : -70, "g" : 0.76, "x" : 1, "y" : 1, "x_reset" : 0.26, "y_reset" : 0.47},  # "g" : 0.76
#     "a_current": {"E": -70, "g": 2.3, "x": 1, "y": 1, "x_reset" : 0.74,  "y_reset" : 0.69}, # 2.3 "g": 4.36,
#     "m_current": {"E": -80, "g": 0.4, "x": 1, "y": None, "x_reset" : 0.18, "y_reset" : None }, # 0.4 "g": 0.76,
#     "ahp": {"E": -70, "g": 0.32, "x": 1, "y": None, "x_reset" : 0.018, "y_reset" : None}, # 0.32 "g": 0.6,
#     "hcn" : { "E": -17, "g": 0.003, "x": None, "y": 1, "x_reset" : None, "y_reset" : 0.002 }, # 0.003
# }
#
# dt = 0.1
# duration = 500
# cbrd = lib.BorgGrahamNeuron(neuron_params)
#
# animator = Animator(cbrd, [0, 200, 0, duration, 0, duration], [0, 0.4, 0, 1000, 0, 5.2])
# animator.run(dt, duration, 0)






