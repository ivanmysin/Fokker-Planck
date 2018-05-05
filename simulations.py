
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import CBRDlib as lib

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


    def update_on_plot(self, idx):


        x1, y1, x2, y2, y3 = self.model.update(self.dt)

        # y3 = self.model.get_artificial_signals()

        self.line1.set_data(x1, y1)
        self.time_text.set_text("simulation time is %.2f in ms" % idx)



        self.line2.set_data(x2, y2)

        if len(y3) > 0:
            # print(len(x2), len(y3) )
            self.line3.set_data(x2, y3)

        return [self.line1, self.time_text, self.line2, self.line3]


    def run(self, dt, duration, interval=10):

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
        "refactory": 3.0,
        "ref_dvdt" : 3.0,

        "N" : 400,
        "dts" : 0.5,
        "use_CBRD" : sim_params["use_CBRD"],

        "w_in_distr" : 1.0,
    }

    if not sim_params["use_CBRD"]:
        neuron_params["N"] = 10000



    sine_generator_params = {
        "fr" : sim_params["fsine"],
        "phase" : 0,
        "amp_max" : 200.0,
        "amp_min" : 0,
    }

    poisson_generator_params = {
        "fr" : sim_params["fpoisson"],
        "w" : 200.0,
        "refactory" : 10,
        "length" : 5,
    }

    synapse_params = {
        "w" : sim_params["ext_sinapse_w"], #0.0003
        "delay" : 1.0,
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

    std_of_Vt = sim_params["std_of_Vt"]
    if std_of_Vt > 0:
        vt_min = neuron_params["Vt"] - 3 * std_of_Vt
        vt_max = neuron_params["Vt"] + 3 * std_of_Vt
        vt_arr = np.linspace(vt_min, vt_max, Nn)
        vt_int = (vt_max - vt_min) / Nn
        p_vt = 1 / (std_of_Vt * np.sqrt(2 * np.pi)) * np.exp(-(neuron_params["Vt"] - vt_arr)**2 / (2 * std_of_Vt** 2))
        p_vt = p_vt * vt_int


    for idx in range(Nn):
        neuron_params_tmp = neuron_params.copy()
        neuron_params_tmp["w_in_distr"] = 1 / Nn

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
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.Synapse(synapse_params_tmp)
            synapses.append(synapse)


    # set synapses from neurons to neurons
    for pre_idx in neurons_idxs:
        for post_idx in neurons_idxs:
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.Synapse(synapse_params_tmp)
            synapses.append(synapse)

    net = lib.Network(neurons, synapses)

    if sim_params["show_animation"]:
        animator = Animator(net, [0, 200, 0, duration, 0, duration], [0, 0.4, 0, 1000, 0, 5.2])
        animator.run(dt, duration, 0)
    else:
        ts_states, sum_Pts, times, flow, artsignals = net.update(dt, duration)

    flow = net.getflow()
    artsignals = net.get_artificial_signals()

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
        "ext_sinapse_w" : 0.001,

        "fpoisson" : 25,
        "fsine"    : 8,

        "duration" : 500,
        "show_animation" : False,
    }

    return default_simulation_params4LIF
########################################################################################################################
def variate_Iext_at_different_std_and_w(Iext, variateVt, path):
    simulation_params = get_default_simulation_params4LIF()

    simulation_params["duration"] = 500
    simulation_params["Iext"] = Iext

    synaptic_w = np.linspace(0.0001, 0.005, 10)
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
            # params_tmp["Iext"] = Iext

            params_tmp["ext_sinapse_w"] = synw

            if variateVt:
                params_tmp["std_of_Vt"] = parm
            else:
                params_tmp["std_of_Iext"] = parm

            flow, _ = run_simulation_LIF(params_tmp)
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

            synch_coeff = np.std(flow[-1000:])

            if synch_coeff >= 3:
                synch_coeff = 1
            else:
                synch_coeff = 0

            synch_coeff_arr[idx1, idx2] = synch_coeff

    fig, ax = plt.subplots()
    cax = ax.pcolor(std_Vt_arr, synaptic_w, synch_coeff_arr, cmap='gray', vmin=0, vmax=1)
    if variateVt:
        ax.set_xlabel("std of Vt")
    else:
        ax.set_xlabel("std of Iext")
    ax.set_ylabel("synaptic w")
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['synchronious', 'asynchronious'])
    fig.savefig(path + "synch_areas")
    plt.show()

def synchronization_regimesLIF(path):
    iext_arr = np.linspace(0.05, 0.3, 5)
    variateVt = True

    for iext in iext_arr:
        path_tmp = path + "variateVt_iext={}".format(iext) + "/"
        if not (os.path.isdir(path_tmp)):
            os.mkdir(path_tmp)

        variate_Iext_at_different_std_and_w(iext, variateVt, path_tmp)


    variateVt = False
    for iext in iext_arr:
        path_tmp = path + "variateIext_iext={}".format(iext) + "/"
        if not (os.path.isdir(path_tmp)):
            os.mkdir(path_tmp)

        variate_Iext_at_different_std_and_w(iext, variateVt, path_tmp)


########################################################################################################################

path ="/home/ivan/Data/CBRD/synchronization_LIF/"

synchronization_regimesLIF(path)




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












# gbarS = 1
# S = 0
# dsdt = 0
# tau = 1
# tau_s = 5.4
#
#
#
#
# dt = 0.1
# dur = 1000
#
# S_hist = []
# time = np.arange(0, dur, dt)
# pre_flow = np.zeros_like(time)  # * 0.2 * 0.5 * (np.cos(2*np.pi*2*time) + 1)
# pre_flow[::100] = 0.5
#
# # pre_flow *= 10
#
# for idx, t in enumerate(time):
#
#     dsdt = dsdt + dt * (tau * gbarS * pre_flow[idx] / tau_s**2 - S / tau_s**2 - 2 * dsdt / tau_s)
#     S = S + dt * dsdt
#
#     S_hist.append(S)
#
# S_hist = np.asarray(S_hist)
#
# S_hist *= 10
#
# plt.plot(time, S_hist)
# plt.show()
