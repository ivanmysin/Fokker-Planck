import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import FuncAnimation
from scipy.special import erf
from scipy.stats import pearsonr
from scipy.signal import argrelextrema, parzen
# import HH_models as hh

import CBRDlib as lib


############################################

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






#############################################

def run_simulation(Nn=10, std_of_Vt=0.0, std_of_Iext=0.0, iext=14, ntimesinputs=0, nspatialinputs=0):

    # if std_of_Vt == 0 and std_of_Iext == 0:
    #     Nn = 1

    neuron_params = {
        "Vreset" : -10, # -60,
        "Vt" : 20, # -50,
        "gl" : 0.01,  # 0.02,
        "El" : 0, # -60
        "C" : 0.2, # 0.2,
        "Iext" : iext, # 0.14, # 0.15,
        "sigma" : 3.0,
        "refactory": 5.0,
        "tau_t" : 0,

        "Nro" : 400,
        "dts" : 0.5,
        "use_CBRD" : True,
        "N" : 4000,

        "w_in_distr" : 1.0,
    }

    sine_generator_params = {
        "fr" : 2,
        "phase" : 0,
        "amp_max" : 0.25,
        "amp_min" : 0,
    }

    poisson_generator_params = {
        "fr" : 25, # 3,
        "w" : 0.5,
        "refactory" : 10,
        "length" : 5,
    }

    synapse_params = {
        "w" : 10.0,  # 2000, # 20.0,
        "delay" : 20.0,
        "pre" : 0,
        "post" : 0,

        "tau_s" : 5.4,
        "tau" : 1,
        "gbarS" : 1.0,
        "Erev" : 50,
    }

    # if Nn > 1:
    #     synapse_params["w"] /= Nn #**2

    dt = 0.1
    duration = 1000

    neurons = []
    synapses = []


    for _ in range(ntimesinputs):
        neuron = SineGenerator(sine_generator_params)
        neurons.append(neuron)

    for _ in range(nspatialinputs):
        neuron = PoissonGenerator(poisson_generator_params)
        neurons.append(neuron)


    if std_of_Iext > 0:
        iext_min = neuron_params["Iext"] - 3 * std_of_Iext
        iext_max = neuron_params["Iext"] + 3 * std_of_Iext
        iext_arr = np.linspace(iext_min, iext_max, Nn)
        i_ext_int = (iext_max - iext_min) / Nn
        p_ext = 1 / (std_of_Iext * np.sqrt(2 * np.pi)) * np.exp(-(neuron_params["Iext"] - iext_arr) ** 2 / (2 * std_of_Iext** 2))
        p_ext = p_ext * i_ext_int

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
        neurons.append(neuron)


    for pre_idx in range(Nn): # range(Nn)
        for post_idx in range(Nn):
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.Synapse(synapse_params_tmp)
            synapses.append(synapse)

    net = lib.Network(neurons, synapses)
    animator = Animator(net, [0, 200, 0, duration, 0, duration], [0, 0.4, 0, 1000, 0, 5.2])
    animator.run(dt, duration, 0)


    # flow = net.getflow()
    # ts_states, sum_Pts, times, flow, generators_signal = net.update(dt, duration)
    # return flow, generators_signal

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


    std_vt_arr = np.linspace(0, 12, 2) # 24
    flow_time_arr = []
    corr_arr = []
    for stdvt in std_vt_arr:
        flow_time, generators_signal = run_simulation(std_of_Vt=stdvt, iext=iext, nspatialinputs=0, ntimesinputs=1)
        t = np.linspace(0, 900, flow_time.size)
        generators_signal = generators_signal[0]

        flow_time_arr.append(flow_time)
        R, p = pearsonr(flow_time, generators_signal)
        corr_arr.append(R)


        fig, axs = plt.subplots(nrows=1, ncols=1)
        sine = generators_signal * np.std(flow_time) + np.mean(flow_time)
        axs.plot(t, sine, color="blue")
        axs.plot(t, flow_time, color="red" )
        plt.show(block=False)


    flow_time_arr = np.asarray(flow_time_arr)
    np.save("time_input", flow_time_arr)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(std_vt_arr, corr_arr)
    axs.set_title("I/O correlation")
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

def run_HH(Nn=10, std_of_iext=0.2, iext=0.5, ntimesinputs=0, nspatialinputs=0, fp=10, fs=8, path=None):


    if (std_of_iext == 0):
        Nn = 1

    neuron_params = {
        "C" : 0.7, # mkF / cm^2
        "Vreset" : -40,
        "Vt" : -55,
        "Iext" : iext, # 0.3, # nA / cm^2
        "saveV": False,
        "ref_dvdt" : 1.0,
        "refactory" : 5.5,
        "sigma" : 0.3,

        "use_CBRD": True,

        "w_in_distr" : 1.0,
        "saveCV" : True,

        "N": 400,
        "dts": 0.5,

        "gl" : 0.025,
        "El" : -61.22,

        "leak"  : {"E" : -61.22, "g" : 0.025 },
        "dr_current" : {"E" : -70, "g" : 0.76, "x" : 1, "y" : 1, "x_reset" : 0.26, "y_reset" : 0.47},  # "g" : 0.76
        "a_current": {"E": -70, "g": 2.3, "x": 1, "y": 1, "x_reset" : 0.74,  "y_reset" : 0.69}, # 2.3 "g": 4.36,
        "m_current": {"E": -80, "g": 0.4, "x": 1, "y": None, "x_reset" : 0.18, "y_reset" : None }, # 0.4 "g": 0.76,
        "ahp": {"E": -70, "g": 0.32, "x": 1, "y": None, "x_reset" : 0.018, "y_reset" : None}, # 0.32 "g": 0.6,
        "hcn" : { "E": -17, "g": 0.003, "x": None, "y": 1, "x_reset" : None, "y_reset" : 0.002 }, # 0.003
    }


    sine_generator_params = {
        "fr" : fs,
        "phase" : 0,
        "amp_max" : 0.5,
        "amp_min" : 0,
    }

    poisson_generator_params = {
        "fr" : fp,
        "w" : 0.5,
        "refactory" : 10,
        "length" : 10,
    }

    simplesynapse_params = {
        "w" : 1.0,
        "delay" : 0,
        "pre" : 0,
        "post" : 0,
    }

    synapse_params = {
        "w": 0.001,
        "delay": 1.0,
        "pre": 0,
        "post": 0,

        "tau_s": 5.4,
        "tau_a": 1.0,
        "gbarS": 1.0,
        "Erev": 0,
    }

    dt = 0.1
    duration = 1500

    neurons = []
    synapses = []
    # synapse_params["w"] /= Nn**2

    for _ in range(ntimesinputs):
        neuron = lib.SineGenerator(sine_generator_params)
        neurons.append(neuron)

    for _ in range(nspatialinputs):
        neuron = lib.PoissonGenerator(poisson_generator_params)
        neurons.append(neuron)

    if std_of_iext != 0:
        iext_min = neuron_params["Iext"] - 3 * std_of_iext
        iext_max = neuron_params["Iext"] + 3 * std_of_iext
        iext_arr = np.linspace( iext_min, iext_max, Nn )
        i_ext_int = (iext_max - iext_min) / Nn
        p = 1 / (std_of_iext * np.sqrt(2 * np.pi) ) * np.exp( -(neuron_params["Iext"] - iext_arr)**2 / (2 * std_of_iext**2 ) )
        p = p * i_ext_int


    for idx in range(Nn):
        neuron_params_tmp = neuron_params.copy()

        if Nn > 1:
            neuron_params_tmp["Iext"] = iext_arr[idx]
            neuron_params_tmp["w_in_distr"] = p[idx] # 1.0 / Nn #

        neuron = lib.BorgGrahamNeuron(neuron_params_tmp)
        neurons.append(neuron)




    for pre_idx in range(ntimesinputs):
        for post_idx in range(ntimesinputs+nspatialinputs, Nn+ntimesinputs+nspatialinputs):
            synapse_params_tmp = simplesynapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.SimlestSinapse(synapse_params_tmp)
            synapses.append(synapse)

    for pre_idx in range(ntimesinputs):
        for post_idx in range(ntimesinputs + nspatialinputs, Nn + ntimesinputs + nspatialinputs):
            synapse_params_tmp = simplesynapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.SimlestSinapse(synapse_params_tmp)
            synapses.append(synapse)
    # set all to all connections !!!!!
    # w_sum = 0
    for pre_idx in range(ntimesinputs, Nn+ntimesinputs+nspatialinputs):
        for post_idx in range(ntimesinputs+nspatialinputs, Nn+ntimesinputs+nspatialinputs):
            synapse_params_tmp = synapse_params.copy()
            synapse_params_tmp["pre"] = neurons[pre_idx]
            synapse_params_tmp["post"] = neurons[post_idx]
            synapse = lib.Synapse(synapse_params_tmp)
            synapses.append(synapse)


    #w_sum += synapse_params_tmp["w"] / Nn
    # print(w_sum)
    # assert (False)

    net = lib.Network(neurons, synapses)
    # animator = Animator(net, [0, 200, 0, duration, 0, duration], [0, 1, 0, 100, 0, 0.7], path=path)
    # animator.run(dt, duration, 0)

    ts_states, sum_Pts, times, flow, artsignals = net.update(dt, duration)

    flow = net.getflow()
    cv = net.getCV()
    generators_signal = net.get_artificial_signals()


    return flow, generators_signal, cv
    # flow = net.getflow()

    # ts_states, sum_Pts, times, flow, generators_signal = net.update(dt, duration)
    # return flow, generators_signal



def synchronization_regimes_HH():
    std_of_iext_arr = np.linspace(0, 3, 5) # 10
    flow_arr = []

    synchs_arr = []

    for iext_std in std_of_iext_arr:

        flow, _ = run_HH(Nn=15, std_of_iext=iext_std, iext=0.8)
        flow_arr.append(flow)
        synch_coeff = np.std(flow[-1000:])

        synchs_arr.append(synch_coeff)



    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(std_of_iext_arr, synchs_arr)

    plt.show()


def answer_2_spacial_inputs_HH(path):
    iscompute = False
    std_iext_arr = np.linspace(0, 1, 5)  # 10
    iext = 0.2
    fp_arr = np.linspace(2, 15, 10) # np.linspace(50, 70, 3) #
    ppv_arr = []
    max_ppv_arr = []

    win = parzen(15)
    win /= np.sum(win)

    fig, axs = plt.subplots(nrows=1, ncols=2)

    for fp in fp_arr:
        for stext in std_iext_arr:
            path_tmp = path + '{:.2f}'.format(stext) + "_" + '{:.2f}'.format(fp)
            print(path_tmp)
            if iscompute:
                flow_spatial, generators_signal, cv = run_HH(Nn=10, std_of_iext=stext, iext=iext, nspatialinputs=1, fp=fp, path=path_tmp)
                generators_signal = generators_signal[0]
                np.savez(path_tmp, flow=flow_spatial, artsignal=generators_signal, cv=cv)

            else:
                data = np.load(path_tmp + ".npz")
                flow_spatial = data["flow"]
                generators_signal = data["artsignal"]


            flow_spatial = np.convolve(flow_spatial, win, mode="same")

            answs_idx = argrelextrema(flow_spatial, np.greater, order=50)[0]
            input_idx = np.argwhere(np.diff(generators_signal) < 0)
            # answs_idx = answs_idx[flow_spatial[answs_idx] > np.percentile(flow_spatial, 80)]

            ppv = calculate_ppv(input_idx, answs_idx)
            ppv_arr.append(ppv)

            time = np.linspace(0, 1500, flow_spatial.size)
            fig_flow, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
            ax[0].plot(time, flow_spatial)
            ax[1].plot(time, generators_signal)
            fig_flow.savefig(path_tmp + ".png")
            plt.close(fig_flow)




        ppv_arr = np.asarray(ppv_arr)
        max_ppv_arr.append(np.max(ppv_arr))
        axs[0].plot(std_iext_arr, ppv_arr, label='{:.2f}'.format(fp))

        ppv_arr = []

    axs[0].set_ylabel("PPV")
    axs[0].set_xlabel(r"$\sigma_{I_{ext}}$")
    axs[0].legend()


    axs[1].plot(fp_arr, max_ppv_arr)
    axs[1].set_ylabel("PPV*")
    axs[1].set_xlabel("Частота стимулов")
    fig.tight_layout()


    path_tmp = path + "ppv_from_iext_std"
    fig.savefig(path_tmp)
    plt.close(fig)


def answer_2_time_inputs_HH(path):
    iscompute = False

    std_iext_arr = np.linspace(0, 1, 5)  # 10
    iext = 0.2
    fs_arr = np.logspace(0.1, 2, 10, endpoint=False)

    corr_arr = []
    max_corr_fs_arr = []


    fig, axs = plt.subplots(nrows=1, ncols=2)

    for fs in fs_arr:
        for stext in std_iext_arr:
            path_tmp = path + '{:.2f}'.format(stext) + "_" + '{:.2f}'.format(fs)
            print(path_tmp)

            if iscompute:
                flow_time, generators_signal, cv = run_HH(Nn=10, std_of_iext=stext, iext=iext, ntimesinputs=1, fs=fs,
                                                        path=path_tmp)

                generators_signal = generators_signal[0]

                np.savez(path_tmp, flow=flow_time, artsignal=generators_signal, cv=cv, std_of_iext=stext, fs=fs)

            else:
                data = np.load(path_tmp + ".npz")
                flow_time = data["flow"]
                generators_signal = data["artsignal"]

            time = np.linspace(0, 1500, flow_time.size)
            fig_flow, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
            ax[0].plot(time, flow_time)
            ax[1].plot(time, generators_signal)
            fig_flow.savefig(path_tmp + ".png")
            plt.close(fig_flow)


            R, p = pearsonr(flow_time, generators_signal)
            corr_arr.append(R)





        corr_arr = np.asarray(corr_arr)
        max_corr_fs_arr.append(std_iext_arr[ np.argmax(corr_arr) ] )


        axs[0].plot(std_iext_arr, corr_arr, label='{:.2f}'.format(fs))

        corr_arr = []

    axs[0].set_ylabel("I/O корреляция")
    axs[0].set_xlabel(r"$\sigma_{I_{ext}}$")
    axs[0].legend()

    axs[1].plot(fs_arr, max_corr_fs_arr)
    axs[1].set_ylabel(r"$\sigma_{I_{ext}}$ *")
    axs[1].set_xlabel("частота синусойды")
    fig.tight_layout()

    path_tmp = path + "IO_corralation_from_iext_std"
    fig.savefig(path_tmp)
    plt.close(fig)




def main():
    # run_simulation(Nn=10, std_of_Vt=5.0, std_of_Iext=0.0, iext=18)
    # Nn=10, std_of_Vt=0.0, std_of_Iext=0.0, iext=15, ntimesinputs=0, nspatialinputs=0)


    # synchronization_regimes()
    # answer_2_inputs()

    synchronization_regimes_HH()
    # path = "/home/ivan/Data/CBRD/poisson_input/"
    # answer_2_spacial_inputs_HH(path)
    #
    # path = "/home/ivan/Data/CBRD/sine_input/"
    # answer_2_time_inputs_HH(path)

    # run_HH(Nn=10, std_of_iext=0.8, iext=0.5, ntimesinputs=0, nspatialinputs=0, path=path+"1.mp4")




if __name__ == "__main__":
    main()


