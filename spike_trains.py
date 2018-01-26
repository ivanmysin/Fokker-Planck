
import numpy as np
from scipy.signal import gaussian
from scipy.stats import pearsonr
import itertools
import matplotlib.pyplot as plt
import brian2 as br2
# br2.set_device('cpp_standalone')

def conv_spike_trains(spike_trains, std=50, tmax=None):

    kernel = gaussian(50, std=std)
    # kernel = kernel / kernel.sum()

    convoled = []
    for sp in spike_trains:

        if sp.size < 5: continue

        if tmax is None:
            t = np.max(sp)
        else:
            t = tmax

        sp_tr = np.zeros(int(t))
        sp_tr[ np.round(sp).astype(int) ] = 1



        sp_tr = np.convolve(sp_tr, kernel, mode="same")

        # plt.figure()
        # plt.plot(sp_tr)
        # plt.plot(sp_tr2)
        # plt.show(block=False)

        convoled.append(sp_tr)



    return convoled


def get_correlations(spike_trains, p_of_pair=1):

    r_array = []
    p_array = []

    if ( len(spike_trains) == 1):
        spike_trains = spike_trains[0]
        pairs = itertools.combinations(spike_trains, 2)

    if ( len(spike_trains) == 2):
        pairs = itertools.product(spike_trains[0], spike_trains[1])



    for pair in pairs:
        if ( p_of_pair < np.random.rand() ): continue
        R, p = pearsonr(pair[0], pair[1])
        r_array.append(R)
        p_array.append(p)


    return np.asarray(r_array), np.asarray(p_array)


def model_network(Ne, Ni, params, duration=2000):
    rand = np.random.rand
    randn = np.random.randn

    rand_params = params["rand_params"]
    Iteta = params["Iteta"]
    Iextmean = params["Iextmean"]


    if rand_params:
        re = rand(Ne)
        ri = rand(Ni)
    else:
        re = np.zeros(Ne)
        ri = np.ones(Ni)

    a = np.append(0.02 * np.ones(Ne), 0.02 + 0.08 * ri)
    b = np.append(0.2 * np.ones(Ne), 0.25 - 0.05 * ri)
    c = np.append(-65 + 15 * re ** 2, -65 * np.ones(Ni))
    d = np.append(8 - 6 * re ** 2, 2 * np.ones(Ni))

    S = np.append(0.5 * rand(Ne + Ni, Ne), -rand(Ne + Ni, Ni), axis=1)

    S = S * ( (Ne + Ni) / 1000 )

    v = -65 + 2 * randn(Ne + Ni)
    u = b * v

    firings = np.empty((2, 0), dtype=float)

    neuron_indexes = np.arange(v.size)

    if Iextmean > 0:
        Iext_mean = Iextmean * randn(Ne + Ni) # rand(Ne + Ni) #

    for t in range(1, duration):
        # simulation of 1000 ms



        I = np.append(params["Enoise"] * randn(Ne), params["Inoise"] * randn(Ni))  # thalamic input  # + I_teta #

        if (Iextmean > 0):
            I += Iext_mean

        if (Iteta != 0):
            I += Iteta * (np.cos(2 * np.pi * t * 0.001 * 8) + 1) * 0.5

            I[Ne:] += -Iteta * (np.cos(2 * np.pi * t * 0.001 * 8 + 1.2) + 1) * 0.5

            I[:Ne] += 0.3 * Iteta * (np.cos(2 * np.pi * t * 0.001 * 8 + np.pi) + 1) * 0.5

        fired = v >= 30
        # indices of spikes
        if (np.sum(fired) > 0):
            firings = np.append(firings, np.array([np.repeat(t, np.sum(fired)), neuron_indexes[fired]]), axis=1)

        v[fired] = c[fired]
        u[fired] += d[fired]
        I = I + np.sum(S[:, fired], axis=1)
        v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)  # step 0.5 ms
        v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)  # % for numerical stability
        u = u + a * (b * v - u)

    return firings

def LIF_network(Ne, Ni, params, duration=2000):


    N = Ne + Ni
    Vr = 0 * br2.mV
    theta = 20 * br2.mV
    tau = 20 * br2.ms
    # delta = 2 * br2.ms
    taurefr = 5 * br2.ms
    duration = duration * br2.ms
    C = 1000
    sparseness = float(C) / N
    J = 0.02 * br2.mV
    # muext = 25*mV
    # sigmaext = 5 * br2.mV

    eqs = """
    dV/dt = (-V + muext + sigmaext * sqrt(0.1 * ms) * xi)/tau : volt 
    muext : volt
    sigmaext : volt
    """
    # tau
    group = br2.NeuronGroup(N, eqs, threshold='V>theta',
                        reset='V=Vr', refractory=taurefr, method='euler', dt=0.1*br2.ms)
    group.V = "Vr + 5 * randn() * mV"
    Ext = group[:Ne]
    Inh = group[Ne:]

    Ext.sigmaext = params["Enoise"] * br2.mV
    Inh.sigmaext = params["Inoise"]  * br2.mV

    if params["Iextmean"] > 0:
        group.muext = "( 20 + %.4f * randn() )*mV" % params["Iextmean"]
    else:
        group.muext = "20 * mV"



    conn_Ext2all = br2.Synapses(Ext, group, on_pre='V += J * rand()')  # , delay=delta
    conn_Ext2all.connect(p=sparseness)

    conn_Inh2All = br2.Synapses(Inh, group, on_pre='V -= J * rand()')  # , delay=delta
    conn_Inh2All.connect(p=sparseness)

    # mon = StateMonitor(group, 'V', record=0)

    M = br2.SpikeMonitor(group)

    br2.run(duration)



    firings = np.empty((2, 0), dtype=float)
    spike_trains = []
    for idx in range(N):
        sp = M.values("t")[idx] / br2.ms

        if (np.sum(sp) > 0):
            firings = np.append(firings, np.array( [sp, np.repeat(idx, sp.size)] ), axis=1)
            spike_trains.append(sp)

    return firings, spike_trains

# LIF_network()