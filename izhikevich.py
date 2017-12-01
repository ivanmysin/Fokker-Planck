import numpy as np
import matplotlib.pyplot as plt
import spike_trains as sptr



rand = np.random.rand
randn = np.random.randn

Ne = 800
Ni = 200

duration=5000


params_array = [
    {
        "rand_params" : False,
        "Iteta" : 0,
        "Iextmean" : 0,
        "Enoise" : 4.5,
        "Inoise" : 3.5,
    },

    {
        "rand_params": False,
        "Iteta": 0,
        "Iextmean": 2,
        "Enoise" : 4.5,
        "Inoise" : 3.5,
    },

    {
        "rand_params": False,
        "Iteta": 0.5,
        "Iextmean": 0,
        "Enoise": 4.5,
        "Inoise": 3.5,
    },

]

fig1, axarr1 = plt.subplots(ncols=1, nrows=len(params_array), sharex=True)
fig2, axarr2 = plt.subplots(ncols=1, nrows=len(params_array), sharex=True)


for idx, params in enumerate(params_array):
    firings = sptr.model_network(Ne, Ni, params, duration=duration)

    spike_trains = []
    t_start_processing = 500

    for idx2 in range(Ne + Ni):
        sp = firings[0, firings[1, :] == idx2]
        sp = sp[sp >= t_start_processing]
        spike_trains.append(sp)

    pyr_convoled = sptr.conv_spike_trains(spike_trains[:Ne], 10, duration)
    int_convoled = sptr.conv_spike_trains(spike_trains[Ne:], 10, duration)


    r_pyr_pyr, _ = sptr.get_correlations([pyr_convoled], p_of_pair = 1)
    r_int_int, _ = sptr.get_correlations([int_convoled], p_of_pair = 1)
    r_pyr_int, _ = sptr.get_correlations([pyr_convoled, int_convoled], p_of_pair = 1)



    axarr1[idx].scatter( firings[0, :], firings[1, :], s=2 )
    axarr1[idx].set_xlim(0, duration)




    axarr2[idx].hist(r_pyr_pyr, bins=100, histtype='step', normed=True, label="Pyr - Pyr")
    axarr2[idx].hist(r_int_int, bins=100, histtype='step', normed=True, label="Int - Int")
    axarr2[idx].hist(r_pyr_int, bins=100, histtype='step', normed=True, label="Pyr - Int")
    axarr2[idx].legend()


fig1.show()
fig2.show()
plt.show(block=True)

