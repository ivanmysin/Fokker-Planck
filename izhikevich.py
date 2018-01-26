import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import matplotlib.pyplot as plt
import spike_trains as sptr

plt.rc('axes', linewidth=2)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('lines', linewidth=3)
plt.rc('lines', markersize=4)
plt.rc('lines', c="black")


rand = np.random.rand
randn = np.random.randn

Ne = 800
Ni = 200

duration = 5000


params_array = [
    {
        "rand_params" : False,
        "Iteta" : 0,
        "Iextmean" : 0,
        "Enoise" : 1,
        "Inoise" : 1,
    },

    {
        "rand_params": False,
        "Iteta": 0,
        "Iextmean": 0,
        "Enoise" : 2,
        "Inoise" : 2,
    },

    {
        "rand_params": False,
        "Iteta": 0,
        "Iextmean": 0,
        "Enoise": 3,
        "Inoise": 3,
    },

    {
        "rand_params": False,
        "Iteta": 0,
        "Iextmean": 0,
        "Enoise": 4,
        "Inoise": 4,
    },

]

#
# params_array = [
#     {
#         "rand_params" : False,
#         "Iteta" : 0,
#         "Iextmean" : 0,
#         "Enoise" : 1,
#         "Inoise" : 1,
#     },
#
#     {
#         "rand_params": False,
#         "Iteta": 0,
#         "Iextmean": 1,
#         "Enoise" : 1,
#         "Inoise" : 1,
#     },
#
#     {
#         "rand_params": False,
#         "Iteta": 0,
#         "Iextmean": 2,
#         "Enoise": 1,
#         "Inoise": 1,
#     },
#
#     {
#         "rand_params": False,
#         "Iteta": 0,
#         "Iextmean": 3,
#         "Enoise": 1,
#         "Inoise": 1,
#     },
#
# ]



fig1, axarr1 = plt.subplots(ncols=1, nrows=len(params_array), sharex=True)
fig2, axarr2 = plt.subplots(ncols=1, nrows=len(params_array), sharex=True)

t_start_processing = 200

for idx, params in enumerate(params_array):
    firings, _ = sptr.LIF_network(Ne, Ni, params, duration=duration)      # model_network(Ne, Ni, params, duration=duration)

    spike_trains = []


    for idx2 in range(Ne + Ni):
        sp = firings[0, firings[1, :] == idx2]
        sp = sp[sp >= t_start_processing]
        spike_trains.append(sp)

    pyr_convoled = sptr.conv_spike_trains(spike_trains[:Ne], 1, duration+1)
    int_convoled = sptr.conv_spike_trains(spike_trains[Ne:], 1, duration+1)

    r_pyr_pyr, _ = sptr.get_correlations([pyr_convoled], p_of_pair = 1)
    r_int_int, _ = sptr.get_correlations([int_convoled], p_of_pair = 1)
    r_pyr_int, _ = sptr.get_correlations([pyr_convoled, int_convoled], p_of_pair = 1)



    axarr1[idx].scatter( firings[0, :], firings[1, :], s=1, color="k")
    axarr1[idx].set_xlim(0, 500)
    axarr1[idx].set_ylim(0, Ne+Ni)


    axarr2[idx].hist(r_pyr_pyr, bins=100, histtype='step', weights=np.ones(r_pyr_pyr.size)/r_pyr_pyr.size, density=False, label="E - E", linestyle='solid', color="k")
    axarr2[idx].hist(r_int_int, bins=100, histtype='step', weights=np.ones(r_int_int.size)/r_int_int.size, density=False, label="I - I", linestyle='dashed', color="k")
    axarr2[idx].hist(r_pyr_int, bins=100, histtype='step', weights=np.ones(r_pyr_int.size)/r_pyr_int.size, density=False, label="E - I", linestyle='-.', color="k")
    axarr2[idx].set_ylim(0, 0.2)
    axarr2[idx].set_xlim(-0.5, 1)
    axarr2[idx].legend()


fig1.show()
fig2.show()



plt.show(block=True)

