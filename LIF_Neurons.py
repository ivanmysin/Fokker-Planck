# aproximation of LIF neurons

import numpy as np
import matplotlib.pyplot as plt


n_states = 200
V_min = -80
V_max = -50
sigma = 1
Iext = 0.8
gl = 0.1
tau_m = 20
V_reset = -50
dt = 0.01
duraction = 100
delta_V = (V_max - V_min) / n_states
V_range = np.linspace(V_min, V_max, n_states)
V_states = np.zeros(n_states) # - 70 # np.random.rand(n_states) * (V_max - V_min) + V_min
V_states[np.argmin((V_range + 70)**2 )] = 1

V_states = V_states / V_states.sum()

k = sigma**2 / 2 / tau_m

fig, ax = plt.subplots()
line, = ax.plot(V_range, V_states)
plt.show(block=False)

t = 0
while(t <= duraction):

    nu = sigma ** 2 / 2 * (V_states[-1] - V_states[-2]) / delta_V / tau_m

    for idx, V_st in enumerate(V_states):

        a = (-V_st + Iext / gl) / tau_m

        if idx == 0:
            Vst_i_minus_1 = 0
        else:
            Vst_i_minus_1 = V_states[idx - 1]

        if idx == n_states - 1:
            Vst_i_plus_1 = 0
        else:
            Vst_i_plus_1 = V_states[idx + 1]



        V_st = V_st + dt * k * (Vst_i_plus_1 - 2*V_st + Vst_i_minus_1) / delta_V**2  - dt * a * (V_st - Vst_i_minus_1) / delta_V # + dt *

        V_states[idx] = V_st

    V_states[0] = V_states[0] + dt * nu / delta_V

    line.set_data(V_range, V_states)

    t += dt

# plt.show(block=False)
# plt.figure()
# plt.plot(V_range, V_states)
# plt.plot([V_min, V_max], [0, 0])
plt.show(block=True)
print ("End")