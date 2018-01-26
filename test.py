
import numpy as np
import matplotlib.pyplot as plt



theta_phase = np.linspace(-np.pi, np.pi, 200)
theta_wave = 0.5 * (np.cos(theta_phase) + 1)

gamma_phase = theta_phase * 8

gamma_wave = 0.2 * (0.5 * (np.cos(gamma_phase) + 1)) * theta_wave

gamma_wave = np.roll(gamma_wave, 70)

plt.plot(theta_phase, theta_wave)

plt.plot(theta_phase, gamma_wave)

plt.show()