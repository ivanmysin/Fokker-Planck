from brian2 import *
import numpy as np
n = 1000
duration = 0.2*second

area = 20000*umetre**2
Cm = 1*ufarad * cm**-2 * area
gl = 0.1 * msiemens * cm**-2 * area
El = -60*mV
#Iext =  0.5 * namp # "(rand() + 0.5) * namp" #
sigma = 0.5 * namp

Ee = 0 * mV
taue = 20 * ms
we = 0.0000006 * nS  # excitatory synaptic weight

eqs = '''
dv/dt = ( gl*(El-v) + ge*(Ee-v) + Iext + sigma * xi * msecond**0.5) / Cm: volt (unless refractory)
dge/dt = -ge*(1./taue) : siemens
Iext : amp
'''

group = NeuronGroup(n, eqs, threshold='v > -50*mV', reset='v = -60*mV',
                    refractory=5*ms, method="milstein")

group.v = "(-60 + 10*randn() ) * mV"
group.ge = 0 * msiemens * cm**-2 * area
group.Iext = "(0.0 + 0.0) * namp" # "(randn()*1.5 + 0.0) * namp"

Ce = Synapses(group, group, on_pre='ge+=we')
Ce.connect(p=0.1)

monitor = StateMonitor(group, 'v', record=0)
spikemonitor = SpikeMonitor(group)
raster = PopulationRateMonitor(group)

run(duration)

subplot(211)
plot(spikemonitor.t/ms, spikemonitor.i, '.')
xlim(0, duration/ms)

subplot(212)
plot(raster.t/ms, raster.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(0, duration/ms)

show()

