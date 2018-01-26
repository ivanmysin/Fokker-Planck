from brian2 import *

N = 1000
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 5*ms
duration = 1 * second
C = 1000
sparseness = float(C)/N
J = 0.01 * mV
# muext = 25*mV
sigmaext = 1*mV

eqs = """
dV/dt = (-V + muext + sigmaext * sqrt(tau) * xi)/tau : volt
muext : volt
"""

group = NeuronGroup(N, eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr, method='euler', dt=0.1*ms)
group.V = "Vr + 5 * randn() * mV"
group.muext =  "(20 + randn()*5)*mV" # "20 * mV" #

Ext = group[:800]
Inh = group[800:]

conn_Ext2all = Synapses(Ext, group, on_pre='V += J * rand()') # , delay=delta
conn_Ext2all.connect(p=sparseness)

conn_Inh2All = Synapses(Inh, group, on_pre='V -= J * rand()') # , delay=delta
conn_Inh2All.connect(p=sparseness)

# mon = StateMonitor(group, 'V', record=0)

M = SpikeMonitor(group)
LFP = PopulationRateMonitor(group)

run(duration)

#plot(mon.t/ms, mon.V[0]/mV, 'k')

subplot(211)
plot(M.t/ms, M.i, '.')
xlim(0, duration/ms)

subplot(212)
plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(0, duration/ms)

show()