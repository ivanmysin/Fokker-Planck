
from libc.math cimport exp, cos
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference, preincrement
import numpy as np
cimport numpy as np
from libcpp.queue cimport queue

from scipy.special import erf





cdef class CBRD:

    cdef np.ndarray t_states, ro, V, dVdt

    cdef int Nro
    cdef double dts, ro_H_integral, ts, sqrt_2, Vt, sigma
    
    

    def __cinit__(self, params):
        self.Nro = params["Nro"]
        self.dts = params["dts"]

        self.t_states = np.linspace(0, self.Nro*self.dts, self.Nro)
        self.ro = np.zeros_like(self.t_states)

        self.ro[-1] = 1 / self.dts

        self.ro_H_integral = 0

        self.ts = 0
        
        self.sqrt_2 = np.sqrt(2)
        
        


    cdef H_function(self, np.ndarray[np.double_t, ndim=1] V, np.ndarray[np.double_t, ndim=1] dVdt, double tau_m, double Vt, double sigma):
        cdef np.ndarray T = (Vt - V) / sigma / self.sqrt_2

        cdef np.ndarray A = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        cdef np.ndarray dT_dt = -1.0 / sigma / self.sqrt_2 * dVdt
        
        
        dT_dt[dT_dt > 0] = 0
        cdef np.ndarray F_T = np.sqrt(2 / np.pi) * np.exp(-T**2) / (1.000000001 + erf(T))
        cdef np.ndarray B = -self.sqrt_2 * dT_dt * F_T * tau_m

        cdef np.ndarray H = (A + B) / tau_m
        return H


    cdef update_ro(self, double dt, np.ndarray[np.double_t, ndim=1] dVdt, double tau_m):
        cdef bool shift = False

        if self.ts >= self.dts:
            self.ro[-1] += self.ro[-2]
            self.ro[:-1] = np.roll(self.ro[:-1], 1)
            self.ro[0] = self.ro_H_integral
            self.ro_H_integral = 0
            self.ts = 0
            shift = True

            # self.ro /= np.sum(self.ro) * self.dts
            # print (np.sum(self.ro) * self.dts)


        cdef np.ndarray H = self.H_function(self.V, dVdt, tau_m, self.Vt, self.sigma)

        cdef np.ndarray dro = self.ro * (1 - np.exp(-H * dt))  # dt * self.ro * H  #

        self.ro -= dro


        self.ro[self.ro < 0] = 0

        self.ro_H_integral += np.sum(dro)



        self.ts += dt

        return shift


cdef class Neuron(CBRD):
    
    cdef double Vreset

    def __cinit__(self, params):
        pass
    
    cdef add_Isyn(self, np.ndarray Isyn):
        pass


    def update(self, double dt):
        return 0, 0, 0, 0

    cdef reset(self):
        self.V = self.Vreset

    cdef get_flow(self):
        return 0
        
########################################################################        
        
cdef class LIF_Neuron(Neuron):
    
    
    cdef np.ndarray firing, times, tsmc, Isyn
    cdef double gl, El, C, refactory, w_in_distr, Iext, g_tot, tau_m
    cdef double  t
    cdef int ref_idx
    cdef bool is_use_CBRD, artifitial_generator
                           

    def __cinit__(self, params):

        self.Vt = params["Vt"]
        self.gl = params["gl"]
        self.El = params["El"]
        self.C = params["C"]
        self.sigma = params["sigma"]
        self.Vreset = params["Vreset"]


        self.refactory = params["refactory"]

        self.w_in_distr = params["w_in_distr"]

        self.Iext = params["Iext"]
        self.g_tot = self.gl

        self.is_use_CBRD =  params["use_CBRD"]

        if not self.is_use_CBRD :
            self.V = np.zeros(params["N"]) + self.Vreset
            self.tsmc = np.zeros_like(self.V) + 200

        else:
            # CBRD.__cinit__(self, params["Nro"], params["dts"])
            self.V = np.zeros(self.Nro) + self.Vreset
            self.ref_idx = int (self.refactory // self.dts)
            self.sigma = self.sigma / np.sqrt(2)

            # if not params["tau_t"] is None:
            #    self.Vt -= 1 + np.exp(-params["tau_t"] / (self.t_states + 0.000001) )

        self.tau_m = self.C / self.g_tot
        self.Isyn = np.zeros_like(self.V)
        self.t = 0

        self.firing = np.zeros(1, dtype=np.double)
        self.times = np.zeros(1, dtype=np.double)

        self.artifitial_generator = False

    cdef add_Isyn(self, np.ndarray Isyn):
        self.Isyn += Isyn

    cpdef update(self, double dt, double duration=0):

        if (duration == 0):
            duration = dt

        cdef double t = 0
        cdef np.ndarray [bool, ndim=1] spiking
        cdef bool shift 
        
        while(t < duration):

            dVdt = -self.V/self.tau_m + self.Iext/self.tau_m + self.Isyn
            # cdef np.ndarray 
            # (self.gl * (self.El - self.V) + self.Iext + self.Isyn) / self.C

            if not self.is_use_CBRD:
                dVdt += np.random.normal(0, self.sigma, self.V.size) / np.sqrt(dt)
                dVdt[self.ts < self.refactory] = 0
            else:
                dVdt[:self.ref_idx] = 0

            self.V += dt * dVdt
            self.Isyn = np.zeros_like(self.V)


            if not self.is_use_CBRD :
                spiking = self.V >= self.Vt 
                self.V[spiking] = self.Vreset
                self.firing = np.append(self.firing, 1000 * np.mean(spiking) / dt )
                self.tsmc += dt
                self.tsmc[spiking] = 0

            else:
                shift = self.update_ro(dt, dVdt, self.tau_m)
                self.firing = np.append(self.firing , 1000 * self.ro[0])
                if shift:
                    self.V[:-1] = np.roll(self.V[:-1], 1)
                    self.V[0] = self.Vreset



            self.times = np.append(self.times, self.times[-1] + dt)

            t += dt

            
        if self.is_use_CBRD:
            return self.t_states, self.w_in_distr * self.ro, self.times, self.w_in_distr * self.firing #, self.t_states, self.V
        else:
            return np.zeros(400), np.zeros(400), self.times, self.firing # , [], []


    cpdef is_artifitial_generator(self):
         return False

    cdef get_flow(self):
        return self.w_in_distr * self.firing[-1] * 0.001

    def get_flow_hist(self):
        return self.w_in_distr * self.firing

    def get_CV(self):
        return self.w_in_distr * self.CVhist

########################################################################

cdef class  SineGenerator:
    
    cdef np.ndarray hist
    cdef double fr, phase, amp_max, amp_min, flow, t
    cdef bool artifitial_generator
    
    def __cinit__(self, params):

        self.fr = params["fr"]
        self.phase = params["phase"]
        self.amp_max = params["amp_max"]
        self.amp_min = params["amp_min"]
        self.flow = 0
        self.t = 0
        self.hist = np.zeros(1, dtype=np.double)

        self.artifitial_generator = True

    cdef update(self, double dt):
        self.flow = 0.5 * (np.cos(2*np.pi*self.fr*self.t + self.phase) + 1) * (self.amp_max - self.amp_min) + self.amp_min
        self.t += 0.001 * dt
        self.hist = np.append(self.hist, self.flow)
        return 0, 0, self.flow, 0

    cdef get_flow(self):
        return self.flow

    cdef add_Isyn(self, Isyn):
        pass

    def get_hist(self):
        return self.hist
    
    cpdef is_artifitial_generator(self):
        return True

cdef class PoissonGenerator:
	
    cdef np.ndarray hist
    cdef double fr, phase, w, refactory, flow, previos_t, length, start
    cdef bool artifitial_generator
    
    def __cinit__(self, params):

        self.fr = params["fr"]
        self.w = params["w"]
        self.refactory = params["refactory"]
        self.length = params["length"]

        self.flow = 0
        self.previos_t = params["refactory"] + 10
        self.start = self.length + 1

        self.hist = np.zeros(1, dtype=np.double)
        self.artifitial_generator = True


    cdef update(self, double dt):

        cdef double r
        if self.flow == self.w:
            self.start += dt
            if self.start >= self.length:
                self.flow = 0
                self.previos_t = 0

        elif self.flow == 0:

            if self.previos_t > self.refactory:
                r = 1000 * np.random.rand() / dt
                if (r < self.fr):
                    self.flow = self.w
                    self.start = 0



        self.previos_t += dt
        self.hist.append(self.flow)
        return 0, 0, self.flow, 0


    def get_hist(self):
        return self.hist

    cdef get_flow(self):
        return self.flow

    cdef add_Isyn(self, Isyn):
        pass
    
    cpdef is_artifitial_generator(self):
        return True
########################################################################
cdef class Synapse:
    

    cdef double w, delay, tau_s, tau, gbarS, S, dsdt, Erev
    cdef Neuron pre, post
    cdef queue [double] pre_hist
    
    
    def __cinit__(self, params):
        self.w = params["w"]
        self.delay = params["delay"]
        self.pre = params["pre"]
        self.post = params["post"]


        self.tau_s = params["tau_s"] # 5.4
        self.tau = params["tau"] # 1
        self.gbarS = params["gbarS"] # 1.0
        self.Erev = params["Erev"] # 50
        
        self.S = 0
        self.dsdt = 0
        
        if self.delay > 0:
             for _ in range(int(self.delay)):
                  self.pre_hist.push(0)

    cpdef update(self, double dt):
            

        cdef double pre_flow = self.pre.get_flow()

        if (self.delay > 0):
            self.pre_hist.push(pre_flow)
            pre_flow = self.pre_hist.front()
            self.pre_hist.pop()
        



        self.dsdt = self.dsdt + dt * (self.tau * self.gbarS * pre_flow / self.tau_s**2 - self.S / self.tau_s**2 - 2 * self.dsdt / self.tau_s)
        self.S = self.S + dt * self.dsdt

        cdef np.ndarray Isyn = self.S * self.gbarS * self.w * (self.post.V - self.Erev )

        # print(self.post.V[-1])
        # Isyn = -10 * pre_flow * self.w
        self.post.add_Isyn(-Isyn)

        return
########################################################################

cdef class Network:

    cdef list neurons, synapses

    def __cinit__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses


    cpdef update(self, dt, duration=None):

        if (duration is None):
            duration = dt


        t = 0

        while(t < duration):

            sum_Pts = 0
            sum_flow = 0

            for idx, neuron in enumerate(self.neurons):
                ts_states, Pts, times, flow  = neuron.update(dt)
                if not (neuron.is_artifitial_generator()):
                    sum_Pts += Pts
                    sum_flow += flow

            
            for synapse in self.synapses:
                synapse.update(dt)

            t += dt


        generators_signal = []


        for neuron in self.neurons:
            if (neuron.is_artifitial_generator()):
                generators_signal.append(neuron.hist )

        return  ts_states, sum_Pts, times, sum_flow, generators_signal  #         return times, sum_flow, generators_signal  #


    def getflow(self):
        flow = 0
        for neuron in self.neurons:
            if not neuron.is_artifitial_generator():
                flow += neuron.get_flow_hist()


        # flow /= len(self.neurons)

        return flow

    def getCV(self):
        sumCV = 0

        for neuron in self.neurons:
            if not neuron.is_artifitial_generator():
                sumCV += np.asarray( neuron.get_CV() )

        # flow /= len(self.neurons)

        return sumCV

    def get_artificial_signals(self):
        signals = []

        for neuron in self.neurons:

            if neuron.is_artifitial_generator():
                signals.append(neuron.get_hist())

        return signals

       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
