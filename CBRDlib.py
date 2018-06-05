import numpy as np
from scipy.special import erf

SQRT_FROM_2 = np.sqrt(2)
SQRT_FROM_2_PI = np.sqrt(2 / np.pi)

# base neuron class for CBRD or Monte-Carlo
class BaseNeuron:

    def __init__(self, params):

        self.Vreset = params["Vreset"]
        self.Vt = params["Vt"]

        self.gl = params["gl"]
        self.El = params["El"]
        self.C = params["C"]
        self.sigma = params["sigma"]

        self.ref_dvdt = params["ref_dvdt"]   # refactory in updates of V and variables of states
        self.refactory = params["refactory"] # refactory for threshold

        self.w_in_distr = params["w_in_distr"] # weight of neuron in model
        self.Iext = params["Iext"]
        self.Isyn = 0
        self.gsyn = 0
        self.is_use_CBRD = params["use_CBRD"] # flag use CBRD or Monte-Carlo


        self.N = params["N"]

        self.V = np.zeros(self.N)
        self.dts = params["dts"]

        if self.is_use_CBRD:

            self.t_states = np.linspace(0, self.N * self.dts, self.N)
            self.ro = np.zeros_like(self.t_states)
            self.ro[-1] = 1 / self.dts
            self.ro_H_integral = 0
            self.ts = 0



            self.ref_idx = int(self.refactory / self.dts)
            self.ref_dvdt_idx = int(self.ref_dvdt / self.dts)

            self.max_roH_idx = 0

        else:
            self.ts = np.zeros_like(self.V) + 200

        self.firing = [0]
        self.CVhist = [0]
        self.saveCV = True



    def H_function(self, V, dVdt, tau_m, Vt, sigma):
        T = (Vt - V) / sigma / SQRT_FROM_2
        A = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        dT_dt = -1.0 / sigma / SQRT_FROM_2 * dVdt
        dT_dt[dT_dt > 0] = 0
        F_T = SQRT_FROM_2_PI * np.exp(-T**2) / (1.000000001 + erf(T))
        B = -SQRT_FROM_2 * dT_dt * F_T * tau_m
        H = (A + B) / tau_m
        return H


    def update_ro(self, dt, dVdt, tau_m):
        shift = False

        if self.ts >= self.dts:
            self.ro[-1] += self.ro[-2]
            self.ro[:-1] = np.roll(self.ro[:-1], 1)
            self.ro[0] = self.ro_H_integral
            self.ro_H_integral = 0
            self.ts = 0
            shift = True

            # self.ro /= np.sum(self.ro) * self.dts
            # print (np.sum(self.ro) * self.dts)


        H = self.H_function(self.V, dVdt, tau_m, self.Vt, self.sigma)

        H[:self.ref_idx] = 0
        dro = self.ro * (1 - np.exp(-H * dt))  # dt * self.ro * H  #
        self.max_roH_idx = np.argmax(dro)
        self.ro -= dro

        # self.ro[self.ro < 0] = 0
        self.ro_H_integral += np.sum(dro)
        self.ts += dt

        # print(self.ro)

        # if self.saveCV and shift:
        #     roH = self.ro * H * self.dts
        #     isi_mean_of2 = np.sum(self.t_states**2 * roH) / np.sum(roH)
        #
        #     isi_mean = (np.sum(self.t_states * roH) / np.sum(roH))**2
        #
        #     if isi_mean < isi_mean_of2 and isi_mean_of2 > 0:
        #         CV = np.sqrt(isi_mean_of2 / isi_mean - 1)
        #     else:
        #         CV = 0
        #
        #
        #     self.CVhist.append(CV)
        # else:
        #     self.CVhist.append(self.CVhist[-1])



        return shift


    def add_Isyn(self, Isyn, gsyn):
        self.Isyn += Isyn
        self.gsyn += gsyn

    def update(self, dt):
        return 0, 0, 0, 0


    def get_flow(self):
        return self.w_in_distr * self.firing[-1]

    def get_flow_hist(self):
        return self.w_in_distr * np.asarray(self.firing)

    def get_CV(self):
        return self.w_in_distr * np.asarray(self.CVhist)

    def getV(self):
        return self.V

    def get_weights(self):

        if self.is_use_CBRD:
            weights = self.w_in_distr * self.ro * self.dts
        else:
            weights = self.w_in_distr * np.ones_like(self.V) / self.V.size

        return weights


##################################################################

class LIF_Neuron(BaseNeuron):
    artifitial_generator = False

    def __init__(self, params):

        super(LIF_Neuron, self).__init__(params)

        self.V += self.Vreset

        self.Vhist = [self.V[-1]]

        self.times = [0]

        if self.is_use_CBRD:
            self.sigma = self.sigma / self.gl * np.sqrt(0.5 * self.gl / self.C)


    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt

        t = 0
        while (t < duration):

            # dVdt = -self.V / self.tau_m + self.Iext / self.tau_m + self.Isyn
            dVdt = (self.gl * (self.El - self.V) + self.Iext + self.Isyn) / self.C

            tau_m = self.C / (self.gl + self.gsyn)

            if self.is_use_CBRD:
                dVdt[:self.ref_dvdt_idx ] = 0
            else:
                dVdt += np.random.normal(0, self.sigma, self.V.size)  / self.C / np.sqrt(dt)
                dVdt[self.ts < self.ref_dvdt] = 0


            self.V += dt * dVdt

            # print(self.V[-1])

            self.Vhist.append(self.V[-1])

            self.Isyn = 0
            self.gsyn = 0

            if self.is_use_CBRD:
                shift = self.update_ro(dt, dVdt, tau_m)
                self.firing.append(1000 * self.ro[0])
                if shift:
                    self.V[:-1] = np.roll(self.V[:-1], 1)
                    self.V[0] = self.Vreset

            else:
                spiking = self.V >= self.Vt
                self.V[spiking] = self.Vreset
                self.firing.append(1000 * np.mean(spiking) / dt)
                self.ts += dt
                self.ts[spiking] = 0


            self.times.append(self.times[-1] + dt)

            t += dt

        if self.is_use_CBRD:
            return self.t_states, self.w_in_distr * self.ro, self.times, self.w_in_distr * np.asarray(
                self.firing),[]  # , self.t_states, self.V
        else:
            return np.zeros(400), np.zeros(400), self.times, self.w_in_distr * np.asarray(self.firing), []
#########################################################################################

##################################################################
class Channel:
    def __init__(self, gmax, E, V, x=None, y=None, x_reset=None, y_reset=None):
        self.gmax = gmax
        self.E = E
        self.g = 0
        if not x is None:
            self.x = self.get_x_inf(V)
            self.x_reset = x_reset
        else:
            self.x = None

        if not y is None:
            self.y = self.get_y_inf(V)
            self.y_reset = y_reset
        else:
            self.y = None

    def update(self, dt, V, mask=slice(0, None)):
        self.g = self.gmax
        V = V[mask]
        if not (self.x is None):
            x_inf = self.get_x_inf(V)
            tau_x = self.get_tau_x(V)
            self.x[mask] = x_inf - (x_inf - self.x[mask]) * np.exp(-dt / tau_x)
            self.g *= self.x

        if not (self.y is None):
            y = self.y[mask]
            y_inf = self.get_y_inf(V)
            tau_y = self.get_tau_y(V)
            self.y[mask] = y_inf - (y_inf - self.y[mask]) * np.exp(-dt / tau_y)
            self.g *= self.y


    def get_I(self, V):
        I = self.g * (V - self.E)
        return I

    def get_g(self):
        return self.g

    def reset(self, spiking):
        if not (self.x is None):
            self.x[spiking] = self.x_reset

        if not (self.y is None):
            self.y[spiking] = self.y_reset

    def get_x_inf(self, V):
        return 0

    def get_y_inf(self, V):
        return 0

    def get_tau_x(self, V):
        return 0

    def get_tau_y(self, V):
        return 0

    def roll(self, max_roH_idx):
        if not (self.x is None):
            tmpx = self.x[max_roH_idx]
            self.x[:-1] = np.roll(self.x[:-1], 1)
            self.x[0] = tmpx

        if not (self.y is None):
            tmpy = self.y[max_roH_idx]
            self.y[:-1] = np.roll(self.y[:-1], 1)
            self.y[0] = tmpy

class KDR_Channel(Channel):

    def get_a(self, V):
        a = 0.17 * np.exp((V + 5) * 0.09)
        return a

    def get_b(self, V):
        b  = 0.17 * np.exp(-(V + 5) * 0.022)
        return b

    def get_tau_x(self, V):
        a = self.get_a(V)
        b = self.get_b(V)

        tau_x = 1 / (a + b) + 0.8

        return tau_x

    def get_x_inf(self, V):
        a = self.get_a(V)
        b = self.get_b(V)
        x_inf = a / (a + b)
        return x_inf

    def get_tau_y(self, V):
        return 300

    def get_y_inf(self, V):
        y_inf = 1 / (1 + np.exp((V + 68) * 0.038) )
        return y_inf

class A_channel(Channel):

    def get_ax(self, V):
        a = 0.08 * np.exp((V+41)*0.089)
        return a

    def get_bx(self, V):
        b = 0.08 * np.exp(-(V+41)*0.016)
        return b

    def get_ay(self, V):
        a = 0.04*np.exp(-(V+49)*0.11)
        return a

    def get_by(self, V):
        b = 0.04
        return b

    def get_tau_x(self, V):
        a = self.get_ax(V)
        b = self.get_bx(V)
        tau_x = 1 / (a + b) + 1
        return tau_x

    def get_x_inf(self, V):
        a = self.get_ax(V)
        b = self.get_bx(V)
        x_inf = a / (a + b)
        return x_inf

    def get_tau_y(self, V):
        a = self.get_ay(V)
        b = self.get_by(V)
        tau_y = 1 / (a + b) + 2
        return tau_y

    def get_y_inf(self, V):
        a = self.get_ay(V)
        b = self.get_by(V)
        y_inf = a  / (a + b)
        return y_inf

    def update(self, dt, V, mask=slice(0, None)):
        self.g = self.gmax
        V = V[mask]
        if not (self.x is None):
            x_inf = self.get_x_inf(V)
            tau_x = self.get_tau_x(V)
            self.x[mask] = x_inf - (x_inf - self.x[mask]) * np.exp(-dt / tau_x)
            self.g *= self.x**4

        if not (self.y is None):
            y_inf = self.get_y_inf(V)
            tau_y = self.get_tau_y(V)
            self.y[mask] = y_inf - (y_inf - self.y[mask]) * np.exp(-dt / tau_y)
            self.g *= self.y**3

class M_channel(Channel):

    def reset(self, spiking):
        self.x[spiking] += self.x_reset * (1 - self.x[spiking])

    def get_a(self, V):
        a = 0.003 * np.exp( (V+45) * 0.135 )
        return a

    def get_b(self, V):
        b = 0.003 * np.exp(-(V+45)*0.09)
        return b

    def get_x_inf(self, V):
        a = self.get_a(V)
        b = self.get_b(V)
        x_inf = a / (a + b)
        return x_inf

    def get_tau_x(self, V):
        a = self.get_a(V)
        b = self.get_b(V)
        tau_x = 1 / (a + b) + 8
        return tau_x

    def update(self, dt, V, mask=slice(0, None)):
        V = V[mask]
        x_inf = self.get_x_inf(V)
        tau_x = self.get_tau_x(V)
        self.x[mask] = x_inf - (x_inf - self.x[mask]) * np.exp(-dt / tau_x)
        self.g = self.gmax * self.x**2

class AHP_Channel(Channel):

    def get_tau_x(self, V):
        tau_x = 2000 / (3.3 * np.exp( (V + 35)/20 ) + np.exp(-(V + 35)/20) )
        return tau_x

    def get_x_inf(self, V):
        x_inf = 1 / (1 + np.exp(-(V + 35)/10)) # x_inf = 1 / (1 + np.exp(-(V + 35) / 4))  #
        return x_inf

    def reset(self, spiking):
        self.x[spiking] += self.x_reset * (1 - self.x[spiking])

    def update(self, dt, V, mask=slice(0, None)):
        V = V[mask]
        x_inf = self.get_x_inf(V)
        tau_x = self.get_tau_x(V)
        self.x[mask] = x_inf - (x_inf - self.x[mask]) * np.exp(-dt / tau_x)
        self.g = self.gmax * self.x

class HCN_Channel(Channel):

    def get_tau_y(self, V):
        return 180

    def get_y_inf(self, V):
        y_inf = 1.0 / (1 + np.exp((V + 98) * 0.075) )
        return y_inf

    def update(self, dt, V, mask=slice(0, None)):

        V = V[mask]

        y_inf = self.get_y_inf(V)
        tau_y = self.get_tau_y(V)
        self.y[mask] = y_inf - (y_inf - self.y[mask]) * np.exp(-dt / tau_y)
        self.g = self.gmax * self.y


class BorgGrahamNeuron(BaseNeuron):
    artifitial_generator = False
    def __init__(self, params):

        super(BorgGrahamNeuron, self).__init__(params)

        self.V += self.El
        # self.saveV = params["saveV"]
        # self.saveCV = params["saveCV"]
        self.times = [0]

        leak = Channel(params["gl"], params["El"], self.V)
        dr_current = KDR_Channel(params["dr_current"]["g"], params["dr_current"]["E"], self.V, 1, 1,
                                 params["dr_current"]["x_reset"], params["dr_current"]["y_reset"])
        a_current = A_channel(params["a_current"]["g"], params["a_current"]["E"], self.V, 1, 1,
                              params["a_current"]["x_reset"], params["a_current"]["y_reset"])
        m_current = M_channel(params["m_current"]["g"], params["m_current"]["E"], self.V, 1, None,
                              params["m_current"]["x_reset"], params["m_current"]["y_reset"])
        ahp = AHP_Channel(params["ahp"]["g"], params["ahp"]["E"], self.V, 1, None, params["ahp"]["x_reset"],
                          params["ahp"]["y_reset"])
        hcn = HCN_Channel(params["hcn"]["g"], params["hcn"]["E"], self.V, None, 1, params["hcn"]["x_reset"],
                          params["hcn"]["y_reset"])

        self.channels = [leak, dr_current, a_current, m_current, ahp, hcn]


        if self.is_use_CBRD:
            g_tot = 0
            for ch in self.channels:
                mask = np.ones_like(self.V, dtype=np.bool)
                ch.update(0.001, self.V, mask)
                g_tot += ch.get_g()
            self.sigma = self.sigma / g_tot * np.sqrt(0.5 * g_tot / self.C)

        self.roHsumMC = 0
        self.ts4MC = 0
        # if self.saveV:
        self.Vhist = [self.V[-1]]
        # if self.saveCV:
        #     self.CVhist = [0]

    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt

        t = 0
        while (t < duration):

            g_tot = 0
            I = 0

            if self.is_use_CBRD:
                mask = np.ones_like(self.V, dtype=np.bool)
                mask[self.ro < 0.0001] = False
                mask[:self.ref_dvdt_idx] = False
                mask[-1] = True
            else:
                mask = self.ts > self.ref_dvdt
                I += np.random.normal(0, self.sigma, self.V.size) / np.sqrt(dt)



            for ch in self.channels:
                ch.update(dt, self.V, mask)
                I -= ch.get_I(self.V)
                g_tot += ch.get_g()

            g_tot += self.gsyn
            I += self.Iext
            I += self.Isyn



            dVdt = I / self.C

            if self.is_use_CBRD:
                dVdt[:self.ref_dvdt_idx ] = 0
            else:
                dVdt[self.ts < self.ref_dvdt] = 0


            self.V += dt * dVdt

            if self.is_use_CBRD:
                self.Vhist.append( np.sum(self.V * self.ro * self.dts) )
            else:
                self.Vhist.append( np.mean(self.V) )


            self.Isyn = 0
            self.gsyn = 0

            if (self.is_use_CBRD):
                tau_m = self.C / g_tot
                shift = self.update_ro(dt, dVdt, tau_m)
                # print( np.abs(self.V[-1] - self.V[-2]) )
                if shift:
                    self.V[:-1] = np.roll(self.V[:-1], 1)
                    self.V[0] = self.Vreset
                    for ch in self.channels:
                        ch.roll(self.max_roH_idx)
                        ch.reset(0)




            else:
                spiking = np.logical_and((self.V >= self.Vt), (self.ts > self.refactory))
                self.ts += dt

                if np.sum(spiking) > 0:
                    self.V[spiking] = self.Vreset
                    self.ts[spiking] = 0
                    for ch in self.channels:
                        ch.reset(spiking)

                self.roHsumMC = 1000 * self.w_in_distr * np.mean(spiking) / dt



            t += dt

            self.times.append(self.times[-1] + dt)
            if self.is_use_CBRD:
                self.firing.append(1000 * self.w_in_distr * self.ro[0])
            else:
                self.firing.append(self.roHsumMC)

        if self.is_use_CBRD:
            return self.t_states, self.w_in_distr * self.ro, self.times, np.asarray(
                self.firing),[]  # , self.t_states, self.V
        else:
            return np.array([0]), np.array([0]), self.times, np.asarray(self.firing),self.Vhist  # , [], []




#########################################################################################
class  SineGenerator:
    artifitial_generator = True

    def __init__(self, params):

        self.fr = params["fr"]
        self.phase = params["phase"]
        self.amp_max = params["amp_max"]
        self.amp_min = params["amp_min"]
        self.flow = 0
        self.t = 0
        self.hist = [self.flow]

    def update(self, dt):
        self.flow = 0.5 * (np.cos(2*np.pi*self.fr*self.t + self.phase) + 1) * (self.amp_max - self.amp_min) + self.amp_min
        self.t += 0.001 * dt
        self.hist.append(self.flow)
        return 0, 0, self.flow, 0, 0

    def get_flow(self):
        return self.flow

    def add_Isyn(self, Isyn):
        pass

    def get_hist(self):
        return np.asarray(self.hist)

class PoissonGenerator:
    artifitial_generator = True

    def __init__(self, params):

        self.fr = params["fr"]
        self.w = params["w"]
        self.refactory = params["refactory"]
        self.length = params["length"]

        self.flow = 0
        self.previos_t = params["refactory"] + 10
        self.start = self.length + 1

        self.hist = [self.flow]

    def update(self, dt):

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
        return 0, 0, self.flow, 0, 0


    def get_hist(self):
        return np.asarray(self.hist)

    def get_flow(self):
        return self.flow

    def add_Isyn(self, Isyn):
        pass


#########################################################################################
class SimlestSinapse:
    def __init__(self, params):
        self.w = params["w"]
        self.delay = params["delay"]
        self.pre = params["pre"]
        self.post = params["post"]
        self.pre_hist = []

    def update(self, dt):

        if (len(self.pre_hist) == 0) and (self.delay > 0):
            for _ in range(int(self.delay/dt)):
                self.pre_hist.append(0)

        pre_flow = self.pre.get_flow()

        if (self.delay > 0):
            self.pre_hist.append(pre_flow)
            pre_flow = self.pre_hist.pop(0)

        Isyn = self.w * pre_flow

        self.post.add_Isyn(Isyn, 0)

        return



class Synapse(SimlestSinapse):
    def __init__(self, params):
        super(Synapse, self).__init__(params)

        self.tau_s = params["tau_s"] # 5.4
        self.tau = params["tau_a"] # 1
        self.gbarS = params["gbarS"] # 1.0
        self.Erev = params["Erev"] # 50

        self.S = 0
        self.dsdt = 0
        self.tau_s_2 =  self.tau_s**2
        
    def update(self, dt):

        if (len(self.pre_hist) == 0) and (self.delay > 0):
            for _ in range(int(self.delay/dt)):
                self.pre_hist.append(0)

        pre_flow = self.pre.get_flow()

        if (self.delay > 0):
            self.pre_hist.append(pre_flow)
            pre_flow = self.pre_hist.pop(0)


        self.dsdt = self.dsdt + dt * (self.tau * self.gbarS * pre_flow / self.tau_s_2 - self.S / self.tau_s_2 - 2 * self.dsdt / self.tau_s)
        self.S = self.S + dt * self.dsdt


        gsyn = self.S * self.gbarS * self.w
        Isyn = gsyn * (self.post.getV() - self.Erev )

        self.post.add_Isyn(-Isyn, gsyn)

        return


#####################################
class Network:
    def __init__(self, neurons, synapses, is_save_distrib = False):
        self.neurons = neurons
        self.synapses = synapses

        self.CVhist = [0]

        self.is_save_distrib = is_save_distrib

        self.x4p = np.empty(100, dtype=np.float)
        self.Pxvar = np.empty((100, 0), dtype=np.float)



    def update(self, dt, duration=None):

        if (duration is None):
            duration = dt

        t = 0
        while(t < duration):

            sum_Pts = 0
            sum_flow = 0

            if self.is_save_distrib:
                Varr = np.empty(0, dtype=float)
                weights = np.empty(0, dtype=float)

            for idx, neuron in enumerate(self.neurons):
                ts_states, Pts, times, flow, _  = neuron.update(dt)
                if not (neuron.artifitial_generator):
                    sum_Pts += Pts
                    sum_flow += flow

                if self.is_save_distrib:
                    Varr = np.append( Varr, neuron.getV() )
                    weights = np.append(weights,  neuron.get_weights() )


            if self.is_save_distrib:
                disr_y, distr_x = np.histogram(Varr, bins=self.x4p.size, weights=weights, range=[-80, -39], density=True)

                # print(self.x4p.size)

                self.Pxvar = np.append(self.Pxvar, disr_y.reshape(-1, 1), axis=1)
                self.x4p = distr_x[:-1]

            for synapse in self.synapses:
                synapse.update(dt)


            t += dt


        generators_signal = []
        for neuron in self.neurons:
            if (neuron.artifitial_generator):
                generators_signal.append(np.asarray( neuron.hist) )



        return  ts_states, sum_Pts, times, sum_flow, generators_signal

    def get_distrib(self):
        return self.x4p, self.Pxvar

    def getflow(self):
        flow = 0
        for neuron in self.neurons:
            if not neuron.artifitial_generator:
                flow += neuron.get_flow_hist()


        # flow /= len(self.neurons)

        return flow

    def getCV(self):
        sumCV = 0

        for neuron in self.neurons:
            if not neuron.artifitial_generator:
                sumCV += np.asarray( neuron.get_CV() )

        # flow /= len(self.neurons)

        return sumCV

    def get_artificial_signals(self):
        signals = []

        for neuron in self.neurons:

            if neuron.artifitial_generator:
                signals.append(neuron.get_hist())

        return signals

########################################################################################################################
# вспомогательные функции
def get_contour_line(x, y, z):
    from skimage import measure
    from skimage.filters import gaussian

    x = x.reshape(-1)
    y = y.reshape(-1)

    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    z = gaussian(z, sigma=1.5)

    contours = measure.find_contours(z, 0.5)
    # Display the image and plot all contours found

    for n, contour in enumerate(contours):
        x_area = contour[:, 1] * dx + np.min(x)
        y_area = contour[:, 0] * dy + np.min(y)

    return x_area, y_area

