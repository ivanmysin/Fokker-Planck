import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
from scipy.signal import parzen
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import erf


class EzhikevichModel:
    def __init__(self, a, b, c, d, Iext):
        self.V = -65
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.Iext = Iext
        self.u = self.b * self.V

        self.t = 0

    def update(self, dt):
        self.V = self.V + 0.5 * dt * (0.04 * self.V ** 2 + 5 * self.V + 140 - self.u + self.Iext)
        self.V = self.V + 0.5 * dt * (0.04 * self.V ** 2 + 5 * self.V + 140 - self.u + self.Iext)

        self.u = self.u + dt * self.a * (self.b * self.V - self.u)

        if self.V >= 30:
            self.V = self.c
            self.u += self.d

        self.t += dt
        return self.t - dt, self.V

class MC_LIF_Neurons:
    def __init__(self, N, muext, sigmaext, tau=20, Vt=20, Vr=0):
        self.N = N
        self.tau = 20
        self.Vt = Vt  # threshold
        self.Vr = Vr  # reset
        self.muext = muext
        self.sigmaext = sigmaext
        self.V = np.zeros(self.N) + self.Vr

        self.win = parzen(7)

    def update(self, dt):

        self.V += dt * (-self.V + self.muext + self.sigmaext * np.random.randn(self.N)) / self.tau

        fired = self.V >= self.Vt
        self.V[fired] = self.Vr

        y, x = np.histogram(self.V, bins=100, range=[-1, 20], weights=None, density=False)

        y = np.convolve(y, self.win, mode="same")

        y = y / y.sum()

        return x[:-1], y



class MC_LIF_Networks:
    def __init__(self, N, muext, sigmaext, S, tau=20, Vt=20, Vr=0):
        self.N = N
        self.tau = 20
        self.Vt = Vt  # threshold
        self.Vr = Vr  # reset
        self.muext = muext
        self.sigmaext = sigmaext
        self.S = S

        self.V = np.zeros(self.N) + self.Vr

        self.win = parzen(7)
        self.Isyn = 0

        self.firings_x = np.empty((1, 0), dtype=float)
        self.firings_y = np.empty((1, 0), dtype=float)
        self.neuron_indexes = np.arange(self.N)
        self.t = 0

    def update(self, dt):

        self.V += dt * (-self.V + self.muext + self.Isyn + self.sigmaext**2 * np.random.randn(self.N) ) / self.tau

        fired = self.V >= self.Vt
        self.V[fired] = self.Vr

        self.Isyn = np.sum(self.S[:, fired], axis=1)


        y1, x1 = np.histogram(self.V, bins=100, range=[0, 20], weights=None, density=False)

        y1 = np.convolve(y1, self.win, mode="same")

        y1 = y1 / y1.sum()

        self.firings_x = np.append(self.firings_x, self.t  )
        self.firings_y = np.append(self.firings_y, 1000 * np.sum(fired) / self.N / dt)

        self.t += dt

        return x1[:-1], y1, self.firings_x, self.firings_y

class HeatEquation:

    def __init__(self, N, a, h):

        self.N = N
        self.U = np.zeros(N)
        self.U[N//2] = 10
        self.a = a
        self.hrev = 1 / h**2
        self.x_grid = np.linspace(0, 1, self.N)

        self.t = 0


    def update(self, dt):

        self.U[0] = 0
        self.U[-1] = 0

        dU = 0

        for i in range(1, self.N - 1):


            self.U[i] = self.U[i] + dt * self.a * self.hrev * ( self.U[i+1] - 2 * self.U[i] + self.U[i-1] )

            dU += dt * self.a * self.hrev * ( self.U[i+1] - 2 * self.U[i] + self.U[i-1] )


        if (np.sum(self.U < 0) > 0):
            print ("Below zero!!!")
        print (dU)

        self.t += dt
        return self.x_grid, self.U, [], []

class HeatEquationWithNonImplicit(HeatEquation):

        def __init__(self, N, a, h):

            super().__init__(N, a, h)


        def update(self, dt):


            self.U[0] = 0
            self.U[-1] = 0

            old_U = np.copy(self.U)

            b = self.a * dt * self.hrev

            A = np.zeros([self.N, self.N], dtype=float)

            di = np.diag_indices(self.N)

            A[di] = -(1 + 2 * b)

            A[ (di[0]+1)[:-1], di[1][:-1]] = b

            A[ di[0][:-1], (di[1]+1)[:-1]] = b

            self.U = np.dot(np.linalg.inv(A), -self.U)

            print("Conservative of scheme is ", np.sum (self.U - old_U) )

            self.t += dt
            return self.x_grid, self.U, [], []




class Fokker_Plank4LIF:

    def __init__(self, hstates, muext, sigmaext, tau=20, Vt=20, Vr=5):


        self.Vstates = np.arange(0, Vt, hstates)
        self.dV = hstates

        self.Nstates = self.Vstates.size

        self.Pv = np.zeros_like(self.Vstates)
        self.Pv[np.argmin(np.abs(self.Vstates - Vr))] = 1


        self.Vt = Vt  # threshold
        self.Vr = Vr  # reset
        self.muext = muext
        self.sigmaext = sigmaext
        self.hrev = sigmaext**2 / 2 / tau
        self.tau = tau

        self.reset_ind = np.argmin( (self.Vstates - self.Vr)**2 )

        self.shift = (-self.Vstates + self.muext) / self.tau


        self.t = 0
        self.flow_x = np.empty((1, 0), dtype=float)
        self.flow_y = np.empty((1, 0), dtype=float)

        self.Pv[0] = 0
        self.Pv[-1] = 0




    def update(self, dt):

        self.Pv[0] = 0
        self.Pv[-1] = 0


        s = 0

        for i in range(1, self.Nstates - 1):


            dPv_dv = (self.Pv[i + 1] - self.Pv[i - 1]) / (2 * self.dV)

            d2P_dv2 = (self.Pv[i + 1] - 2 * self.Pv[i] + self.Pv[i - 1]) / (self.dV**2)

            dP = dt * self.hrev * d2P_dv2 - dt * dPv_dv * self.shift[i] + dt * self.Pv[i] / self.tau

            s += dP

            self.Pv[i] = self.Pv[i] + dP


        flow = dt * 0.5 * self.sigmaext**2 / self.tau * (self.Pv[-2] - self.Pv[-1]) / self.dV

        # 1 - np.sum(self.Pv) #

        self.Pv[self.reset_ind] += flow

        print(np.sum(self.Pv < 0))

        self.Pv[self.Pv < 0] = 0    # !!!!!!!!!!!!!!!!!!!
        self.Pv = self.Pv / np.sum(self.Pv) # !!!!!!!!!!!!!


        s += flow




        self.flow_x = np.append(self.flow_x, self.t)
        self.flow_y = np.append(self.flow_y, flow / dt * 1000)

        self.t += dt

        return self.Vstates, self.Pv, self.flow_x, self.flow_y # self.Vstates, sig #




class Fokker_Plank4LIFimplicit(Fokker_Plank4LIF):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.Pv = self.Pv.reshape(-1, 1)

    def update(self, dt):


        self.Pv[0] = 0
        self.Pv[-1] = 0

        old_Pv = np.copy(self.Pv)

        b = self.sigmaext**2 * dt * 0.5 / self.tau / self.dV**2

        c = -self.shift * dt * 0.5 / self.dV

        A = np.zeros([self.Nstates, self.Nstates], dtype=float)

        di = np.diag_indices(self.Nstates)

        A[di] = dt/self.tau - 1 - 2 * b

        A[(di[0] + 1)[:-1], di[1][:-1]] = b - c[:-1]

        A[di[0][:-1], (di[1] + 1)[:-1]] = b + c[:-1]

        self.Pv = np.dot(np.linalg.inv(A), -self.Pv)

        flow = dt * 0.5 * self.sigmaext**2 / self.tau * (self.Pv[-2] - self.Pv[-1]) / self.dV

        self.Pv[self.reset_ind] += flow


        # print (np.sum(old_Pv - self.Pv))

        # print (np.sum(self.Pv < 0) )
        # print (self.Pv.shape)

        sig = np.sign(self.Pv)

        self.Pv[self.Pv < 0] = 0

        self.Pv = self.Pv / np.sum(self.Pv)

        self.flow_x = np.append(self.flow_x, self.t)
        self.flow_y = np.append(self.flow_y, flow / dt * 1000)

        self.t += dt

        return self.Vstates, self.Pv, self.flow_x, self.flow_y #  self.Vstates, sig #



class TransportEquastion:

    def __init__(self, dt, dts, Nts, tau_m=10, Vr=0, Vt=10, Iext=15, sigma=1.5):

        # параметры пространства
        self.dt = dt
        self.dts = dts
        self.Nts = Nts

        # объявляем переменные для симуляции Р
        self.ts_states = np.linspace(0, self.dts * self.Nts, self.Nts)
        self.Pts = np.zeros_like(self.ts_states)
        self.Pts[-1] = 1
        self.dt_dts_ratio = self.dt/self.dts
        self.half_dt_ratio = 0.5 * (1 - self.dt_dts_ratio)
        self.dP = np.zeros_like(self.Pts)

        # параметры нейрона
        self.tau_m = tau_m
        self.Vt = Vt
        self.Vr = Vr
        self.Iext = Iext
        self.sigma = sigma

        # параметры для симуляции потенциала
        self.V = np.zeros_like(self.ts_states) + self.Vr
        self.dV = np.zeros_like(self.ts_states)
        self.dV_dt = np.zeros_like(self.ts_states)

        # переменные для записи потока
        self.flow_x = []
        self.flow_y = []

        self.t = 0

    def update_V(self):

        for i in range(1, self.Nts-1):

            if i > 1:

                a = self.V[i+1] - self.V[i]
                b = self.V[i] - self.V[i-1]

                a_ = self.V[i] - self.V[i-1]
                b_ = self.V[i-1] - self.V[i-2]

                wi = self.limiter(a, b)
                wi_1 = self.limiter(a_, b_)

            else:
                a = self.V[i + 1] - self.V[i]
                b = self.V[i] - self.V[i - 1]
                wi = self.limiter(a, b)
                wi_1 = 0

            self.dV_dt[i] = (-self.V[i] + self.Iext) / self.tau_m
            self.dV[i] = -self.dt_dts_ratio * (self.V[i] - self.V[i-1] + self.half_dt_ratio*( wi - wi_1 ) ) + self.dt * self.dV_dt[i]



        self.dV[0] = 0
        self.dV[-1] = self.dt * (-self.V[-1] + self.Iext) / self.tau_m
        self.V += self.dV


    def H_function(self):

        k = self.tau_m / self.dt
        g_tot = 1 / self.tau_m # (Cm = 1) !!!!

        T = np.sqrt(0.5*(1+k)) * g_tot * (self.Vt - self.V) / self.sigma

        A_inf = np.exp(0.0061 - 1.12 * T - 0.257 * T**2 - 0.072 * T**3 - 0.0117 * T**4)
        A = A_inf * (1 - (1 + k)**(-0.71 + 0.0825 * (T + 3) ) )

        dT_dt = -g_tot/self.sigma * np.sqrt(0.5+0.5*k) * self.dV_dt

        dT_dt[dT_dt < 0] = 0

        F_T = np.sqrt(2/np.pi) * np.exp(-T**2) / (1 + erf(T))

        B = -np.sqrt(2) * self.tau_m * dT_dt * F_T

        H = A + B

        return H

    def limiter(self, a, b):
        if a * b <= 0:
            w = 0

        elif a < 0 and a*b > 0:
            w = -min( [abs(a + b)*0.5, 2*min([abs(a), abs(b) ]) ])

        else:
            w = min( [abs(a + b)*0.5, 2*min([ abs(a), abs(b) ]) ])

        return w




    def update(self, dt):

        H = self.H_function()

        for i in range(1, self.Nts-1):

            a = self.Pts[i + 1] - self.Pts[i]
            b = self.Pts[i] - self.Pts[i - 1]
            wi = self.limiter(a, b)

            if i > 1:
                a_ = self.Pts[i] - self.Pts[i-1]
                b_ = self.Pts[i-1] - self.Pts[i-2]
                wi_1 = self.limiter(a_, b_)

            else:
                wi_1 = 0

            self.dP[i] = - self.dt_dts_ratio * (self.Pts[i] - self.Pts[i-1] + self.half_dt_ratio*( wi - wi_1 ) ) - self.dt * self.Pts[i]*H[i]

        a_ = self.Pts[-1] - self.Pts[-2]
        b_ = self.Pts[-2] - self.Pts[-3]
        wi_1 = self.limiter(a_, b_)

        self.dP[0] = -self.dt_dts_ratio * self.Pts[0] - self.dt * np.sum(-self.Pts * H)
        self.dP[-1] = self.dt_dts_ratio * ( self.Pts[-2] + self.half_dt_ratio*wi_1 ) - self.dt * self.Pts[-1] * H[-1]

        self.Pts += self.dP


        # print(np.sum( self.Pts) )     # !!!!!!!!!
        # self.Pts /= np.sum( self.Pts) # !!!!!!!!!


        self.flow_x.append(self.t)
        self.flow_y.append(1000 * self.Pts[0] / self.dts)


        self.update_V()

        self.t += self.dt

        return self.ts_states, self.Pts, self.flow_x, self.flow_y


class Animator:
    def __init__(self, model, xlim, ylim, Yappend=True, Xappend=True):
        self.xdata = []
        self.ydata = []

        self.Fig, self.ax = plt.subplots(nrows=2, ncols=1)
        self.line1, = self.ax[0].plot(self.xdata, self.ydata, 'b', animated=True)
        self.time_text = self.ax[0].text(0.05, 0.9, '', transform=self.ax[0].transAxes)

        self.ax[0].set_xlim(xlim[0], xlim[1])
        self.ax[0].set_ylim(ylim[0], ylim[1])


        self.line2, = self.ax[1].plot([], [], 'b', animated=True)

        self.ax[1].set_xlim(xlim[2], xlim[3])
        self.ax[1].set_ylim(ylim[2], ylim[3])

        self.model = model

        self.Yappend = Yappend
        self.Xappend = Xappend


    def update_on_plot(self, idx):

        x1, y1, x2, y2 = self.model.update(self.dt)

        if (self.Xappend):
            self.xdata.append(x1)
        else:
            self.xdata = x1

        if (self.Yappend):
            self.ydata.append(y1)
        else:
            self.ydata = y1


        self.line1.set_data(self.xdata, self.ydata)
        self.time_text.set_text("simulation time is %.2f in ms" % idx)

        self.line2.set_data(x2, y2)

        return [self.line1, self.time_text, self.line2]

    def run(self, dt, duration, interval=10):

        self.dt = dt
        self.duration = duration
        ani = FuncAnimation(self.Fig, self.update_on_plot, frames=np.arange(0, self.duration, self.dt), interval=interval, blit=True, repeat=False)

        plt.show()







# def main():
#     a = 0.02
#     b = 0.2
#     c = -65
#     d = 8
#     Iext = 4
#
#
#     neuron = EzhikevichModel(a, b, c, d, Iext)
#     animator = Animator(neuron, [0, 1000], [-80, 40], Yappend=True, Xappend=True)
#     animator.run(0.5, 1000)


def main():

    N = 5000
    muext = 25 # + 1.5 * np.random.randn(N)
    sigmaext = 3.5
    S = np.zeros( [N, N] ) # 2*np.random.rand(N, N)


    # np.append(0.5 * rand(Ne + Ni, Ne), -rand(Ne + Ni, Ni), axis=1)


    LIF_neurons = MC_LIF_Networks(N, muext, sigmaext, S, tau=20, Vt=20, Vr=5)

    animator = Animator(LIF_neurons, [0, 20, 0, 1000], [0, 0.2, 0, 400], Yappend=False, Xappend=False)
    animator.run(0.1, 1000, 10)


# def main():
#     # solve heat equation
#     N = 100
#     a = 0.01
#     h = 0.5
#
#     he = HeatEquationWithNonImplicit(N, a, h)
#
#     animator = Animator(he, [0, 1, 0, 1000], [-0.5, 10, 0, 1], Yappend=False, Xappend=False)
#     animator.run(0.5, 1000, 10)




# def main():
#
#
#     h = 0.25
#     muext = 25
#     sigmaext = 1.5
#     tau = 20
#     dt = 0.1
#     LIF_neurons = Fokker_Plank4LIF(h, muext, sigmaext, tau=tau, Vt=20, Vr=5)
#     # LIF_neurons = Fokker_Plank4LIFimplicit(h, muext, sigmaext, tau=tau, Vt=20, Vr=5)
#
#     animator = Animator(LIF_neurons, [0, 20, 0, 1000], [0, 0.5, 0, 100], Yappend=False, Xappend=False)
#
#
#
#     stability = 0.5 * sigmaext**2 * dt / tau / h**2
#
#
#
#     print (stability)
#     animator.run(dt, 1000, 10)

#
# def main():
#
#
# # solve transport equation
#     Nts = 400
#     dt = 0.1
#     dts = 0.5
#
#     he = TransportEquastion(dt, dts, Nts)
#
#     animator = Animator(he, [0, 200, 0, 1000], [0, 1.2, 0, 1000], Yappend=False, Xappend=False)
#
#
#     animator.run(dt, 1000, 10)



if __name__ == "__main__":
    main()