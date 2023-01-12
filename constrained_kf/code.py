import numpy


class KF():

    def __init__(self, sample_time, p_psd, w_psd, b_psd, m_cov):

        self.dt = sample_time

        self.p_psd = p_psd
        self.w_psd = w_psd
        self.b_psd = b_psd
        self.m_cov = m_cov

        self.x = None
        self.P = None

    def measurement(self, state):

        return self.H(state) @ state


    def A(self, state):

        x1 = state[0]
        w = state[2]
        # A = numpy.array([[1, self.T], [-self.w*self.w*self.T, 1.0]])
        # A = numpy.array([[1, self.T, 0.0], [-w * w * self.T, 1.0, -2 * x1 * w * self.T], [0.0, 0.0, 1.0]])
        # A = numpy.array([[1, self.T, 0.0], [-w * self.T, 1.0, - x1 * self.T], [0.0, 0.0, 1.0]])
        A = numpy.array([[1, self.T, 0.0, 0.0, 0.0], [-w * self.T, 1.0, - x1 * self.T, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, self.T], [0.0, 0.0, 0.0, 0.0, 1.0]])

        return A


    def H(self, state):

        H = numpy.array([[1.0, 0.0, 0.0, 1.0, 0.0]])

        return H


    def Q(self, state):

        x1 = state[0]
        w = state[2]
        # self.Q = numpy.array([[(T ** 3)/ 3.0, (T ** 2) / 2.0], [(T ** 2) / 2.0, T]]) * self.p_psd
        # Q = numpy.array([[(self.T ** 3)/ 3.0 * self.p_psd, (self.T ** 2) / 2.0 * self.p_psd, 0.0], [(self.T ** 2) / 2.0 * self.p_psd, self.T * self.p_psd - (self.T ** 3) * 4.0 / 3.0 * x1 * x1 * w * w * self.w_psd, -(self.T ** 2) * x1 * w * self.w_psd], [0.0, -(self.T ** 2) * x1 * w * self.w_psd, self.T * self.w_psd]])
        # Q = numpy.array([[(self.T ** 3)/ 3.0 * self.p_psd, (self.T ** 2) / 2.0 * self.p_psd, 0.0], [(self.T ** 2) / 2.0 * self.p_psd, self.T * self.p_psd - (self.T ** 3) / 3.0 * x1 * x1 * self.w_psd, -(self.T ** 2) / 2.0 * x1 * self.w_psd], [0.0, -(self.T ** 2) / 2.0  * x1 * self.w_psd, self.T * self.w_psd]])
        Q = numpy.array([[(self.T ** 3)/ 3.0 * self.p_psd, (self.T ** 2) / 2.0 * self.p_psd, 0.0, 0.0, 0.0], [(self.T ** 2) / 2.0 * self.p_psd, self.T * self.p_psd - (self.T ** 3) / 3.0 * x1 * x1 * self.w_psd, -(self.T ** 2) / 2.0 * x1 * self.w_psd, 0.0, 0.0], [0.0, -(self.T ** 2) / 2.0  * x1 * self.w_psd, self.T * self.w_psd, 0.0, 0.0], [0.0, 0.0, 0.0, (self.T ** 3)/ 3.0 * self.b_psd, (self.T ** 2) / 2.0 * self.b_psd], [0.0, 0.0, 0.0, (self.T ** 2) / 2.0 * self.b_psd, self.T * self.b_psd]])

        return Q


    def R(self, state):

        R = numpy.array([[self.m_cov]])

        return R


    def setup(self, p_0, b_0, f_0):

        # x_0 = numpy.array([p_0, 0.0, 2.0 * numpy.pi * f_0])
        # P_0 = numpy.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
        x_0 = numpy.array([p_0, 0.0, (2.0 * numpy.pi * f_0) ** 2, b_0, 0.0])
        P_0 = numpy.array([[0.01, 0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0, 0.0], [0.0, 0.0, 0.01, 0.0, 0.0], [0.0, 0.0, 0.0, 0.01, 0.0], [0.0, 0.0, 0.0, 0.0, 0.01]])

        self.x = x_0
        self.P = P_0


    def filter_update(self, x, P, y):

        # Integrate dynamics
        x_m = self.integrate_dynamics(x)

        # EKF prediction steps
        A = self.A(x)
        P_m = A @ P @ A.T + self.Q(x)

        # innovation
        e = y - self.measurement(x_m)

        # EKF correction steps
        H = self.H(x_m)
        R = self.R(x_m)
        S = H @ P_m @ H.T + R
        K = P_m @ H.T @ numpy.linalg.inv(S)
        x_p = x_m + K @ e
        # P_p = P_m - K @ S @ K.T
        # P_p = (numpy.eye(3) - K @ H) @ P_m @ (numpy.eye(3) - K @ H).T + K @ R @ K.T
        P_p = (numpy.eye(5) - K @ H) @ P_m @ (numpy.eye(5) - K @ H).T + K @ R @ K.T

        return x_p, P_p


    def filter(self, y):

        self.x, self.P = self.filter_update(self.x, self.P, y)

        return self.x, self.P


    def filter_batch(self, ys):

        states = []
        for i in range(ys.shape[0]):
            x, P = self.filter(ys[i])
            states.append(x)

        return numpy.array(states)


class Filter():

    def __init__(self, frequency, p_psd, m_cov):

        T = 1.0 / float(frequency)

        f = 1.0
        w = 2.0 * numpy.pi * f
        self.A = numpy.array([[1, T], [-w*w*T, 1.0]])
        self.H = numpy.array([[1.0, 0.0]])
        self.Q = numpy.array([[(T ** 3)/ 3.0, (T ** 2) / 2.0], [(T ** 2) / 2.0, T]]) * p_psd
        self.R = numpy.array([[m_cov]])

        self.x = None
        self.P = None


    def setup(self):

        x_0 = numpy.array([0.0, 0.0])
        P_0 = numpy.array([[0.01, 0.0], [0.0, 0.01]])

        self.x = x_0
        self.P = P_0
        self._filter = pykalman.KalmanFilter\
        (
            initial_state_mean = x_0,
            initial_state_covariance = P_0,
            transition_matrices = self.A,
            observation_matrices = self.H,
            transition_covariance = self.Q,
            observation_covariance = self.R,
            em_vars = ['']
        )


    def filter(self, y):

        self.x, self.P = self._filter.filter_update(self.x, self.P, y)

        return self.x


    def filter_batch(self, ys):

        x, P = self._filter.filter(ys)

        return x


    def smooth_batch(self, ys):

        self._filter.filter(ys)
        x, P = self._filter.smooth(ys)

        return x


def main():

    # Params
    fps = 30.0
    T = 20.0
    f1 = 0.5
    f2 = 0.6
    a = 0.1

    # Generate data
    dt = 1.0 / fps
    N = int(T / dt)
    t = dt * numpy.linspace(0, N, num = N)
    x_gt = []
    b_gt = []
    bv_gt = []
    v_gt = []
    meas = []
    w = []
    switch = True
    f = f1

    speed = 0.1
    integral = 0

    for i in range(N):
        integral += speed * dt
        arg = 2 * numpy.pi * f * i * dt
        if switch and (i * dt > T / 2) and abs(numpy.sin(arg)) < 0.001:
            f = f2
            switch = False
        b = integral
        p = a * numpy.sin(arg)
        v = a * 2 * numpy.pi * f * numpy.cos(arg)
        m = p + b + numpy.random.normal(0.0, 0.01)
        x_gt.append(p)
        v_gt.append(v)
        b_gt.append(b)
        bv_gt.append(speed)
        meas.append(m)
        w.append(f)

    x_gt = numpy.array(x_gt)
    v_gt = numpy.array(v_gt)
    b_gt = numpy.array(b_gt)
    bv_gt = numpy.array(bv_gt)
    meas = numpy.array(meas)
    w = numpy.array(w)
    # x_gt = a * numpy.sin(2 * numpy.pi * f * t)
    # v_gt = a * 2 * numpy.pi * f * numpy.cos(2 * numpy.pi * f * t)
    # meas = x_gt + numpy.random.normal(0.0, 0.01, N)

    # Filter data
    p_psd = 1.0
    w_psd = 1000.0
    b_psd = 100.0
    cov = 0.1
    p_0 = 0.0
    b_0 = 0.0
    f_0 = 1.0
    filter = EKFFilter(fps, p_psd, w_psd, b_psd, cov)
    filter.setup(p_0, b_0, f_0)
    estimate = filter.filter_batch(meas)
    print(estimate.shape)

    # Plot
    fig, ax = plt.subplots(5, figsize = (20, 10))
    ax[0].plot(t, x_gt)
    ax[0].plot(t, estimate[:, 0])
    # ax[0].plot(t, meas)
    # ax[0].set_ylim(-0.2, 0.2)

    ax[1].plot(t, v_gt)
    ax[1].plot(t, estimate[:, 1])

    ax[2].plot(t, w)
    ax[2].plot(t, numpy.sqrt(estimate[:, 2]) / (2.0 * numpy.pi))

    ax[3].plot(t, b_gt)
    ax[3].plot(t, estimate[:, 3])

    ax[4].plot(t, bv_gt)
    ax[4].plot(t, estimate[:, 4])

    plt.show()


if __name__ == '__main__':
    main()
