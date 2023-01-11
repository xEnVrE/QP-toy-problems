import numpy
from qpsolvers import solve_qp


class IK():

    def __init__(self, solver, sample_time, gain):

        self.solver = solver
        self.dt = sample_time
        self.k = gain
        self.k_lim = 0.5

        self.has_joint_limits = False
        self.has_joint_vel_limits = False


    def set_limits(self, q_lower, q_upper):

        self.has_joint_limits = True
        self.q_lower = q_lower
        self.q_upper = q_upper


    def set_max_abs_vel(self, q_dot_max):

        self.has_joint_vel_limits = True
        self.q_dot_max_abs = q_dot_max


    def solve(self, q, x, x_des, jacobian):

        v = (x_des - x) / self.dt
        P = jacobian.T @ jacobian
        r = - jacobian.T @ (v * self.k)

        G = None
        h = None
        if self.has_joint_limits:
            G_q = numpy.zeros(shape = (2, 1))
            G_q[0, 0] = 1.0
            G_q[1, 0] = -1.0

            h_q = numpy.zeros(shape = (2, 1))
            h_q[0, 0] = self.k_lim * (self.q_upper - q) / self.dt
            h_q[1, 0] = -self.k_lim * (self.q_lower - q) / self.dt

            G = G_q
            h = h_q

        if self.has_joint_vel_limits:
            G_qd = numpy.zeros(shape = (2, 1))
            G_qd[0, 0] = 1.0
            G_qd[1, 0] = -1.0

            h_qd = numpy.zeros(shape = (2, 1))
            h_qd[0, 0] = self.q_dot_max_abs
            h_qd[1, 0] = self.q_dot_max_abs

            if G is None:
                G = G_qd
                h = h_qd
            else:
                G = numpy.concatenate((G_q, G_qd))
                h = numpy.concatenate((h_q, h_qd))

        sol = solve_qp(P, r, G, h, solver = self.solver)

        return sol[0]
