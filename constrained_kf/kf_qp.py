import numpy
import time
from qpsolvers import solve_qp


class KF():

    def __init__(self, solver):

        self.solver = solver
        self.has_equality_constraint = False


    def set_initial_condition(self, x_0, P_0):

        # Initial conditions
        self.x = x_0
        self.P = P_0


    def set_motion_model(self, A, Q):

        # Motion model
        self.A = A

        # Process covariance matrix
        self.Q = Q


    def set_measurement_model(self, H, R):

        # Measurement model
        self.H = H

        # Measurement covariance matrix
        self.R = R


    def set_equality_constraint(self, constraint):

        self.has_equality_constraint = True
        self.equality_constraint = constraint


    def step(self, x, P, y):

        # Prediction
        x_m = self.A @ x
        P_m = self.A @ P @ self.A.T + self.Q

        # Covariance correction step
        S = self.H @ P_m @ self.H.T + self.R
        K = P_m @ self.H.T @ numpy.linalg.inv(S)
        P_p = (numpy.eye(x.shape[0]) - K @ self.H) @ P_m @ (numpy.eye(x.shape[0]) - K @ self.H).T + K @ self.R @ K.T

        # QP-based state correction step
        P_p_inv = numpy.linalg.inv(P_p)
        R_inv = numpy.linalg.inv(self.R)

        P_pred = P_p_inv
        P_corr = self.H.T @ R_inv @ self.H
        P_qp = P_pred + P_corr

        q_pred = (-P_p_inv @ x_m).reshape(x.shape)
        q_corr = (-self.H.T @ R_inv @ y).reshape(x.shape)
        q_qp = q_pred + q_corr

        A_qp = None
        b_qp = None
        if self.has_equality_constraint:
            A_qp = self.equality_constraint.A(x)
            b_qp = self.equality_constraint.b(x)

        x_p = solve_qp(P_qp, q_qp, A = A_qp, b = b_qp, solver = self.solver)

        return x_p, P_p


    def filter(self, y):

        self.x, self.P = self.step(self.x, self.P, y)

        return self.x, self.P


    def filter_batch(self, ys):

        times = []
        states = []
        for i in range(ys.shape[1]):
            t0 = time.time()
            x, P = self.filter(ys[:, i])

            states.append(x)
            times.append(time.time() - t0)

        return numpy.array(times), numpy.array(states).T
