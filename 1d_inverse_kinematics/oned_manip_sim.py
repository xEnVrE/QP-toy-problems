import numpy

class ManipulatorSim():

    def __init__(self, initial_condition, lower_limit, upper_limit, sample_time):

        self.dt = sample_time
        self.q = initial_condition
        self.q_lower = lower_limit
        self.q_upper = upper_limit


    def step(self, q_dot):

        q_new = self.q + q_dot * self.dt

        # if q_new >= self.q_lower:
        #     q_new = self.q_lower
        # elif q_new <= self.q_upper:
        #     q_new = self.q_upper

        self.q = q_new


    def get_q(self):

        return self.q
