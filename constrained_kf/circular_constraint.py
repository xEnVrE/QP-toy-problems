import numpy


class CircularConstraint():


    def __init__(self, gain, radius):

        self.radius = radius
        self.gain = gain


    def A(self, state):

        x = state[0]
        y = state[2]

        A = numpy.zeros(shape = (1, 4))
        A[0, 1] = 2 * x
        A[0, 3] = 2 * y

        return A


    def b(self, state):

        x = state[0]
        y = state[2]

        b = - self.gain * (x ** 2 + y ** 2 - self.radius ** 2)

        return numpy.array([b])
