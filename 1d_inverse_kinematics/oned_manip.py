import numpy

class Manipulator():

    def __init__(self, length):

        self.length = length


    def fk(self, q):

        x = numpy.cos(q) * self.length
        y = numpy.sin(q) * self.length

        position = numpy.zeros(shape = (2, 1))
        position[0, 0] = x
        position[1, 0] = y

        return position


    def J(self, q):

        x_J = -self.length * numpy.sin(q)
        y_J = self.length * numpy.cos(q)

        jacobian = numpy.zeros(shape = (2, 1))
        jacobian[0, 0] = x_J
        jacobian[1, 0] = y_J

        return jacobian
