import numpy


class MeasurementModel():

    def __init__(self, number_components, covariance):

        sub_d = 2

        self._H = numpy.zeros(shape = (number_components, sub_d * number_components))
        for i in range(number_components):
            self._H[i, i * sub_d] = 1.0

        self._R = numpy.eye(number_components) * covariance


    def H(self):

        return self._H


    def R(self):

        return self._R
