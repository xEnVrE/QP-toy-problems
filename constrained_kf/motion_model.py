import numpy


class MotionModel():

    def __init__(self, sample_time, number_components, psd):

        sub_d = 2

        # A matrix s.t. x_k = A x_{k-1}
        A_block = numpy.zeros(shape = (sub_d, sub_d))
        A_block[0, 0] = 1.0
        A_block[0, 1] = sample_time
        A_block[1, 1] = 1.0

        self._A = numpy.zeros(shape = (sub_d * number_components, sub_d * number_components))
        for i in range(number_components):
            self._A[i * sub_d : (i + 1) * sub_d, i * sub_d : (i + 1) * sub_d] = A_block

        # discrete process covariance matrix Q
        Q_block = numpy.zeros(shape = A_block.shape)
        Q_block[0, 0] = (sample_time ** 3) / 3.0 * psd
        Q_block[0, 1] = (sample_time ** 2) / 2.0 * psd
        Q_block[1, 0] = Q_block[0, 1]
        Q_block[1, 1] = sample_time * psd

        self._Q = numpy.zeros(shape = self._A.shape)
        for i in range(number_components):
            self._Q[i * sub_d : (i + 1) * sub_d, i * sub_d : (i + 1) * sub_d] = Q_block


    def A(self):

        return self._A


    def Q(self):

        return self._Q
