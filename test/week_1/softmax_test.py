import numpy
from week_1.softmax import softmax
from unittest import TestCase

__author__ = 'Georgi Tsatsev <gtsatsev@domotz.com>'


class TestSoftmax(TestCase):
    def setUp(self):
        self.scores_1d = [1.0, 2.0, 3.0]
        self.scores_3d = numpy.array([[1, 2, 3, 6],
                                      [2, 4, 5, 6],
                                      [3, 8, 7, 6]])
        self.testing = numpy.testing

    @staticmethod
    def _expected_1d():
        return [0.090030573170380462, 0.24472847105479767, 0.6652409557748219]

    @staticmethod
    def _expected_3d():
        return [[0.09003057317038046, 0.002428258029591337, 0.01587623997646677, 0.33333333333333337],
                [0.24472847105479767, 0.017942534803329194, 0.11731042782619837, 0.33333333333333337],
                [0.6652409557748219, 0.9796292071670795, 0.8668133321973348, 0.33333333333333337]]

    def test_softmax_1d(self):
        result = softmax(self.scores_1d)
        self.assertListEqual(result.tolist(), self._expected_1d())

    def test_softmax_3d(self):
        result = softmax(self.scores_3d)
        self.assertListEqual(result.tolist(), self._expected_3d())
