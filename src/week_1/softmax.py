from __future__ import division
import numpy as np
import matplotlib.pyplot as plot

__author__ = 'Georgi Tsatsev <gtsatsev@domotz.com>'

'''
This file contains an example of th Softmax function,
 ... essentially a normalization function
'''


def softmax(scores_array):
    if type(scores_array) == list or len(scores_array.shape) == 1:
        return np.array([np.exp(score)/sum(np.exp(scores_array)) for score in scores_array])

    probability_array = np.zeros(scores_array.shape)
    for i in range(scores_array.shape[1]):
        probability_array[:, i] = np.exp(scores_array[:, i])/sum(np.exp(scores_array[:, i]))
    return probability_array

x = np.arange(-0.2, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])


plot.plot(x, softmax(scores).T, linewidth=2)
plot.show()

