import numpy as np
import matplotlib.pyplot as plot

__author__ = 'Georgi Tsatsev <gtsatsev@domotz.com>'

'''
This file contains an example of th Softmax function,
 ... essentially a normalization function
'''


def softmax(scores):
    total = sum(scores)
    probability_list = [score/total for score in scores]
    return np.array(probability_list)


scores = [3.0, 2.0, 1.0]
print softmax(scores)
x = np.arange(-0.2, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])


plot.plot(x, softmax(scores).T, linewidth=2)
plot.show()

