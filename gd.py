#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 20:46:58 2015-12-28

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import matplotlib.pyplot as plt
import LogisticReg as lr
import codecs


def scale(x):
    ave = {1: 0., 2: 0.}
    dist = {1: 0., 2: 0.}
    maximum = {1: 0., 2: 0.}
    minimum = {1: 1e30, 2: 1e30}
    for s in x:
        for i in s.keys():
            ave[i] += s[i]
            if maximum[i] < s[i]:
                maximum[i] = s[i]
            if minimum[i] > s[i]:
                minimum[i] = s[i]
    ave[1] /= len(x)
    ave[2] /= len(x)
    dist[1] = maximum[1] - minimum[1]
    dist[2] = maximum[2] - minimum[2]
    r = []
    for s in x:
        vec = {}
        for i in s.keys():
            vec[i] = (s[i] - ave[i]) / dist[i]
        r.append(vec)
    return r


def plotTheta(theta, x):
    t0 = theta[0][0]
    t1 = theta[0][1]
    t2 = theta[0][2]
    y = []
    for a in x:
        y.append(-(t1 * a + t0) / t2)
    return y

if __name__ == '__main__':
    x, y = lr.make_data(r"ex4.dat")
    x = scale(x)
    x1 = {1: [], 0: []}
    x0 = {1: [], 0: []}
    for i in range(len(x)):
        if y[i] == 1:
            x0[1].append(x[i][1])
            x1[1].append(x[i][2])
        else:
            x0[0].append(x[i][1])
            x1[0].append(x[i][2])

    test_x = [-0.5 + i * 0.1 for i in range(10)]
    plt.plot(x0[0], x1[0], 'o')
    plt.plot(x0[1], x1[1], 'r+')
    plt.title('Gradient Ascent (Batch Updating)')

    case, tag = lr.create(x, y, 2, 0.5)
    case.train_batch(max_iter=200, learn_rate=0.1, delta=1e-3)

    test_y = plotTheta(case.Theta, test_x)
    plt.plot(test_x, test_y, 'g')
    plt.show()
