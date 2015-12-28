#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12:54:43 2015-12-12

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import matplotlib.pyplot as plt
import LogisticReg as lr
import codecs


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
    x1 = {1: [], 0: []}
    x0 = {1: [], 0: []}
    for i in range(len(x)):
        if y[i] == 1:
            x0[1].append(x[i][1])
            x1[1].append(x[i][2])
        else:
            x0[0].append(x[i][1])
            x1[0].append(x[i][2])

    test_x = [i for i in range(20, 60, 5)]
    plt.plot(x0[0], x1[0], 'o')
    plt.plot(x0[1], x1[1], 'r+')
    plt.title('Newton Method')

    case, tag = lr.create(x, y, 2, 0.5)
    case.train_newton()

    test_y = plotTheta(case.Theta, test_x)
    plt.plot(test_x, test_y, 'r')
    plt.show()
