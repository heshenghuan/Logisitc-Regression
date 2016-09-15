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
    t0 = theta[0]
    t1 = theta[1]
    t2 = theta[2]
    y = []
    if t2 == 0:
        # In the 1st round, theta is [0, 0, 0]
        return [0]*len(x)
    for a in x:
        y.append(-(t1 * a + t0) / t2)
    return y


def show_graph(fig, axarr, x0, x1, theta, cost):
    length = len(theta)
    test_x = [i for i in range(20, 60, 5)]

    for i in range(length):
        test_y = plotTheta(theta[i], test_x)
        axarr[0].cla()
        axarr[0].set_xlim([15, 65])
        axarr[0].set_ylim([40, 90])
        axarr[0].plot(x0[0], x1[0], 'o')
        axarr[0].plot(x0[1], x1[1], 'r+')
        axarr[0].plot(test_x, test_y, 'g')
        axarr[0].set_title('Logistic Regression (Newton Method)')
        axarr[1].set_title('Logistic Regression Cost value')
        axarr[1].scatter(i + 1, cost[i], color='r')
        axarr[1].set_xlim([1, length + 1])
        axarr[1].set_ylim([0.4, 0.7])
        fig.canvas.draw()

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

    case, tag = lr.create(x, y, 2, 0.5)
    theta, mle = case.train_newton(max_iter=200, thrd=1e-6)

    fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
    # fig.title('Stochastic Gradient Ascent (Online Updating)')
    fig.show()
    show_graph(fig, axarr, x0, x1, theta, mle)
    raw_input()
