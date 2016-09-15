#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 20:56:49 2015-12-28

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import matplotlib.pyplot as plt
import LogisticReg as lr
import codecs
import sys


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
    t0 = theta[0]
    t1 = theta[1]
    t2 = theta[2]
    y = []
    for a in x:
        y.append(-(t1 * a + t0) / t2)
    print y
    return y


def show_graph(fig, axarr, x0, x1, theta, cost):
    length = len(theta)
    test_x = [-0.5 + i * 0.1 for i in range(10)]

    for i in range(length):
        test_y = plotTheta(theta[i], test_x)
        axarr[0].cla()
        axarr[0].set_xlim([-0.6, 0.6])
        axarr[0].set_ylim([-0.6, 0.6])
        axarr[0].plot(x0[0], x1[0], 'o')
        axarr[0].plot(x0[1], x1[1], 'r+')
        axarr[0].plot(test_x, test_y, 'g')
        axarr[0].set_title('Logistic Regression (SGD)')
        axarr[1].set_title('Logistic Regression MLE value')
        axarr[1].scatter(i + 1, cost[i], color='r')
        axarr[1].set_xlim([1, length + 1])
        axarr[1].set_ylim([-0.8, -0.4])
        fig.canvas.draw()
       # plt.show()


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

    # test_x = [-0.5 + i * 0.1 for i in range(10)]
    # plt.plot(x0[0], x1[0], 'o')
    # plt.plot(x0[1], x1[1], 'r+')
    # plt.title('Stochastic Gradient Ascent (Online Updating)')

    case, tag = lr.create(x, y, 2, 0.5)
    theta, mle = case.train_sgd(max_iter=1000, learn_rate=0.1, delta=1e-4)

    # test_y = plotTheta(case.Theta, test_x)
    # plt.plot(test_x, test_y, 'b')
    # plt.show()
    fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
    # fig.title('Stochastic Gradient Ascent (Online Updating)')
    fig.show()
    show_graph(fig, axarr, x0, x1, theta, mle)
    raw_input()
