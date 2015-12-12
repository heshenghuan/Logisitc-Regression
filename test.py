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


x, y = lr.make_data(r"ex4.dat")
# x = scale(x)
case, tag = lr.create(x, y, 2, 0.5)
# case.train_batch(max_iter=200, learn_rate=0.1, delta=1e-3)
# case.train_sgd(max_iter=1000, learn_rate=0.1, delta=1e-4)
case.train_newton()
