#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14:42:19 2015-12-09

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import codecs
import numpy as np
import math
import random


def calc_acc(label_list1, label_list2):
    same = [int(x == y) for x, y in zip(label_list1, label_list2)]
    acc = float(same.count(1)) / len(same)
    return acc


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class LogisticReg:

    def __init__(self, feat_size=0, thrd=0.5):
        """
        Returns an instance of LogisticReg.

        x = LogisticReg(), x is an instance of class LogisticReg.
        feat_size is the feature dimensionalty.
        thrd is the threshold of probability that a sample will be tagged
        as positive class.
        """
        self.label_list = []
        self.sample_list = []
        self.feat_size = feat_size
        self.thrd = thrd
        self.Theta = None

    def printInfo(self):
        print "Samples' size:     ", len(self.sample_list)
        print "Feature dimension: ", self.feat_size
        print "Weight matrix: "
        print self.Theta

    def saveModel(self, path=None, name=None):
        """
        Stores the model under given path.
        """
        if not path:
            print "Using default path(./) to save the model."
            path = r'./'
        else:
            if not os.path.exists(path):
                os.makedirs(path)
                print "Folder doesn\'t exist, automatically create the folder."
            print "Storing model file under folder:", path, '.'

        if not name:
            print "Using default name(./) to save the model."
            name = r'Theta.txt'

        output = codecs.open(path + name, 'w')
        output.write(str(self.feat_size + 1) + '\n')
        for i in range(self.feat_size + 1):
            output.write(str(self.Theta[0, i]) + '\n')
        output.close()

        self.sample_list = []
        self.label_list = []
        self.feat_size = 0
        self.Theta = None

    def loadModel(self, filename=None):
        """
        Loads model from text file.

        The first line in the model file is an positive integer n
        n = feature dimensionalty of model's weight.

        And following n lines is the value of each dimension by
        increase order.
        """
        if not filename:
            print "Not give any model file!"
            print "Trying to read default model file: ./Theta.txt"
            filename = r"./Theta.txt"

        try:
            inputs = codecs.open(filename, 'r')
            first = True
            cnt = 0
            for line in inputs.readlines():
                if first:
                    self.feat_size = int(line.strip()) - 1
                    first = False
                    self.Theta = np.zeros((1, self.feat_size + 1))
                else:
                    self.Theta[0, cnt] = float(line.strip())
                    cnt += 1
            inputs.close()
            return True
        except IOError:
            print "Error: File %s doesn't exist!" % filename
            return False

    def initTheta(self):
        """
        Initializes the Theta vector.

        If the dimension of featrue is 0, this function will do nothing and
        returns False. Otherwise, returns True.
        """
        if self.feat_size != 0:
            self.Theta = np.zeros((1, self.feat_size + 1))
            return True
        else:
            print "Error: The dimension of feature can not be ZERO!"
            return False

    def setFeatSize(self, size=0):
        """
        Sets feature dimensions by the given size.
        """
        if size == 0:
            print "Warning: ZERO dimensions of feature will be set!"
            print "         This would causes some trouble unpredictable!"
            print "         Please make sure the dimension of feature is 0!"
        self.feat_size = size

    def __getSampleVec(self, sample):
        """
        Returns a row vector by 1*(n+1).
        """
        sample_vec = np.zeros((1, self.feat_size + 1))
        for i in sample.keys():
            sample_vec[0][i] = sample[i]
        sample_vec[0][0] = 1.0

        return sample_vec

    def predict(self, sample_vec):
        """
        Returns the predict vector of probabilities.
        """
        X = sample_vec.T
        # print np.dot(self.Theta[0, :], X)[0]
        pred = sigmoid(np.dot(self.Theta[0, :], X)[0])
        return pred

    def train_batch(self, max_iter=200, learn_rate=0.1, delta=1e-3):
        """
        Training a logistic regression model using Gradient descent method.
        the samples and labels should be already assigned to field
        self.sample_list and self.label_list.

        max_iter: the maximum number of iteration(default 200).
        learn_rate: the learning rate of train process(default 1.0).
        delta: the threshold of cost function value(default 0.001).
        """
        print '-' * 60
        print "START TRAIN (Gradient descent):"

        theta_rd = []
        mle_rd = []

        # training process
        n = len(self.label_list)
        mle_pre, mle = 0., 0.
        rd = 0
        while rd < max_iter:
            omega = np.zeros((1, self.feat_size + 1))
            error = 0
            for i in range(n):
                y = self.label_list[i]
                hx = self.predict(self.sample_list[i])
                pred = 1 if hx > self.thrd else 0
                if y != pred:
                    error += 1
                # print hx
                mle += y * math.log(hx) + (1 - y) * math.log(1 - hx)
                omega += learn_rate * (y - hx) * self.sample_list[i]

            acc = 1 - float(error) / n
            mle /= n
            print "Iter %4d    MLE:%4.4f    Acc:%.4f" % (rd, mle, acc)
            a = []
            for nn in self.Theta[0]:
                a.append(nn)
            theta_rd.append(a)
            mle_rd.append(mle)
            if rd != 0 and (mle - mle_pre) < delta and mle >= mle_pre:
                print "\n\nReach the minimal cost value threshold!"
                break
            mle_pre = mle
            self.Theta += omega
            rd += 1

        if rd == max_iter:
            print "Train loop has reached the maximum of iteration."
        print "Training process finished."
        return [theta_rd, mle_rd]

    def train_sgd(self, max_iter=200, learn_rate=1.0, delta=1e-3):
        """
        Training a logistic regression model using Stochastic Gradient descent
        method.
        the samples and labels should be already assigned to field
        self.sample_list and self.label_list.

        max_iter: the maximum number of iteration(default 200).
        learn_rate: the learning rate of train process(default 1.0).
        delta: the threshold of cost function value(default 0.001).
        """
        print '-' * 60
        print "START TRAIN (Stochastic Gradient descent):"

        theta_rd = []
        mle_rd = []

        # training process
        m = len(self.label_list)
        mle_pre, mle = 0., 0.
        rd = 0
        while rd < max_iter * m:
            if rd % m == 0 and rd != 0:
                loop = rd / m
                error = 0
                mle = 0.
                for i in range(m):
                    y = self.label_list[i]
                    hx = self.predict(self.sample_list[i])
                    pred = 1 if hx > self.thrd else 0
                    if y != pred:
                        error += 1
                    mle += y * math.log(hx) + (1 - y) * math.log(1 - hx)

                acc = 1 - float(error) / m
                mle /= m
                print "Iter %4d    MLE:%4.4f    Acc:%.4f" % (loop, mle, acc)
                a = []
                for nn in self.Theta[0]:
                    a.append(nn)
                theta_rd.append(a)
                mle_rd.append(mle)
                if loop != 0 and (mle - mle_pre) < delta and mle >= mle_pre:
                    print "\n\nReach the minimal cost value threshold!"
                    break
                mle_pre = mle

            i = random.randint(0, m - 1)
            y = self.label_list[i]
            hx = self.predict(self.sample_list[i])
            pred = 1 if hx > self.thrd else 0
            self.Theta += learn_rate * (y - hx) * self.sample_list[i]
            rd += 1

        if rd == max_iter * m:
            print "Train loop has reached the maximum of iteration."
        print "Training process finished."
        return [theta_rd, mle_rd]

    def train_newton(self, max_iter=100, thrd=1e-5):
        """
        Training a logistic regression model using Newton's method
        the samples and labels should be already assigned to field
        self.sample_list and self.label_list.
        """

        n = len(self.sample_list)
        m = self.feat_size
        theta_rd = []
        mle_rd = []
        rd = 0
        cost = 0.
        cost_pre = 0.
        while rd < max_iter:
            cost = 0.
            error = 0
            grad = np.zeros((1, m + 1))
            H = np.zeros((m + 1, m + 1))
            for i in range(n):
                x = self.sample_list[i]
                y = self.label_list[i]
                hx = self.predict(self.sample_list[i])
                pred = 1 if hx > self.thrd else 0
                if y != pred:
                    error += 1
                cost -= y * math.log(hx) + (1 - y) * math.log(1 - hx)
                grad += (hx - y) * x
                H += hx * (1 - hx) * np.dot(x.T, x)
            cost /= n
            grad /= n
            H /= n
            acc = 1 - float(error) / n
            print "Iter %4d    Cost:%4.8f    Acc:%.4f" % (rd, cost, acc)
            a = []
            for nn in self.Theta[0]:
                a.append(nn)
            theta_rd.append(a)
            mle_rd.append(cost)
            if rd != 0 and (cost_pre - cost) < thrd and cost_pre >= cost:
                print "\n\nReach the minimal cost value threshold!"
                break
            self.Theta -= np.dot(np.mat(H).I, grad.T).T
            cost_pre = cost
            rd += 1

        if rd == max_iter * m:
            print "Train loop has reached the maximum of iteration."
        print "Training process finished."
        return [theta_rd, mle_rd]

    def classify(self, sample_test):
        """Classify the sample_test, returns the most likely label."""
        X = self.__getSampleVec(sample_test)
        prb = self.predict(X)
        label = 1 if prb > self.thrd else 0
        return label

    def batch_classify(self, sample_test_list):
        """
        Doing classification for a list of sample.

        Returns a list of predicted label for each test sample.
        """
        labels = []
        for sample in sample_test_list:
            labels.append(self.classify(sample))
        return labels

    def read_train_file(self, filepath):
        """
        Make traing set from file.
        Returns sample_set, label_set
        """
        data = codecs.open(filepath, 'r')
        max_index = 0
        for line in data.readlines():
            val = line.strip().split('\t')
            self.label_list.append(int(val[0]))
            sample_vec = {}
            val = val[-1].split(" ")
            for i in range(len(val)):
                [index, value] = val[i].split(':')
                sample_vec[int(index)] = float(value)
                if int(index) > max_index:
                    max_index = int(index)
            self.sample_list.append(self.__getSampleVec(sample_vec))
        self.setFeatSize(max_index)

    def loadSamples(self, samples=[], labels=[]):
        """
        Loads samples' and labels' list from data.
        """
        max_index = 0
        self.label_list = labels
        self.sample_list = []
        for samp in samples:
            size = max(samp.keys())
            if size > max_index:
                max_index = size
            self.sample_list.append(self.__getSampleVec(samp))
        return max_index


def make_data(filepath):
    """
    Makes sample list and label list from file.
    Returns a tuple of sample list and label list.
    """
    samples = []
    labels = []
    data = codecs.open(filepath, 'r')
    for line in data.readlines():
        val = line.strip().split('\t')
        labels.append(int(val[0]))
        sample_vec = {}
        val = val[-1].split(" ")
        for i in range(0, len(val)):
            [index, value] = val[i].split(':')
            sample_vec[int(index)] = float(value)
        samples.append(sample_vec)
    return samples, labels


def create(sample_list, label_list, feat_size=1, thrd=0.5):
    """
    Creates an instance of LogisticReg with given parameters.

    label_list: list of the samples' label
    sample_list: list of samples
    thrd: threshold of probability that a sample will be tagged as a positive
          class.
    """
    tmp = LogisticReg(feat_size=feat_size, thrd=thrd)
    tag = True
    if tmp.initTheta():
        # initialization successed
        k = tmp.loadSamples(sample_list, label_list)
        if k > feat_size:
            print "Error: feature dimensionalty not matching!"
            print "Feature dimensionalty you give:  %d" % feat_size
            print "Feature dimensionalty from data: %d" % k
            tag = False
        else:
            print "Initialization successed!"
    else:
        # not successed
        tag = False
        print "Initialization failed! Please checked your parameters."
    return (tmp, tag)
