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
    return 1 / (1 + math.exp(-x / 5000))


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
        sample_vec = np.zeros((1, self.feat_dimension + 1))
        for i in sample.keys():
            sample_vec[0][i] = sample[i]

        return sample_vec

    def predict(self, sample_vec):
        """
        Returns the predict vector of probabilities.
        """
        X = sample_vec.T
        pred = np.dot(self.Theta[j, :], X)[0]
        return pred

    def train_batch(self, max_iter=200, learn_rate=1.0, delta=1e-3):
        """
        Training a logistic regression model, the samples and labels should be
        already assigned to field self.sample_list and self.label_list.

        max_iter: the maximum number of iteration(default 200).
        learn_rate: the learning rate of train process(default 1.0).
        delta: the threshold of cost function value(default 0.001).
        """
        print '-' * 60
        print "START TRAIN BATCH:"

        # training process
        n = len(self.label_list)
        mle_pre, mle = 0.
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
                mle += y * math.log(hx) + (1 - y) * math.log(1 - hx)
                omega += learn_rate * (y - hx) * self.sample_list[i]

            acc = 1 - float(error) / n
            print "Iter %4d    MLE:%4.4f    Acc:%.4f" % (rd, mle, acc)

            if rd != 0 and (mle - mle_pre) < delta and mle >= mle_pre:
                print "\n\nReach the minimal cost value threshold!"
            mle_pre = mle
            rd += 1

        if rd == max_iter:
            print "Train loop has reached the maximum of iteration."
        print "Training process finished."

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
        labels.append(val[0])
        sample_vec = {}
        val = val[-1].split(" ")
        for i in range(0, len(val)):
            [index, value] = val[i].split(':')
            sample_vec[int(index)] = float(value)
        samples.append(sample_vec)
    return samples, labels


def create(label_list, sample_list, thrd=0.5):
    """
    Creates an instance of LogisticReg with given parameters.

    label_list: list of the samples' label
    sample_list: list of samples
    thrd: threshold of probability that a sample will be tagged as a positive
          class.
    """
    tmp = LogisticReg(thrd=thrd)
    tag = True
    if tmp.loadFeatSize(size, classNum):
        # initialization successed
        print "Initialization successed!"
    else:
        # not successed
        tag = False
        print "Initialization failed! Please checked your parameters."
    return (tmp, tag)
