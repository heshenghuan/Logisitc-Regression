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

    def __init__(self):
        """
        Returns an instance of LogisticReg with all field empty.

        x = LogisticReg(), x is an instance of class LogisticReg.
        """
        self.label_list = []
        self.sample_list = []
        self.feat_dimension = 0
        self.Theta = None

    def printInfo(self):
        print "Samples' size:     ", len(self.sample_list)
        print "Feature dimension: ", self.feat_dimension
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
        output.write(str(self.feat_dimension + 1) + '\n')
        for i in range(self.feat_dimension + 1):
            output.write(str(self.Theta[0, i]) + '\n')
        output.close()

        self.sample_list = []
        self.label_list = []
        self.feat_dimension = 0
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
                    self.feat_dimension = int(line.strip()) - 1
                    first = False
                    self.Theta = np.zeros((1, self.feat_dimension + 1))
                else:
                    self.Theta[0, cnt] = float(line.strip())
                    cnt += 1
            inputs.close()
            return True
        except IOError:
            print "Error: File %s doesn't exist!" % filename
            return False
