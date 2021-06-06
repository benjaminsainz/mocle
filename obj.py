#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:47:00 2021

@author: benjamin
"""

import numpy as np


def cents(X, ind):
    centroids = dict()
    for i in set(ind):
        centroids['cluster_{}'.format(i)] = []
        for j in range(len(ind)):
            if ind[j] == i:
                centroids['cluster_{}'.format(i)].append(X[j])
        centroids['cluster_{}'.format(i)] = np.sum(centroids['cluster_{}'.format(i)],
                                                   axis=0)/len(centroids['cluster_{}'.format(i)])
    return centroids


def dev(X, ind):
    centroids = cents(X, ind)
    distances = []
    for i in range(len(ind)):
        for j in set(ind):
            if ind[i] == j:
                distances.append(np.linalg.norm(X[i] - centroids['cluster_{}'.format(j)]))
    return -sum(distances)


def takesecond(elem):
    return elem[1]


def conn(X, ind, nnp=0.05):
    connectivity = 0
    k = int(len(X) * nnp)
    for i in range(len(X)):
        distances = []
        for j in range(len(X)):
            if i != j:
                distances.append((j, np.linalg.norm(X[i] - X[j])))
        distances.sort(key=takesecond, reverse=False)
        nn = distances[:k]
        nn_index = []
        for n in nn:
            nn_index.append(n[0])
        nn_index = enumerate(nn_index, start=1)
        for n in nn_index:
            if ind[i] != ind[n[1]]:
                connectivity += 1 / n[0]
    return -connectivity
