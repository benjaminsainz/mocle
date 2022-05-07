"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""


import numpy as np


def cents(X, ind):
    centroids = dict()
    for i in set(ind):
        centroids['cluster_{}'.format(i)] = []
        for j in range(len(ind)):
            if ind[j] == i:
                centroids['cluster_{}'.format(i)].append(X[j])
        centroids['cluster_{}'.format(i)] = np.sum(centroids['cluster_{}'.format(i)], axis=0)/len(centroids['cluster_{}'.format(i)])
    return centroids


def dev(arguments):
    X, ind, _ = arguments
    centroids = cents(X, ind)
    distances = []
    for i in range(len(ind)):
        for j in set(ind):
            if ind[i] == j:
                distances.append(np.linalg.norm(X[i] - centroids['cluster_{}'.format(j)]))
    return -sum(distances)


def takesecond(elem):
    return elem[1]


def pre_nn_computing(X):
    nnp=0.05
    nn_dict = dict()
    k = int(len(X) * nnp)
    for i in range(len(X)):
        distances = []
        for j in range(len(X)):
            if i != j:
                distances.append((j, np.linalg.norm(X[i] - X[j])))
        distances.sort(key=takesecond, reverse=False)
        nn_dict[i] = distances[:k]
    return nn_dict


def conn(arguments):
    X, ind, nn_dict = arguments
    connectivity = 0
    for i in range(len(X)):
        nn_index = []
        for n in nn_dict[i]:
            nn_index.append(n[0])
        nn_index = enumerate(nn_index, start=1)
        for n in nn_index:
            if ind[i] != ind[n[1]]:
                connectivity += 1 / n[0]
    return -connectivity

