#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:58:28 2021

@author: benjamin
"""

from mcla import *
from obj import *


def binary_tournament(X, pop, f1, f2):
    pop_size = len(pop)
    i, j = np.random.randint(pop_size), np.random.randint(pop_size)
    """
    while j == i:
        j = np.random.randint(pop_size)
    if ((conn(X, pop[i]) <= conn(X, pop[j])) and (dev(X, pop[i]) <= dev(X, pop[j]))) \
            and ((conn(X, pop[i]) < conn(X, pop[j])) or (dev(X, pop[i]) < dev(X, pop[j]))):
        return pop[i]
    else:
        return pop[j]
    """
    if (f2[i] <= f2[j]) and (f1[i] <= f1[j]) and (f2[i] < f2[j]) or (f1[i] < f1[j]):
        return pop[i]
    else:
        return pop[j]


def cross(p1, p2, k, k_range):
    if k_range is True:
        n = np.random.randint(min(list(k)), max(list(k)))
    else:
        n = k
    offspring = MCLA(np.array([p1, p2]), n)
    return offspring
