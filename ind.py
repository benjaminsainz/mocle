"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import numpy as np

from snn import *


def takesecond(elem):
    return elem[1]


def km_agg_cluster(args):
    n, X = args
    pop = []
    km_model = KMeans(n, init='random', n_init=30).fit(X)  # k-means model
    pop.append(km_model.labels_)
    al_model = AgglomerativeClustering(n, linkage='average', affinity='euclidean').fit(X)  # average link model
    pop.append(al_model.labels_)
    sl_model = AgglomerativeClustering(n, linkage='single', affinity='euclidean').fit(X)  # single link model
    pop.append(sl_model.labels_)
    return pop


def snn_cluster(args):
    i, j, X, kmin, kmax = args
    pop = []
    snn_model = SNN(neighbor_num=i, min_shared_neighbor_proportion=j).fit(np.array(X))  # snn model
    if (len(set(snn_model.labels_)) >= kmin) and len(set(snn_model.labels_)) <= kmax:
        pop.append(snn_model.labels_ + 1)
    return pop


def avoid_singleton_clusters(pop):
    no_singletons_pop = []
    for ind in pop:
        ind = list(ind)
        k_values_ind = set(ind)
        singleton_flag = False
        for j in k_values_ind:
            if ind.count(j) == 1:
                singleton_flag == True
        if singleton_flag == False:
            no_singletons_pop.append(ind)
    return no_singletons_pop


def create_pop_with_k(n_clusters, X, pool):
    kmin = n_clusters
    kmax = 2*n_clusters
    k_values = list(range(kmin, kmax+1))
    km_agg_args = [[k, X] for k in k_values]
    parallel_km_agg_pop = list(pool.map(km_agg_cluster, km_agg_args))
    km_agg_pop = [ind for n_pop in parallel_km_agg_pop for ind in n_pop]
    snn_args = [[int(.01*len(X)), 0.5, X, kmin, kmax]]
    snn_parallel_pop = list(pool.map(snn_cluster, snn_args))
    snn_pop = [ind for n_pop in snn_parallel_pop for ind in n_pop]
    pop = km_agg_pop + snn_pop
    no_singletons_pop = avoid_singleton_clusters(pop)
    return no_singletons_pop, k_values


def initial_pop(X, n_clusters, pool, max_gens):
    print('Generating initial population...')
    pop, k = create_pop_with_k(n_clusters, X, pool)
    print('Population size: {}. Generations: {}'.format(len(pop), max_gens))
    return pop, k, len(pop)
