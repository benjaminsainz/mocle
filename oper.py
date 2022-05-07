"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""


from mcla import *
from obj import *

from sklearn.metrics.cluster import adjusted_rand_score


def compute_ari(arguments):
    x, y = arguments
    return adjusted_rand_score(x, y)


def binary_tournament(X, pop, f1, f2):
    pop_size = len(pop)
    i, j = np.random.randint(pop_size), np.random.randint(pop_size)
    if (f2[i] <= f2[j]) and (f1[i] <= f1[j]) and (f2[i] < f2[j]) or (f1[i] < f1[j]):
        return pop[i]
    else:
        return pop[j]


def cross(p1, p2, k_values):
    p1 = list(p1)
    p2 = list(p2)
    p1_n_clusters = len(set(p1))
    p2_n_clusters = len(set(p2))
    parent_ks = [p1_n_clusters, p2_n_clusters]
    parent_range = list(range(min(parent_ks), max(parent_ks)+1, 1))
    N_clusters_max = parent_range[np.random.randint(len(parent_range))]
    offspring = MCLA(np.array([p1, p2]), verbose = 0, N_clusters_max = N_clusters_max)
    return offspring


def select_and_recombine(X, pop, function1_values, function2_values, k_values, temp_pop, pool):
    p1 = binary_tournament(X, pop, function1_values, function2_values)
    p2 = binary_tournament(X, pop, function1_values, function2_values)
    offspring = cross(p1, p2, k_values)
    ari_arguments = [[ind, offspring] for ind in temp_pop]
    rand_indexes_offspring_temp_pop = list(pool.map(compute_ari, ari_arguments))
    offspring_in_temp_pop = max(rand_indexes_offspring_temp_pop)
    return offspring, offspring_in_temp_pop
