"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""


from oper import *
from obj import *
from ind import *
from retr import *


from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import glob
import math
import time
import multiprocessing
import os
import warnings


def index_of(a, lst):
    for i in range(len(lst)):
        if lst[i] == a:
            return i
    return -1


def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def fast_non_dominated_sort(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(0, len(values1))]
    rank = [0 for _ in range(0, len(values1))]
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


def crowding_distance(arguments):
    values1, values2, front = arguments
    distance = [0 for _ in range(len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance


def DataFrame_preparation(fronts, data, solutions, n_clusters, X, pop_size, neg_function1, neg_function2, mocle_time, max_gens, ari, pop):
    d = dict()
    d['Dataset'] = [data]*solutions
    d['Algorithm'] = ['mocle']*solutions
    d['Clusters'] = [n_clusters]*solutions
    d['Instances'] = [len(X)]*solutions
    d['Features'] = [len(X[0])]*solutions
    d['Population size'] = [pop_size]*solutions
    d['Max. gens'] = [max_gens]*solutions
    d['No. objectives'] = [2]*solutions
    d['Obj. 1 name'] = ['deviation']*solutions
    d['Objective 1'] = neg_function1
    d['Obj. 2 name'] = ['connectivity']*solutions
    d['Objective 2'] = neg_function2
    d['Time'] = [mocle_time]*solutions
    d['Adjusted Rand Index'] = ari
    for i in range(len(pop[0])):
        d['X{}'.format(i+1)] = list()
    for solution in fronts[0]:
        for gene in range(len(pop[solution])):
            d['X{}'.format(gene + 1)].append(str(pop[solution][gene]))
    out = pd.DataFrame(d)
    return out

def result_export(data, n_clusters, pop_size, max_gens, run, mocle_time, out, runs):
    if not os.path.exists('mocle-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens)):
        os.makedirs('mocle-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens))
    out.to_csv('mocle-out/{}_{}_{}_{}/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens, data, n_clusters, pop_size, max_gens, run))
    final_df = pd.DataFrame()
    names = glob.glob("mocle-out/{}_{}_{}_{}/solution*".format(data, n_clusters, pop_size, max_gens))
    for name in names:
        temp_df = pd.read_csv(name)
        final_df = pd.concat([final_df, temp_df.sort_values('Adjusted Rand Index', ascending=False).iloc[:1, :]])
    final_df.reset_index(inplace=True, drop=True)
    final_df.iloc[:, 1:].to_csv('mocle-out/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens, runs))
    print('Clustering finished. Runtime: {}.'.format(time.strftime('%H:%M:%S', time.gmtime(mocle_time))))


def process_end_metrics(function1_values, function2_values, fronts, start, y, pop, pool):
    print('Exporting results...')
    neg_function1 = [-1 * function1_values[solution] for solution in fronts[0]]
    neg_function2 = [-1 * function2_values[solution] for solution in fronts[0]]
    mocle_time = time.time() - start
    if y is None:
        ari = [np.nan for _ in fronts[0]]
    else:
        ari = []
        for solution in fronts[0]:
            ari.append(compute_ari([y, pop[solution]]))
    return neg_function1, neg_function2, mocle_time, ari


def crowding_distance_cut(temp_fronts, temp_crowding_distance_values, pop_size):
    new_pop = []
    for i in range(len(temp_fronts)):
        front_indeces = [index_of(temp_fronts[i][j], temp_fronts[i]) for j in range(len(temp_fronts[i]))]
        sorted_front_indeces = sort_by_values(front_indeces, temp_crowding_distance_values[i])
        sorted_front = [temp_fronts[i][sorted_front_indeces[j]] for j in range(len(temp_fronts[i]))]
        sorted_front.reverse()
        for value in sorted_front:
            if value not in new_pop:
                new_pop.append(value)
            if len(new_pop) == pop_size:
                break
        if len(new_pop) == pop_size:
            break
    return new_pop


def temporal_population_metrics(X, temp_pop, pop_size, function1_values, function2_values, pool, nn_dict):
    print('Computing fitness and crowding distances of the temporal population.')
    obj_args = [[X, temp_pop[i], nn_dict] for i in range(pop_size, 2*pop_size)]
    parallel_f1_values = list(pool.map(dev, obj_args))
    parallel_f2_values = list(pool.map(conn, obj_args))
    temp_function1_values = function1_values + parallel_f1_values
    temp_function2_values = function2_values + parallel_f2_values
    temp_fronts = fast_non_dominated_sort(temp_function1_values, temp_function2_values)
    crowding_distance_arguments = []
    for i in range(len(temp_fronts)):
        crowding_distance_arguments.append([temp_function1_values, temp_function2_values, temp_fronts[i]])  
    temp_crowding_distance_values = list(pool.map(crowding_distance, crowding_distance_arguments))
    print('Temporal Population Size: {}. Pareto Front Size: {}'.format(len(temp_pop), len(temp_fronts[0])))
    return temp_fronts, temp_crowding_distance_values, temp_function1_values, temp_function2_values


def temporal_population_generator(pop, pop_size, X, function1_values, function2_values, k_values, pool):
    print('Creating temporal population...')
    temp_pop = pop.copy()
    while len(temp_pop) != 2*pop_size:
        offspring = np.array([0])
        offspring_in_temp_pop = 1
        while offspring_in_temp_pop == 1 or len(set(offspring)) < 2:
            offspring, offspring_in_temp_pop = select_and_recombine(X, pop, function1_values, function2_values, k_values, temp_pop, pool)
        temp_pop.append(offspring)
    return temp_pop


def evolutionary_process(max_gens, pop, pop_size, X, function1_values, function2_values, k_values, pool, start, nn_dict):
    gen = 1
    while gen <= max_gens:
        print('=========================== Generation {} ==========================='.format(gen))
        temp_pop = temporal_population_generator(pop, pop_size, X, function1_values, function2_values, k_values, pool)
        temp_fronts, temp_crowding_distance_values, temp_function1_values, temp_function2_values = temporal_population_metrics(X, temp_pop, pop_size, function1_values, function2_values, pool, nn_dict)
        new_pop = crowding_distance_cut(temp_fronts, temp_crowding_distance_values, pop_size)
        pop = [temp_pop[i] for i in new_pop]
        function1_values = [temp_function1_values[i] for i in new_pop]
        function2_values = [temp_function2_values[i] for i in new_pop]
        gen = gen + 1
        print('Elapsed time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start))))   
    return temp_function1_values, temp_function2_values, temp_fronts, temp_pop


def print_initialization(run, data, n_clusters):
    print('================================== TEST {} ================================='.format(run))
    print('Clustering {} into {} clusters using MOCLE'.format(data, n_clusters))


def run_mocle(data, n_clusters, runs=10, max_gens=50):
    data, n_clusters, X, y = retrieval(data, n_clusters)
    for run in range(1, runs+1):
        print_initialization(run, data, n_clusters)
        start = time.time()
        pool = multiprocessing.Pool()
        init_pop, k_values, pop_size = initial_pop(X, n_clusters, pool, max_gens)
        nn_dict = pre_nn_computing(X)
        objective_arguments = [[X, init_pop[i], nn_dict] for i in range(pop_size)]
        init_f1_values = list(pool.map(dev, objective_arguments))
        init_f2_values = list(pool.map(conn, objective_arguments))
        last_f1_values, last_f2_values, last_fronts, last_temp_pop = evolutionary_process(max_gens, init_pop, pop_size, X, init_f1_values, init_f2_values, k_values, pool, start, nn_dict)
        neg_function1, neg_function2, mocle_time, ari = process_end_metrics(last_f1_values, last_f2_values, last_fronts, start, y, last_temp_pop, pool)
        out = DataFrame_preparation(last_fronts, data, len(last_fronts[0]), n_clusters, X, pop_size, neg_function1, neg_function2, mocle_time, max_gens, ari, last_temp_pop)
        result_export(data, n_clusters, pop_size, max_gens, run, mocle_time, out, runs)

