import math
from oper import *
from obj import *
from ind import *
import time
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob


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
            if (values1[p] > values1[q] and values2[p] > values2[q]) \
                    or (values1[p] >= values1[q] and values2[p] > values2[q]) \
                    or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) \
                    or (values1[q] >= values1[p] and values2[q] > values2[p]) \
                    or (values1[q] > values1[p] and values2[q] >= values2[p]):
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


def crowding_distance(values1, values2, front):
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


def run_mocle(X, n_clusters, runs, data, y=None, max_gens=50, k_range=True, representation='label', pareto_plot=False):
    for run in range(1, runs+1):
        print('============= TEST {} ============='.format(run))
        print('Clustering started using MOCLE'.format(data))
        print('Dataset: {}, Clusters: {}, Instances: {}, Features: {}'.format(data, n_clusters, len(X), len(X[0])))

        X = StandardScaler().fit_transform(X)
        start = time.time()
        if k_range is True:
            kmin = n_clusters
            kmax = 2*n_clusters
            k = range(kmin, kmax+1)
        else:
            k = n_clusters
        pop = initial_pop(X, k, k_range, representation)
        pop_size = len(pop)
        print('Population size: {}, Generations: {}'.format(pop_size, max_gens))
        gen = 1
        print('Starting genetic process...')
        while gen <= max_gens:
            print('========== Generation {} =========='.format(gen))
            
            # Fitness calculation and non-dominated sorting of the current population
            print('Computing the population\'s fitness values and crowding distances')
            function1_values = [dev(X, pop[i]) for i in range(pop_size)]
            function2_values = [conn(X, pop[i])for i in range(pop_size)]
            non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
            crowding_distance_values = []
            for i in range(0, len(non_dominated_sorted_solution)):
                crowding_distance_values.append(crowding_distance(function1_values[:], function2_values[:],
                                                                  non_dominated_sorted_solution[i][:]))
    
            # Fronts plot of the current population
            if pareto_plot:
                pareto = non_dominated_sorted_solution
                neg_function1 = [i * -1 for i in function1_values]
                neg_function2 = [j * -1 for j in function2_values]
                plt.figure(figsize=(12, 8), dpi=200)
                plt.title('Fronts of generation {}'.format(gen))
                plt.xlabel('Deviation', fontsize=15)
                plt.ylabel('Connectivity', fontsize=15)
                colors = cm.rainbow(np.linspace(0, 1, len(pareto)))
                for front in range(len(pareto)):
                    minimum = min(pareto[front])
                    maximum = max(pareto[front])+1
                    if front == 0:
                        f_name = 'Pareto Front'
                    else:
                        f_name = 'Front {}'.format(pareto)
                    plt.scatter(neg_function1[minimum:maximum], neg_function2[minimum:maximum],
                                color=colors[front], label=f_name)
                plt.legend()
                plt.show()
            
            # Populate temp_pop with offspring and parents
            temp_pop = pop[:]
            while len(temp_pop) != 2*pop_size:
                print('Populating temporal population avoiding clones. Size: {}'.format(len(temp_pop)))
                
                # Populate a temporal population avoiding clones
                offspring_in_temp_pop = 1
                offspring = np.array([])
                while offspring_in_temp_pop == 1:
                    
                    # Binary tournament for obtaining two parents
                    print('Selecting two parents')
                    p1 = binary_tournament(X, pop, function1_values, function2_values)
                    p2 = binary_tournament(X, pop, function1_values, function2_values)
                    
                    # Offspring generation with crossover
                    print('Generating children avoiding clones')
                    offspring = cross(p1, p2, k, k_range)
                    rand_indexes_offspring_temp_pop = []
                    for ind in temp_pop:
                        rand_indexes_offspring_temp_pop.append(adjusted_rand_score(ind, offspring))
                    offspring_in_temp_pop = max(rand_indexes_offspring_temp_pop)
                    if offspring_in_temp_pop == 1:
                        print('Clone eliminated')
                temp_pop.append(offspring)
            print('Temporal population complete. Size: {}'.format(len(temp_pop)))
            
            # Fitness, crowding distance calculation and non-dominated sorting of the temp_pop
            print('Calculating fitness and crowding distances of the temporal population')
            temp_function1_values = [dev(X, temp_pop[i]) for i in range(0, 2*pop_size)]
            temp_function2_values = [conn(X, temp_pop[i]) for i in range(0, 2*pop_size)]
            temp_non_dominated_sorted_solution = fast_non_dominated_sort(temp_function1_values[:],
                                                                         temp_function2_values[:])
            temp_crowding_distance_values = []
            for i in range(0, len(temp_non_dominated_sorted_solution)):
                temp_crowding_distance_values.append(crowding_distance(temp_function1_values[:],
                                                                       temp_function2_values[:],
                                                                       temp_non_dominated_sorted_solution[i][:]))
            print('Temporal population sorted')
            
            # Generating the new population
            new_pop = []
            for i in range(0, len(temp_non_dominated_sorted_solution)):
                temp_non_dominated_sorted_solution_2 = [index_of(temp_non_dominated_sorted_solution[i][j],
                                                                 temp_non_dominated_sorted_solution[i])
                                                        for j in range(0, len(temp_non_dominated_sorted_solution[i]))]
                front22 = sort_by_values(temp_non_dominated_sorted_solution_2[:], temp_crowding_distance_values[i][:])
                front = [temp_non_dominated_sorted_solution[i][front22[j]]
                         for j in range(0, len(temp_non_dominated_sorted_solution[i]))]
                front.reverse()
                for value in front:
                    if value not in new_pop:
                        new_pop.append(value)
                    if len(new_pop) == pop_size:
                        break
                if len(new_pop) == pop_size:
                    break
            print('Next generation sorted and ready. Elapsed time: {:.2f}s'.format(time.time()-start))
            
            # Population transition
            pop = [temp_pop[i] for i in new_pop]
            gen = gen + 1
        
        print('Computing fitness values and sorting the last population')
        function1_values = [dev(X, pop[i]) for i in range(pop_size)]
        function2_values = [conn(X, pop[i]) for i in range(pop_size)]
        pareto = fast_non_dominated_sort(function1_values, function2_values)
        neg_function1 = [i * -1 for i in function1_values]
        neg_function2 = [j * -1 for j in function2_values]
        
        if pareto_plot:
            # Fronts plot of the final population
            plt.figure(figsize=(12, 8), dpi=200)
            plt.title('Final fronts of generation {}'.format(gen))
            plt.xlabel('Deviation', fontsize=15)
            plt.ylabel('Connectivity', fontsize=15)
            colors = cm.rainbow(np.linspace(0, 1, len(pareto)))
            for front in range(len(pareto)):
                minimum = min(pareto[front])
                maximum = max(pareto[front])+1
                if front == 0:
                    f_name = 'Pareto Front'
                else:
                    f_name = 'Front {}'.format(pareto)
                plt.scatter(neg_function1[minimum:maximum], neg_function2[minimum:maximum],
                            color=colors[front], label=f_name)
            plt.legend()
            plt.show()
            
        mocle_time = time.time() - start
    
        # Results DataFrame
        solutions = len(pareto[0])
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
        d['Objective 1'] = neg_function1[:max(pareto[0])+1]
        d['Obj. 2 name'] = ['connectivity']*solutions
        d['Objective 2'] = neg_function2[:max(pareto[0])+1]
        d['Time'] = [mocle_time]*solutions
        d['Adjusted Rand Index'] = []
        if y is not None:
            for i in range(solutions):
                d['Adjusted Rand Index'].append(adjusted_rand_score(y, pop[i]))
        else:
            for i in range(solutions):
                d['Adjusted Rand Index'].append(np.nan)
        for i in range(len(X)):
            d['X{}'.format(i+1)] = []
        for j in range(solutions):
            for l in range(len(X)):
                d['X{}'.format(l+1)].append('{}'.format(pop[j][l]))
        out = pd.DataFrame(d)
    
        # Exporting the results
        if not os.path.exists('mocle-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens)):
            os.makedirs('mocle-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens))
        out.to_csv('mocle-out/{}_{}_{}_{}/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens,
                                                                              data, n_clusters, pop_size, max_gens,
                                                                              run))
    
        final_df = pd.DataFrame()
        names = glob.glob("mocle-out/{}_{}_{}_{}/solution*".format(data, n_clusters, pop_size, max_gens))
        for name in names:
            temp_df = pd.read_csv(name)
            final_df = pd.concat([final_df, temp_df.sort_values('Adjusted Rand Index', ascending=False).iloc[:1, :]])
        final_df.reset_index(inplace=True, drop=True)
        final_df.iloc[:, 1:].to_csv('mocle-out/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens,
                                                                                   runs))
    
        print('Clustering finished in {:.4f} seconds.'.format(mocle_time))
