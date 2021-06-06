"""
Created on Thu Mar 18 10:29:01 2021

@author: benjamin
"""
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from snn import *


def takesecond(elem):
    return elem[1]


def initial_pop(X, k, k_range, representation):
    pop = list()
    nn = [int(.2*len(X)), int(.5*len(X)), int(.1*len(X)), int(.2*len(X)), int(.3*len(X)), int(.4*len(X))]
    min_shared = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    if k_range is True:
        kmin = min(list(k))
        kmax = max(list(k))
        for n in k:
            temp = []
            for _ in range(30):
                km_model = KMeans(n, init='random').fit(X)  # k-means model
                temp.append((km_model.labels_, km_model.inertia_))
            temp.sort(key=takesecond)
            pop.append(temp[0][0])
            al_model = AgglomerativeClustering(n, linkage='average', affinity='euclidean').fit(X)  # average link model
            pop.append(al_model.labels_)
            sl_model = AgglomerativeClustering(n, linkage='single', affinity='euclidean').fit(X)  # single link model
            pop.append(sl_model.labels_)
        for i in nn:
            for j in min_shared:
                snn_model = SNN(neighbor_num=i, min_shared_neighbor_proportion=j).fit(X)  # snn model
                if (len(set(snn_model.labels_)) >= kmin) and len(set(snn_model.labels_)) <= kmax:
                    pop.append(snn_model.labels_)
    else:
        for _ in range(k, k*2+1):
            temp = list()
            for _ in range(30):
                km_model = KMeans(k, init='random').fit(X)  # k-means model
                temp.append((km_model.labels_, km_model.inertia_))
            temp.sort(key=takesecond)
            pop.append(temp[0][0])
            al_model = AgglomerativeClustering(k, linkage='average', affinity='euclidean').fit(X)  # average link model
            pop.append(al_model.labels_)
            sl_model = AgglomerativeClustering(k, linkage='single', affinity='euclidean').fit(X)  # single link model
            pop.append(sl_model.labels_)
        for i in nn:
            for j in min_shared:
                snn_model = SNN(neighbor_num=i, min_shared_neighbor_proportion=j).fit(X)  # snn model
                if len(set(snn_model.labels_)) == k:
                    pop.append(snn_model.labels_)

    temp_ind = list()
    if representation == 'sets':
        temp_pop = list()
        for individual in pop:
            d = dict()
            for gene in range(len(individual)):
                d['{}'.format(individual[gene])] = []
            for gene in range(len(individual)):
                for key in d.keys():
                    if individual[gene] == int(key):
                        d[key].append(gene)
            temp_ind = list()
            for c in d.values():
                temp_ind.append(set(c))
            temp_pop.append(np.array(temp_ind))
        return temp_ind
    elif representation == 'label':
        return pop
