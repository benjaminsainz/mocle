#!/usr/bin/env python

# Cluster_Ensembles/src/Cluster_Ensembles/Cluster_Ensembles.py;

# Author: Gregory Giecold for the GC Yuan Lab
# Affiliation: Harvard University
# Contact: g.giecold@gmail.com, ggiecold@jimmy.harvard.edu


"""Cluster_Ensembles is a package for combining multiple partitions 
into a consolidated clustering.
The combinatorial optimization problem of obtaining such a consensus clustering
is reformulated in terms of approximation algorithms for 
graph or hyper-graph partitioning.

References
----------
* Giecold, G., Marco, E., Trippa, L. and Yuan, G.-C.,
"Robust Lineage Reconstruction from High-Dimensional Single-Cell Data". 
ArXiv preprint [q-bio.QM, stat.AP, stat.CO, stat.ML]: http://arxiv.org/abs/1601.02748

* Strehl, A. and Ghosh, J., "Cluster Ensembles - A Knowledge Reuse Framework
for Combining Multiple Partitions".
In: Journal of Machine Learning Research, 3, pp. 583-617. 2002

* Kernighan, B. W. and Lin, S., "An Efficient Heuristic Procedure 
for Partitioning Graphs". 
In: The Bell System Technical Journal, 49, 2, pp. 291-307. 1970

* Karypis, G. and Kumar, V., "A Fast and High Quality Multilevel Scheme 
for Partitioning Irregular Graphs"
In: SIAM Journal on Scientific Computing, 20, 1, pp. 359-392. 1998

* Karypis, G., Aggarwal, R., Kumar, V. and Shekhar, S., "Multilevel Hypergraph Partitioning: 
Applications in the VLSI Domain".
In: IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 7, 1, pp. 69-79. 1999
"""
import metis
import gc
import numpy as np
import operator
import scipy.sparse
import warnings
import six
from six.moves import range
from functools import reduce

import tables
import numbers
import psutil
import subprocess

import collections
import ctypes
import os, sys, operator as op

np.seterr(invalid = 'ignore')
warnings.filterwarnings('ignore')

IDXTYPEWIDTH  = os.getenv('METIS_IDXTYPEWIDTH', '32')

# print(IDXTYPEWIDTH)

if IDXTYPEWIDTH == '32':
    idx_t = ctypes.c_int32
elif IDXTYPEWIDTH == '64':
    idx_t = ctypes.c_int64
else:
    raise EnvironmentError('Env var METIS_IDXTYPEWIDTH must be "32" or "64"')

# idx_t = ctypes.c_int64


def memory():
    """Determine memory specifications of the machine.

    Returns
    -------
    mem_info : dictonary
        Holds the current values for the total, free and used memory of the system.
    """

    mem_info = dict()

    for k, v in psutil.virtual_memory()._asdict().items():
           mem_info[k] = int(v)
           
    return mem_info

def get_chunk_size(N, n):
    """Given a two-dimensional array with a dimension of size 'N', 
        determine the number of rows or columns that can fit into memory.

    Parameters
    ----------
    N : int
        The size of one of the dimensions of a two-dimensional array.  

    n : int
        The number of arrays of size 'N' times 'chunk_size' that can fit in memory.

    Returns
    -------
    chunk_size : int
        The size of the dimension orthogonal to the one of size 'N'. 
    """

    mem_free = memory()['free']
    if mem_free > 60000000:
        chunk_size = int(((mem_free - 10000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 40000000:
        chunk_size = int(((mem_free - 7000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 14000000:
        chunk_size = int(((mem_free - 2000000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 8000000:
        chunk_size = int(((mem_free - 1400000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 2000000:
        chunk_size = int(((mem_free - 900000) * 1000) / (4 * n * N))
        return chunk_size
    elif mem_free > 1000000:
        chunk_size = int(((mem_free - 400000) * 1000) / (4 * n * N))
        return chunk_size
    else:
        print("\nERROR: Cluster_Ensembles: get_chunk_size: "
              "this machine does not have enough free memory resources "
              "to perform MCLA clustering.\n")
        sys.exit(1)

def build_hypergraph_adjacency(cluster_runs):
    """Return the adjacency matrix to a hypergraph, in sparse matrix representation.
    
    Parameters
    ----------
    cluster_runs : array of shape (n_partitions, n_samples)
    
    Returns
    -------
    hypergraph_adjacency : compressed sparse row matrix
        Represents the hypergraph associated with an ensemble of partitions,
        each partition corresponding to a row of the array 'cluster_runs'
        provided at input.
    """

    N_runs = cluster_runs.shape[0]

    hypergraph_adjacency = create_membership_matrix(cluster_runs[0])
    for i in range(1, N_runs):
        hypergraph_adjacency = scipy.sparse.vstack([hypergraph_adjacency,
                                                   create_membership_matrix(cluster_runs[i])], 
                                                   format = 'csr')
    return hypergraph_adjacency

METIS_Graph = collections.namedtuple('METIS_Graph', 'nvtxs ncon xadj adjncy vwgt vsize adjwgt')

def one_to_max(array_in):
    """Alter a vector of cluster labels to a dense mapping. 
        Given that this function is herein always called after passing 
        a vector to the function checkcl, one_to_max relies on the assumption 
        that cluster_run does not contain any NaN entries.

    Parameters
    ----------
    array_in : a list or one-dimensional array
        The list of cluster IDs to be processed.
    
    Returns
    -------
    result : one-dimensional array
        A massaged version of the input vector of cluster identities.
    """
    
    x = np.asanyarray(array_in)
    N_in = x.size
    array_in = x.reshape(N_in)    

    sorted_array = np.sort(array_in)
    sorting_indices = np.argsort(array_in)

    last = np.nan
    current_index = -1
    for i in range(N_in):
        if last != sorted_array[i] or np.isnan(last):
            last = sorted_array[i]
            current_index += 1

        sorted_array[i] = current_index

    result = np.empty(N_in, dtype = int)
    result[sorting_indices] = sorted_array

    return result

def MCLA(cluster_runs, verbose = False, N_clusters_max = None):

    cluster_ensemble = []
    score = np.empty(0)


    if N_clusters_max == None:
        N_clusters_max = int(np.nanmax(cluster_runs)) + 1

    N_runs = cluster_runs.shape[0]
    N_samples = cluster_runs.shape[1]

    #Cluster_Ensembles: MCLA: preparing graph for meta-clustering.
    hypergraph_adjacency = build_hypergraph_adjacency(cluster_runs)
    w = hypergraph_adjacency.sum(axis = 1)

    N_rows = hypergraph_adjacency.shape[0]

    # Next, obtain a matrix of pairwise Jaccard similarity scores between the rows of the hypergraph adjacency matrix.

    scale_factor = 100.0

    #starting computation of Jaccard similarity matrix.
    squared_MCLA = hypergraph_adjacency.dot(hypergraph_adjacency.transpose())

    squared_sums = hypergraph_adjacency.sum(axis = 1)
    squared_sums = np.squeeze(np.asarray(squared_sums))

    chunks_size = get_chunk_size(N_rows, 7)
    for i in range(0, N_rows, chunks_size):
        n_dim = min(chunks_size, N_rows - i)

        temp = squared_MCLA[i:min(i+chunks_size, N_rows), :].todense()
        temp = np.squeeze(np.asarray(temp))

        x = squared_sums[i:min(i+chunks_size, N_rows)]
        x = x.reshape(-1, 1)
        x = np.dot(x, np.ones((1, squared_sums.size)))

        y = np.dot(np.ones((n_dim, 1)), squared_sums.reshape(1, -1))
    
        temp = np.divide(temp, x + y - temp)
        temp *= scale_factor

        Jaccard_matrix = np.rint(temp)
        # print(Jaccard_matrix)

        # del Jaccard_matrix, temp, x, y
        del temp, x, y
        gc.collect()
 
    # Done computing the matrix of pairwise Jaccard similarity scores.

    ####################################################################################################


    e_mat = Jaccard_matrix
    # print(e_mat[0])

    # print(N_rows)
    N_cols = e_mat.shape[1]

    w *= scale_factor
    w = np.rint(w)
    vwgt = []
    for sublist in w.tolist():
        for item in sublist:
            vwgt.append(int(item))

    # print(vwgt)

    diag_ind = np.diag_indices(N_rows)
    e_mat[diag_ind] = 0

    adjncy = []
    adjwgt = []
    xadj = []
    xadjind = 0
    xadj.append(0) #first element always starts with 0

    chunks_size = get_chunk_size(N_cols, 7)
    for i in range(0, N_rows, chunks_size):
        M = e_mat[i:min(i+chunks_size, N_rows)]

        for j in range(M.shape[0]):
            edges = np.where(M[j] > 0)[0]
            weights = M[j, edges]

            xadjind += edges.size
            xadj.append(xadjind)

            adjncy.extend(edges)
            adjwgt.extend(weights)


    adjwgt = list(map(int, adjwgt))

    # max_w = np.max(vwgt)
    # min_w = np.min(vwgt)
    # vwgt_norm = (vwgt-min_w)/(max_w-min_w)
    # print("vwgt : ", vwgt)
    # print("vwgt_norm : ", vwgt_norm+1)

    # print("adjwgt : ", adjwgt)

    # N_rows = 12 
    # N_clusters_max = 10 
    # xadj = [0, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, ]
    # adjncy = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ]
    # vwgt = [90300, 200, 11600, 11400, 9600, 11600, 7200, 8600, 9700, 5800, 7100, 7900, ]
    # adjwgt = [13, 13, 11, 13, 8, 10, 11, 6, 8, 9, 2, 13, 13, 11, 13, 2, 8, 10, 11, 6, 8, 9, ]

    # print(cluster_runs)
    # print("xadj: ", xadj)
    # print("adjncy : ", adjncy)
    # print("vwgt : ", vwgt)
    # print("adjwgt : ", adjwgt)
    # print("\n")


    xadj = (idx_t * len(xadj))(*xadj)
    adjncy = (idx_t * len(adjncy))(*adjncy)
    adjwgt = (idx_t * len(adjwgt))(*adjwgt)
    vwgt = (idx_t * len(vwgt))(*vwgt)

    ncon = idx_t(1)



    G = METIS_Graph(idx_t(N_rows), ncon, xadj, adjncy, vwgt, None, adjwgt)
    # print(G)
  

    (edgecuts, parts) = metis.part_graph(G, N_clusters_max)
    cluster_labels = parts

    # print("parts")
    # print(parts)


    ##########################################################################################################


    cluster_labels = one_to_max(cluster_labels)
    # print(cluster_labels)
    # After 'metis' returns, we are done with clustering hyper-edges

    # We are now ready to start the procedure meant to collapse meta-clusters.
    N_consensus = np.amax(cluster_labels) + 1

    clb_cum = np.zeros(shape=(N_consensus, N_samples))
 
    chunks_size = get_chunk_size(N_samples, 7)
    for i in range(0, N_consensus, chunks_size):
        x = min(chunks_size, N_consensus - i)
        matched_clusters = np.where(cluster_labels == np.reshape(np.arange(i, min(i + chunks_size, N_consensus)), newshape = (x, 1)))
        M = np.zeros((x, N_samples))
        for j in range(x):
            coord = np.where(matched_clusters[0] == j)[0]
            M[j] = np.asarray(hypergraph_adjacency[matched_clusters[1][coord], :].mean(axis = 0))
        clb_cum[i:min(i+chunks_size, N_consensus)] = M
    
    # Done with collapsing the hyper-edges into a single meta-hyper-edge, 
    # for each of the (N_consensus - 1) meta-clusters.

    del hypergraph_adjacency
    gc.collect()

    # print(clb_cum[10])

    # Each object will now be assigned to its most associated meta-cluster.
    chunks_size = get_chunk_size(N_consensus, 4)
    N_chunks, remainder = divmod(N_samples, chunks_size)
    if N_chunks == 0:
        null_columns = np.where(clb_cum[:].sum(axis = 0) == 0)[0]
    else:
        szumsz = np.zeros(0)
        for i in range(N_chunks):
            M = clb_cum[:, i*chunks_size:(i+1)*chunks_size]
            szumsz = np.append(szumsz, M.sum(axis = 0))
        if remainder != 0:
            M = clb_cum[:, N_chunks*chunks_size:N_samples]
            szumsz = np.append(szumsz, M.sum(axis = 0))
        null_columns = np.where(szumsz == 0)[0]

    if null_columns.size != 0:
        # print("INFO: Cluster_Ensembles: MCLA: {} objects with all zero associations "
        #       "in 'clb_cum' matrix of meta-clusters.".format(null_columns.size))
        clb_cum[:, null_columns] = np.random.rand(N_consensus, null_columns.size)

    random_state = np.random.RandomState()

    tmp = np.zeros(shape=(N_consensus, N_samples))

    chunks_size = get_chunk_size(N_samples, 2)
    N_chunks, remainder = divmod(N_consensus, chunks_size)
    if N_chunks == 0:
        tmp[:] = random_state.rand(N_consensus, N_samples)
    else:
        for i in range(N_chunks):
            tmp[i*chunks_size:(i+1)*chunks_size] = random_state.rand(chunks_size, N_samples)
        if remainder !=0:
            tmp[N_chunks*chunks_size:N_consensus] = random_state.rand(remainder, N_samples)

    expr = tables.Expr("clb_cum + (tmp / 10000)")
    expr.set_output(clb_cum)
    expr.eval()

    expr = tables.Expr("abs(tmp)")
    expr.set_output(tmp)
    expr.eval()

    chunks_size = get_chunk_size(N_consensus, 2)
    N_chunks, remainder = divmod(N_samples, chunks_size)
    if N_chunks == 0:
        sum_diag = tmp[:].sum(axis = 0)
    else:
        sum_diag = np.empty(0)
        for i in range(N_chunks):
            M = tmp[:, i*chunks_size:(i+1)*chunks_size]
            sum_diag = np.append(sum_diag, M.sum(axis = 0))
        if remainder != 0:
            M = tmp[:, N_chunks*chunks_size:N_samples]
            sum_diag = np.append(sum_diag, M.sum(axis = 0))

    inv_sum_diag = np.reciprocal(sum_diag.astype(float))

    if N_chunks == 0:
        clb_cum *= inv_sum_diag
        max_entries = np.amax(clb_cum, axis = 0)
    else:
        max_entries = np.zeros(N_samples)
        for i in range(N_chunks):
            clb_cum[:, i*chunks_size:(i+1)*chunks_size] *= inv_sum_diag[i*chunks_size:(i+1)*chunks_size]
            max_entries[i*chunks_size:(i+1)*chunks_size] = np.amax(clb_cum[:, i*chunks_size:(i+1)*chunks_size], axis = 0)
        if remainder != 0:
            clb_cum[:, N_chunks*chunks_size:N_samples] *= inv_sum_diag[N_chunks*chunks_size:N_samples]
            max_entries[N_chunks*chunks_size:N_samples] = np.amax(clb_cum[:, N_chunks*chunks_size:N_samples], axis = 0)

    cluster_labels = np.zeros(N_samples, dtype = int)
    winner_probabilities = np.zeros(N_samples)
    
    chunks_size = get_chunk_size(N_samples, 2)
    for i in reversed(range(0, N_consensus, chunks_size)):
        ind = np.where(np.tile(max_entries, (min(chunks_size, N_consensus - i), 1)) == clb_cum[i:min(i+chunks_size, N_consensus)])
        cluster_labels[ind[1]] = i + ind[0]
        winner_probabilities[ind[1]] = clb_cum[(ind[0] + i, ind[1])]       

    # Done with competing for objects.

    cluster_labels = one_to_max(cluster_labels)

    return cluster_labels


def create_membership_matrix(cluster_run):
    """For a label vector represented by cluster_run, constructs the binary 
        membership indicator matrix. Such matrices, when concatenated, contribute 
        to the adjacency matrix for a hypergraph representation of an 
        ensemble of clusterings.
    
    Parameters
    ----------
    cluster_run : array of shape (n_partitions, n_samples)
    
    Returns
    -------
    An adjacnecy matrix in compressed sparse row form.
    """

    cluster_run = np.asanyarray(cluster_run)

    if reduce(operator.mul, cluster_run.shape, 1) != max(cluster_run.shape):
        raise ValueError("\nERROR: Cluster_Ensembles: create_membership_matrix: "
                         "problem in dimensions of the cluster label vector "
                         "under consideration.")
    else:
        cluster_run = cluster_run.reshape(cluster_run.size)

        cluster_ids = np.unique(np.compress(np.isfinite(cluster_run), cluster_run))
      
        indices = np.empty(0, dtype = np.int32)
        indptr = np.zeros(1, dtype = np.int32)

        for elt in cluster_ids:
            indices = np.append(indices, np.where(cluster_run == elt)[0])
            indptr = np.append(indptr, indices.size)

        data = np.ones(indices.size, dtype = int)

        return scipy.sparse.csr_matrix((data, indices, indptr), shape = (cluster_ids.size, cluster_run.size))
