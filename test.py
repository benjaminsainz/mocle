"""
Authors: Katti Faceli, Marcilio C.P. de Suoto, Daniel S.A. de Araújo, and André C.P.L.F. de Carvalho.
Code: Benjamin Mario Sainz-Tinajero
Year: 2021.
https://github.com/benjaminsainz/mocle
"""

from retr import *
from gen import *


full = ['aggregation', 'breast-cancer-wisconsin', 'breast-tissue', 'dermatology', 'ecoli', 'forest', 'glass', 'iris',
       'jain', 'leaf', 'liver', 'parkinsons', 'pathbased', 'r15', 'seeds', 'segment', 'spiral', 'transfusion', 'wine',
       'zoo']


def test(ds=full, runs=10, max_gens=20):
    for d in ds:
        data, n_clusters, X, y = retrieval(d)
        run_mocle(X, n_clusters, runs, data, y, max_gens, k_range=True, representation='label', pareto_plot=False)


test()
