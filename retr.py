"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from sklearn import datasets
import csv
import numpy as np
import pandas as pd


def retrieval(data):
    if data == 'aggregation':
        n_clusters = 7
        X = np.array(list(csv.reader(open('data/aggregation_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/aggregation_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'breast-cancer-wisconsin':
        n_clusters = 2
        X, y = datasets.load_breast_cancer().data, datasets.load_breast_cancer().target
    elif data == 'breast-tissue':
        n_clusters = 6
        X = np.array(list(csv.reader(open('data/breast-tissue_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/breast-tissue_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'dermatology':
        n_clusters = 6
        X = np.array(list(csv.reader(open('data/dermatology_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/dermatology_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'ecoli':
        n_clusters = 8
        X = np.array(list(csv.reader(open('data/ecoli_X.csv', newline=''))))
        y = list(csv.reader(open('data/ecoli_y.csv', newline='')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'forest':
        n_clusters = 4
        X = np.array(list(csv.reader(open('data/forest_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/forest_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'glass':
        n_clusters = 6
        X = np.array(list(csv.reader(open('data/glass_X.csv', newline=''))))
        y = list(csv.reader(open('data/glass_y.csv', newline='')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'iris':
        n_clusters = 3
        X, y = datasets.load_iris().data, datasets.load_iris().target
    elif data == 'jain':
        n_clusters = 2
        X = np.array(list(csv.reader(open('data/jain_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/jain_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'leaf':
        n_clusters = 36
        X = np.array(list(csv.reader(open('data/leaf_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/leaf_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'liver':
        n_clusters = 16
        X = np.array(list(csv.reader(open('data/liver_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/liver_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'parkinsons':
        n_clusters = 2
        X = np.array(list(csv.reader(open('data/parkinsons_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/parkinsons_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'pathbased':
        n_clusters = 3
        X = np.array(list(csv.reader(open('data/pathbased_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/pathbased_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'r15':
        n_clusters = 15
        X = np.array(list(csv.reader(open('data/r15_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/r15_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'seeds':
        n_clusters = 3
        X = np.array(list(csv.reader(open('data/seeds_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/seeds_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'segment':
        n_clusters = 7
        X = np.array(list(csv.reader(open('data/segment_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/segment_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'spiral':
        n_clusters = 3
        X = np.array(list(csv.reader(open('data/spiral_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/spiral_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'transfusion':
        n_clusters = 2
        X = np.array(list(csv.reader(open('data/transfusion_X.csv', newline=''))))
        y = list(csv.reader(open('data/transfusion_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    elif data == 'wine':
        n_clusters = 3
        X, y = datasets.load_wine().data, datasets.load_wine().target
    elif data == 'zoo':
        n_clusters = 7
        X = np.array(list(csv.reader(open('data/zoo_X.csv', newline='', encoding='utf-8-sig'))))
        y = list(csv.reader(open('data/zoo_y.csv', newline='', encoding='utf-8-sig')))
        X, y = X.astype(float), np.array([item for sublist in y for item in sublist])
    return data, n_clusters, X, y
