"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""


import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import StandardScaler


def original_data_numerical_dropna(data):
    temp_df = pd.read_csv('data/{}_X.csv'.format(data), header=None)
    for column in temp_df.select_dtypes(include=['object']):
        temp_df[column], _ = pd.factorize(temp_df[column])
    if os.path.isfile('data/{}_y.csv'.format(data)):
        y_df = pd.read_csv('data/{}_y.csv'.format(data), header=None)
        temp_df['y'] = y_df[0].to_list()
        temp_df['y'] = temp_df['y'].astype(str)
    else:
        temp_df['y'] = ['1']*len(temp_df)
    return temp_df


def retrieval(data, n_clusters):
    unsorted_df = original_data_numerical_dropna(data)
    if n_clusters == 'auto':
        n_clusters = len(set(unsorted_df.loc[:,'y'].to_list()))
    X = StandardScaler().fit_transform(unsorted_df.iloc[:,:-1])
    X = np.array(X).astype(float)
    if os.path.isfile('data/{}_y.csv'.format(data)):
        y = unsorted_df.loc[:,'y'].to_list()
    else:
        y = None
    return data, n_clusters, X, y
