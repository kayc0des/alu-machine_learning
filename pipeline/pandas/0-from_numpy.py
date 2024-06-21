#!/usr/bin/env python3
''' A function that creates a
pd.DataFrame from np.ndarray'''


import pandas as pd


def from_numpy(array):
    '''
    Creates pd.DataFrame from np.ndarray

    Args:
    array -> numpy array

    Returns:
    DataFrame
    '''
    header = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    m = array.shape[1]
    header = header[0:m]

    df = pd.DataFrame(array, columns=header)

    return df
