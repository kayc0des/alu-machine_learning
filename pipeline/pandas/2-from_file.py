#!/usr/bin/env python3
''' Loads data from a file as a pd.DataFrame '''


import pandas as pd


def from_file(filename, delimiter):
    '''
    Loads data from a file as pd

    Args:
    filename -> file to load from
    delimeter -> column separator

    Returns:
    pd.DataFrame
    '''
    return pd.read_csv(filename, delimiter=delimiter)
