#!/usr/bin/env python3
''' Calculates the weighter MA '''


def moving_average(data, beta):
    '''
    Calculates the MA of a data set

    Args:
    data -> the list of data to calculate the MA
    beta -> weight usef for the <MA

    Returns:
    a lost comtaining the moving averages
    '''
    m_avg = []
    v_tetha = 0
    for i in range(1, len(data) + 1):
        v_tetha =  (beta * v_tetha) + ((1 - beta) * data[i - 1])
        bias = v_tetha / (1 - (beta ** i))
        m_avg.append(bias)

    return m_avg
