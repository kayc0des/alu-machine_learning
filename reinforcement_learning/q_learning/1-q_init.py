#!/usr/bin/env python3
''' Initializes the Q table '''


import gym
import numpy as np

# load function from 0-load_env.py
load_frozen_lake = __import__('0-load_env').load_frozen_lake


def q_init(env):
    '''
    This function initializes the q table for an environment
    Remember the q table is a matrix of states x actions
    
    Args:
        env: the instance of FrozenLakeEnv
    
    Returns
        Q-tables as an np.ndarray of zeros
    '''
    
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    
    q_table = np.zeros(shape=(state_space_size, action_space_size))
    
    return q_table

# Debug
if __name__ == '__main__':
    env = load_frozen_lake()
    Q = q_init(env)
    print(Q.shape)
    env = load_frozen_lake(is_slippery=True)
    Q = q_init(env)
    print(Q.shape)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)
    print(Q.shape)
    env = load_frozen_lake(map_name='4x4')
    Q = q_init(env)
    print(Q.shape)