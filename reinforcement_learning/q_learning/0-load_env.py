#!/usr/bin/env python3
''' Loads Environment '''

'''
Read the docs
Link: https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
'''

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    '''
    This function loads the frozen lake envrionment from OpenAI's gym
    
    Args:
        desc: None or a list containing a custom description of the map to load for the environment
        map_name: None pr a string containing the pre-made map to load
        is_slippery: Boolean to determine if the ice is slippery
    
    Returns:
        The frozen Lake Environment
    '''
    
    if desc == None and map_name == None:
        return gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
    
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery)
    
    return env

if __name__ == '__main__':
    import numpy as np
    
    np.random.seed(0)
    env = load_frozen_lake()
    print(env.desc)
    print(env.P[0][0])
    env = load_frozen_lake(is_slippery=True)
    print(env.desc)
    print(env.P[0][0])
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    print(env.desc)
    env = load_frozen_lake(map_name='4x4')
    print(env.desc)
