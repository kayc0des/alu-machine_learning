#!/usr/bin/env python3
''' Uses Epsilon-greedy to determine the next action '''


import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    '''
    Reference: Exploration vs Exploitation
    Uses epsilon-gredy to determine the next action
    
    Args:
        Q: np array containing q-values
        state: the current state
        epsilon: epsilon to use for the calculation
    
    Returns:
        Next action index
    '''
    
    exploration_rate_threshold = np.random.uniform(0, 1)
    
    if exploration_rate_threshold > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])

    return action

if __name__ == '__main__':
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init

    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)
    Q[7] = np.array([0.5, 0.7, 1, -1])
    np.random.seed(0)
    print(epsilon_greedy(Q, 7, 0.5))
    np.random.seed(1)
    print(epsilon_greedy(Q, 7, 0.5))