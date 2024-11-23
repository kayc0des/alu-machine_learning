# Q Learning

Q-learning is a model-free reinforcement learning algorithm that helps an agent learn the optimal action-selection policy by iteratively updating Q-values, which represent the expected rewards of actions in specific states (Geeks for Geeks, 2024). This folder contains scripts which are part of the Q-learning intranet project for ML techniques II (ALU).

## Scripts

- `0-load_env.py`: Loads FrozenLake environment from OpenAI's gym. If description and map_name isn't specified a random 8x8 map is generated, else the environment is loaded using the arguments passed in the function.
    - Function `load_frozen_lake(args)`
        - Args:
            - desc: None or a list containing a custom description of the map to load for the environment
            - map_name: None pr a string containing the pre-made map to load
            - is_slippery: Boolean to determine if the ice is slippery
        - Returns:
            - Frozen Lake Environment
    ```bash
    python3 0-load_env.py
    ```

- `1-q_init.py`: Initializes the Q-table
    - Function `q_init`
        - Args:
            - env: An instance of the frozen lake environment
        - Returns:
            - Initialized Q table as a numpy.ndarray of zeros
    ```bash
    python3 1-q_init.py
    ```

- `2-epsilon_greedy`: Uses epsilon greedy to determine if the agent should explore or exploit the environment.
    - Function `epsilon_greedy`
        - Args:
            - Q: np array containing q-values
            - state: the current state
            - epsilon: epsilon to use for the calculation
        - Returns:
            - Next Action Index
    ```bash
    python3 2-epsilon_greedy.py
    ```