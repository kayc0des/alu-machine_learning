# Q Learning

This folder contains scripts for q-learning 

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