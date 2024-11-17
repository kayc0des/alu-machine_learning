import gym
import numpy as np
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# Load the trained policy
policy_model = load_model('policy.h5')

# Create the Atari environment
env = gym.make('Breakout-v4')
nb_actions = env.action_space.n

# Prepare the agent
memory = SequentialMemory(limit=1000000, window_length=4)
policy = GreedyQPolicy()
dqn = DQNAgent(model=policy_model, policy=policy, memory=memory,
               nb_actions=nb_actions, nb_steps_warmup=0,
               target_model_update=1, gamma=0.99)
dqn.compile(optimizer=None)  # No need to optimize for playing

# Play the game
dqn.test(env, nb_episodes=5, visualize=True)
env.close()