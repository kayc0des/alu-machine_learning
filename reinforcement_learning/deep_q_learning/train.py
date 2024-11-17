import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Create the Atari environment
env = gym.make('Breakout-v4')
nb_actions = env.action_space.n

# Build the model
def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

# DQN agent
def build_agent(model, nb_actions):
    policy = EpsGreedyQPolicy(eps=0.1)
    memory = SequentialMemory(limit=1000000, window_length=4)
    dqn = DQNAgent(model=model, policy=policy, memory=memory, 
                   nb_actions=nb_actions, nb_steps_warmup=50000,
                   target_model_update=10000, gamma=0.99)
    return dqn

# Training process
input_shape = (4,) + env.observation_space.shape[1:]  # Use stacked frames
model = build_model(input_shape, nb_actions)
dqn = build_agent(model, nb_actions)
dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

# Agent training
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

# Save the final policy
model.save('policy.h5')
env.close()
