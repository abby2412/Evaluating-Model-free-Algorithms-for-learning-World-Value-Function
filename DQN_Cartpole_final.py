# DQN Agent using the Cartpole domain from OpenAI gym

import numpy as np
import gym
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_space_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model


    def memery(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay_buffer(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.argmax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

env = gym.make('CartPole-v1')
state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
agent = DQNAgent(state_space_size, action_space_size)

done = False
batch_size = 32

for episode in range(10):
    state = env.reset()
    state = np.reshape(state, [1, state_space_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_space_size])
        agent.memery(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode:{episode}, score:{time}, epsilon:{agent.epsilon}")
            break
        if len(agent.memory) > batch_size:
            agent.replay_buffer(batch_size)
        
