import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

class replaybuffer:
    def __init__(self):
        self.buffer = deque(maxlen=2000)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = np.array(random.sample(self.buffer, batch_size), dtype=object)
        states = np.array(samples[:, 0].tolist())
        actions = np.array(samples[:, 1].tolist())
        rewards = np.array(samples[:, 2].tolist())
        next_states = np.array(samples[:, 3].tolist())
        dones = np.array(samples[:, 4].tolist())
        return states, actions, rewards, next_states, dones

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
    
class SACAgent:
    def __init__(self, env, hidden_dim=256, alpha=0.1, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.state_dim + self.action_dim, hidden_dim)
        self.target_critic = Critic(self.state_dim + self.action_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.alpha = alpha
        self.gamma = gamma


    def get_action(self, state, test=False):
        state = torch.FloatTensor(state.reshape(1, -1))
        with torch.no_grad():
            action = self.actor(state).numpy()
        if not test:
            action = action + np.random.normal(0, 0.1, size=self.env.action_space.shape[0])
        return np.clip(action, -self.action_max, self.action_max)

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
    def train(self, episodes=100, batch_size=64):
        rewards = []

        replay_buffer = replaybuffer()
        for episode in range(episodes):
            state = self.env.reset()
            reward_ = 0
            for step in range(200):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward_ = reward_ + reward
                replay_buffer.add(state, action, reward, next_state, done)

                if len(replay_buffer.buffer) > batch_size:
                    self.update(replay_buffer, batch_size)

                if done:
                    break
                state = next_state
            rewards.append(reward_)

        plt.figure()
        plt.plot(range(1, episodes+1), rewards, label = 'Soft-Actor', color = 'blue')
        plt.title("Rewards vs Episode for Soft Actor-Critic")
        plt.xlabel('Episodes')
        plt.ylabel("Rewards")
        plt.legend()
        plt.savefig("Plot_soft-actor-critic.png")
        plt.show()

env = gym.make('Pendulum-v0')
agent = SACAgent(env)
agent.train()