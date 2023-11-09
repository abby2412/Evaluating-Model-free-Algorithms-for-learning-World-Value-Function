import numpy as np
import gym
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from collections import namedtuple


class plotting:
    EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

def e_greedy(Q_s, epsilon, A):
    policy = np.ones(A) * epsilon / A
    best_a = np.argmax(Q_s)
    policy[best_a] = 1 - epsilon + (epsilon / A)
    return policy

def Q_Learn(env, episodes, goal_value, alpha=0.5, gamma=1.0, epsilon=0.1):
   
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

    for episode in range(episodes):
        state = env.reset()
        rewards = 0

        for t in itertools.count():
            action_probs = e_greedy(Q[state], epsilon, env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, info = env.step(action)
            rewards = rewards + reward

            best_next_action = np.argmax(Q[next_state])
            td_error = reward + gamma*goal_value
            Q[state][action] = Q[state][action] + (alpha*(td_error - Q[state][action]))

            if done:
                break

            state = next_state

        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] = rewards

    return Q, stats



def SARSA(env, episodes, alpha=0.5, gamma=1.0, epsilon=0.1):

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

    for episode in range(episodes):
        state = env.reset()
        action = np.argmax(Q[state])
        rewards = 0

        for t in itertools.count():
            action_probs = e_greedy(Q[state], epsilon, env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, info = env.step(action)
            rewards = rewards + reward

            if done:
                goal_value = rewards  
                Q[state][action] = Q[state][action] + alpha*(reward + gamma*goal_value - Q[state][action])
                break

            next_action = np.argmax(Q[next_state])
            Q[state][action] = Q[state][action] + alpha*(reward + gamma*Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action

        stats.episode_lengths[episode] = t
        stats.episode_rewards[episode] = rewards

    return Q, stats


env = gym.make('CliffWalking-v0')
num_episodes = 500
goal_value = 5  
Q_sarsa, stats_sarsa = SARSA(env, num_episodes, goal_value)
Q_q_learning, stats_q_learning = Q_Learn(env, num_episodes, goal_value)

plt.figure()
plt.plot(stats_sarsa.episode_rewards, label='SARSA', color='b')
plt.plot(stats_q_learning.episode_rewards, label='Q-Learning', color='r')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.title('Sum of rewards per episode for SARSA')
plt.legend()
plt.show()
