# Q-learning and SARSA algorithm using a normal value function 

import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

def Q_Learn(env, episodes, epsilon, gamma, alpha):
    
    return_ = [] #store the rewards for episode
    
    Q = np.zeros([env.observation_space.n, env.action_space.n]) 
    
    for episode in range(episodes):
       
        rewards = 0
        state, info = env.reset()

        for j in range(100):

            actions = []
            actions.append(env.action_space.sample())   # explore
            actions.append(np.argmax(Q[state]))         # exploit

            action = random.choices(actions, weights=(epsilon, 1-epsilon), k=1)[0]  # e-greedy policy
                
            next_state, reward, done, truncated, info = env.step(action) # taking a step and observing
            
            rewards = rewards + reward 

            next_action = np.argmax(Q[next_state]) #choose next action by exploiting
            
            td_error = reward + gamma*(np.max(Q[next_state,next_action])) - Q[state,action]
            
            Q[state,action] = Q[state,action] + alpha*td_error  #update
            
            state = next_state
            
            if done: 
                break
                
        return_.append(rewards) 
                
    return return_

def SARSA(env, episodes, epsilon, gamma, alpha):
    
    return_ = []
    
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    
    for episode in range(episodes):
       
        rewards = 0
        state, info = env.reset() 

        for j in range(100):

            actions = []
            actions.append(env.action_space.sample())
            actions.append(np.argmax(Q[state]))
          
            action = random.choices(actions, weights=(epsilon, 1-epsilon), k=1)[0]  # e-greedy policy
                
            next_state, reward, done, truncated, info = env.step(action)
            
            rewards = rewards + reward

            actions = []
            actions.append(env.action_space.sample())
            actions.append(np.argmax(Q[next_state]))
            next_action = random.choices(actions, weights=(epsilon, 1-epsilon), k=1)[0]  # e-greedy policy
                
            td_error = reward + gamma*(Q[next_state,next_action]) - Q[state,action] 
            
            Q[state,action] = Q[state,action] + alpha*td_error  #update
            
            state = next_state
            
            if done:
                break
                
        return_.append(rewards)  
                
    return return_


env = gym.make('CliffWalking-v0')
average_Q_Learn = np.zeros(1000)
average_SARSA = np.zeros(1000)

for m in range(10):
    arr = Q_Learn(env, 1000, 0.1, 0.99, 0.1)
    for k in range(1000):
        average_Q_Learn[k] = average_Q_Learn[k] + arr[k]

for r in range(len(average_Q_Learn)):
    average_Q_Learn[r] = average_Q_Learn[r]/10

for m in range(10):
    arr = SARSA(env, 1000, 0.1, 0.99, 0.1)
    for kk in range(1000):
        average_SARSA[kk] = average_SARSA[kk] + arr[kk]

for r in range(len(average_SARSA)):
    average_SARSA[r] = average_SARSA[r]/10

plt.figure()
plt.plot(average_Q_Learn, color = 'red', label = 'Q-learning')
plt.plot(average_SARSA, color = 'green', label = 'SARSA')
plt.title("Average Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Average Rewards")
plt.legend()
plt.savefig("Plot_sarsa_q.png")
plt.show()