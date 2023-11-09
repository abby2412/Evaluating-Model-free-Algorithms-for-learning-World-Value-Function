# Evaluating-Model-free-Algorithms-for-learning-World-Value-Function

## Abstract

Reinforcement learning (RL) is a machine learning method that is a framework for developing intelligent agents that are capable of making decisions in difficult environments. In RL, value functions provide an estimate of the expected cumulative future reward for an agent in a certain state. The pursuit of creating adaptable and goal-oriented agents in the field of reinforcement learning has led to the exploration of World Value Functions (WVFs). World value functions learns the value of reaching an action at each state attempting to reach each and every goal and they can be evaluated using model-free algorithms. Recent work has demonstrated the importance of learning a particular type of goal-oriented value function for reinforcement learning. This learning can be done with any normal reinforcement learning algorithm, but no analysis and research has been done to determine which algorithm is better to use from all the model-free algorithms. This paper focuses on evaluating model-free algorithms on different environments aiming to determine the most suitable algorithm to use, evaluating the performance for learning these type of functions, WVF's, by using OpenAI Gym environments to simulate different environments and vary the reward function to evaluate how it affects the performance of the algorithms. In this paper we have found that SARSA outperforms Q-learning and Dyna q-learning on an environment and that soft-actor critic algorithm can handle complex environments, as well as showing DQN's fluctuating performance and futher work can be done on this. This paper mentions some advantages and disadvantages of the algorithms, which can contribute and be helpful to the development of more efficient and more effective reinforcement learning algorithms for many different applications and to contribute to the field of machine learning, reinforcement learning.

(add link to paper)

## Structure

### Directories

- experiments/ contains the files of the various algorithms ran on the different domains.
- results/ contains plots of the rewards vs episodes for the various algorithms and domains.

### Scripts

- requirements1.txt includes all the packages with their versions needed for the Q-learning, SARSA and soft actor-critic algorithm
- requirements2.txt includes all the packages with their versions needed for the DQN algorithm
- Q_SARSA_WVF_final.py is the implementation of the q-learning and sarsa algo on cliff-walking domain 
- soft-Actor-critic.py is the implementation of the soft actor-critic algo on pendulum domain 
- DQN_Cartpole_final.py is the implementation of the DQN algo on cartpole domain
- Q_SARSA_VF_final.py is the implementation of the q-learning and sarsa algo on cliff-walking domain but with the value function (not world-value function)
- 

### Packages needed

There are certain packages needed, but for certain algorithms certain versions of these packages were needed.
