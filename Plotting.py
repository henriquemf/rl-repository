import random
from IPython.display import clear_output
import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearning import QLearning 
from Sarsa import Sarsa

# ------------------ TAXI DRIVER ------------------

env = gym.make("Taxi-v3", render_mode='ansi').env

qlearn = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_min=0.1, epsilon_dec=1, episodes=5000)
sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_min=0.1, epsilon_dec=1, episodes=5000)

qtable_qlearn, moving_avg_qlearn = qlearn.train('data/taxi_ql_q_table.csv', 'results/taxi_ql_rewards.jpg')
qtable_sarsa, moving_avg_sarsa = sarsa.train('data/taxi_sarsa_q_table.csv', 'results/taxi_sarsa_rewards.jpg')

plt.figure(figsize=(10, 6))
plt.plot(moving_avg_qlearn, label='QLearning')
plt.plot(moving_avg_sarsa, label='Sarsa')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards vs Episodes in Taxi Driver')
plt.legend()
plt.grid(True)
plt.savefig('results/taxi_rewards_qlearning_sarsa.jpg')
plt.show()

# ------------------ CLIFF WALKING ------------------

env = gym.make("CliffWalking-v0").env

qlearn = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_min=0.1, epsilon_dec=1, episodes=5000)
sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_min=0.1, epsilon_dec=1, episodes=5000)

qtable_qlearn, moving_avg_qlearn = qlearn.train('data/cliff_ql_q_table.csv', 'results/cliff_ql_rewards.jpg')
qtable_sarsa, moving_avg_sarsa = sarsa.train('data/cliff_sarsa_q_table.csv', 'results/cliff_sarsa_rewards.jpg')

plt.figure(figsize=(10, 6))
plt.plot(moving_avg_qlearn, label='QLearning')
plt.plot(moving_avg_sarsa, label='Sarsa')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards vs Episodes in Cliff Walking')
plt.legend()
plt.grid(True)
plt.savefig('results/cliff_rewards_qlearning_sarsa.jpg')
plt.show()