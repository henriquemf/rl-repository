import random
from IPython.display import clear_output
import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearning import QLearning
from Sarsa import Sarsa

params = {
    'alpha': 0.1, 
    'gamma': 0.99, 
    'epsilon': 0.1, 
    'epsilon_min': 0.1, 
    'epsilon_dec': 1, 
    'episodes': 5000
}

def calculate_moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def train_and_track_rewards(env_name, algorithm, params):
    env = gym.make(env_name).env

    algorithm_classes = {
        'QLearning': QLearning,
        'Sarsa': Sarsa
    }

    if algorithm not in algorithm_classes:
        raise ValueError("Invalid algorithm name. Use 'QLearning' or 'Sarsa'")

    agent_class = algorithm_classes[algorithm]
    agent = agent_class(env, **params) 

    filename = f'data/cliff_walking_{algorithm.lower()}_q_table.csv'
    q_table = agent.train(filename, plotFile=None)
    return q_table

results = {}
print(results)
algorithms = ['QLearning', 'Sarsa']

for algorithm in algorithms:
    results[algorithm] = train_and_track_rewards("CliffWalking-v0", algorithm, params.copy())

window_size = 50  

plt.figure(figsize=(10, 6))

for algorithm, rewards in results.items():
    smoothed_rewards = calculate_moving_average(rewards, window_size)
    plt.plot(smoothed_rewards, label=algorithm)

plt.xlabel('Episodes')
plt.ylabel('Rewards (Moving Average)')
plt.title('Rewards over Episodes in Cliff Walking (Moving Average)')
plt.legend()
plt.grid(True)
plt.savefig('results/cliff_walking_rewards_qlearning_sarsa_ma.jpg')
plt.show()

env = gym.make("CliffWalking-v0", render_mode="human").env  
q_table = np.loadtxt('data/cliff_walking_qlearning_q_table.csv', delimiter=',')

state, _ = env.reset()
rewards = 0
actions = 0
done = False

while not done:
    print(state)
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)
    rewards += reward
    actions += 1

env.close()
print("\n")
print("Actions taken: {}".format(actions))
print("Rewards: {}".format(rewards))