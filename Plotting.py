import random
from IPython.display import clear_output
import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearning import QLearning 

env = gym.make("Taxi-v3", render_mode='ansi').env

variations = [
    {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.7}, 
    {'alpha': 0.3, 'gamma': 0.95, 'epsilon': 0.8}, 
    {'alpha': 0.2, 'gamma': 0.85, 'epsilon': 0.6}  
]

def train_and_track_rewards(env, qlearn_params):
    qlearn_params.setdefault('epsilon_min', 0.01)  
    qlearn_params.setdefault('epsilon_dec', 0.99)
    qlearn_params.setdefault('episodes', 10000)

    agent = QLearning(env, **qlearn_params)
    rewards_history = []
    for i in range(1, episodes+1):
        state, _ = env.reset()
        rewards = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action) 
            agent.update(state, action, next_state, reward, done) 
            state = next_state
            rewards += reward

        rewards_history.append(rewards)

    return rewards_history

episodes = 10000
results = {}  

for params in variations:
    results[str(params)] = train_and_track_rewards(env, params)

plt.figure(figsize=(10, 6))
for label, rewards in results.items():
    plt.plot(rewards, label=label)

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards over Episodes with Hyperparameter Variations')
plt.legend()
plt.grid(True)
plt.savefig('results/taxi_rewards_variations.jpg')
plt.show()