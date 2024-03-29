import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='ansi').env

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print('\n\n')

env.reset()
print(env.render())
# escolhe uma acao aleatoria
action = env.action_space.sample()
# executa a acao
state, reward, done, truncated, info = env.step(action)
print(env.render())
# executa a acao ir para north
state, reward, done, truncated, info = env.step(1)
print(env.render())


# The filled square represents the taxi, which is yellow without a passenger and green with a passenger.
# The pipe ("|") represents a wall which the taxi cannot cross.
# R, G, Y, B are the possible pickup and destination locations. The blue letter represents the current 
# passenger pick-up location, and the purple letter is the current destination.

# actions:
# 0 = south
# 1 = north
# 2 = east
# 3 = west
# 4 = pickup
# 5 = dropoff

env.close()