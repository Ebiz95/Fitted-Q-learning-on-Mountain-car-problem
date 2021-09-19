import gym
import numpy as np
import csv

from Agent import AgentBilinear

env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)

# all_num_examples = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
# all_max_action = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# all_horizon = [i for i in range(1, 11)]
all_num_examples = [400]	# All number of examples to be tested
all_max_action = [1]		# All max actions to be tested
all_horizon = [None]		# All horizons to be tested (None forces the algorithm to keep computing until convergence)
num_averaging = 1			# Number of times each set of parameters should run (to get a better averaged result)
max_iterations = 500		# Max number of steps allowed in the game
to_csv = False				# Setting this to True exports the results to a csv file
data = []
render_video = True			# Allows the user to visualize the performance of the algorithm (Set to True only when
							# testing one set of parameters.

for num_examples in all_num_examples:
    for max_action in all_max_action:
        for horizon in all_horizon:
            avg_iterations = 0
            min_iterations = max_iterations
            for _ in range(num_averaging):
                state = env.reset()
                agent = AgentBilinear(env, min_action=-max_action, max_action=max_action)
                agent.train(num_examples, horizon=horizon)

                for i in range(1, max_iterations + 1):
                    action = agent.get_action(state)
                    state, r, done, _ = env.step(action)
                    if render_video:
                        env.render()
                    if done:
                        break
                if i < min_iterations:
                    min_iterations = i
                avg_iterations += i
            avg_iterations = avg_iterations / num_averaging
            data.append([num_examples, max_action, horizon, avg_iterations, min_iterations])
            print(f'Num ex: {num_examples}, max a: {max_action}, h: {horizon}, '
            f'AvgIter: {avg_iterations}, MinIter: {min_iterations}')

if to_csv:
    header = ['NumExamples', 'MaxAction', 'Horizon', 'AvgIter', 'MinIter']
    path = 'results.csv'
    with open('results.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerows(data)
