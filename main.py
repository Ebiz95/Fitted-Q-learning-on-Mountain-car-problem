import gym
import time

from Agent import AgentBilinear

env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)

num_examples = 400	# All number of examples to be tested
max_action = 1		# All max actions to be tested
max_iterations = 500		# Max number of steps allowed in the game
render_video = True			# Allows the user to visualize the performance of the algorithm (Set to True only when
							# testing one set of parameters.

state = env.reset()
agent = AgentBilinear(env, min_action=-max_action, max_action=max_action)
agent.train(num_examples)

for _ in range(max_iterations):
    action = agent.get_action(state)
    state, r, done, _ = env.step(action)
    if render_video:
        env.render()
        time.sleep(0.005) # Makes it easier to see the results
    if done:
        break
