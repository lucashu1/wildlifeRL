### DDPG Implementation for wildlife security games ###

import tensorflow as tf
from keras import backend as K
import numpy as np
from random import random

from Env_simulation import Env_simulation
from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer
from Reward_chart import Reward_chart

### TODO - 5/22/17 ###
# (Extra testing)
# Decaying epsilon-greedy exploration for mu/sigmas OR Ornstein-Uhlenbeck noise
# Batch normalization (DDPG paper said this was important)
# Try: add average adversary locations to state
# Try: add dynamic adversary NN


# DDPG Hyperparameters
BUFFER_SIZE = 10000
BUFFER_BATCH_SIZE = 10
MAX_EPISODES = 10000
ROLLOUTS_PER_EPISODE = 100
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
EPSILON_INIT = 0.8 # for epsilon-greedy exploration
EPSILON_DECAY = 0.995

# Environment Setup
GRID_LEN = 10
NUM_FEATURES = 1
NUM_DEFENDERS = 1
NUM_ADVERSARIES = 1
NUM_ANIMALS = 75

# Pre-DDPG Initializations
state_size = GRID_LEN*GRID_LEN*NUM_FEATURES + NUM_DEFENDERS*2
action_size = NUM_DEFENDERS*4

sess = tf.Session()
K.set_session(sess)

env = Env_simulation(GRID_LEN, NUM_FEATURES, NUM_ANIMALS, NUM_ADVERSARIES, NUM_DEFENDERS)
buff = ReplayBuffer(buffer_size = BUFFER_SIZE)
actor = Actor(sess, state_size, action_size, LR_ACTOR, TAU)
critic = Critic(sess, state_size, action_size, LR_CRITIC, TAU)

def_avg_coords = np.zeros(NUM_DEFENDERS*2)

chart = Reward_chart()
epsilon = EPSILON_INIT # for epsilon-greedy exploration

### DDPG Algorithm ###
for episode in range(0, MAX_EPISODES):
	
	env.create_episode()
	grid_vec = env.grid_vector()[0]
	s_t = np.append(grid_vec, def_avg_coords)

	# Epsilon-greedy exploration in action space (mu_sigmas)
	if random() < epsilon:
		a_t = np.random.rand(action_size) # uniform random between 0-1
	else:
		a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))[0] # defender mu/sigma values

	reward_sum = 0
	def_coords_list = []

	# Play out some games with our new mu/sigmas
	for rollout in range(0, ROLLOUTS_PER_EPISODE):
		# place defenders and adversaries, calculate reward
		def_loc = env.place_defenders(a_t)
		adv_loc = env.place_adversaries()
		cur_reward = env.calc_reward_no_coverage()
		reward_sum += cur_reward

		# store defender locations
		def_coords = def_loc[0][0]
		# print("Def coords: ", def_coords)
		def_coords_list.append(def_coords)

		# print("Def coords list: ", def_coords_list)

	r_t = reward_sum / ROLLOUTS_PER_EPISODE # avg reward over rollouts
	def_avg_coords = np.mean(def_coords_list, axis=0) # average over observed defender coords

	# New state: grid features + average defender coords
	s_t1 = np.append(grid_vec, def_avg_coords)
	buff.buffer_add(s_t, a_t, r_t, s_t1)

	# Batch update
	batch = buff.get_elements(BUFFER_BATCH_SIZE)
	states = np.asarray([e[0] for e in batch])
	actions = np.asarray([e[1] for e in batch])
	rewards = np.asarray([e[2] for e in batch])
	new_states = np.asarray([e[3] for e in batch])

	y_t = np.asarray([e[2] for e in batch]) # to be used for target Q vals

	# Update critic
	target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
	rewards_vector = [[r]*action_size for r in rewards] # make dimension match target_q_values
	# print("Rewards: ", rewards)
	# print("Target q values: ", target_q_values)
	y_t = rewards_vector + GAMMA*target_q_values
	loss = critic.model.train_on_batch([states,actions], y_t)

	# Update actor
	a_for_grad = actor.model.predict(states)
	grads = critic.gradients(states, a_for_grad)
	actor.train(states, grads)

	# Update target networks
	actor.update_actor_target()
	critic.update_critic_target()

	# Gradually decrease exploration
	epsilon *= EPSILON_DECAY

	# Print to terminal
	print("Episode: ", episode)
	print("Epsilon: ", epsilon)
	# print("S_t", s_t)
	print("Defender mu_sigma (a_t): ", a_t)
	# print("Defender locations list: ", def_coords_list)
	print("Defender average coords (row, col): ", def_avg_coords)
	print("Average reward (r_t): ", r_t)
	print("Critic Loss: ", loss)
	print()

	chart.add_timestep(episode, r_t)


chart.show_episode()






