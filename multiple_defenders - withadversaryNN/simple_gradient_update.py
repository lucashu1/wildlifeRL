### implementing the simple network - gradient update #####
# ** means to be updated later
from Env_simulation import Env_simulation
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras import backend as K
from Reward_chart import Reward_chart
import tensorflow as tf
import numpy as np
import csv

#basic algorithm ###
#constants
num_cells = 100
NUM_FEATURES = 7
NUM_DEFENDERS = 5

#csv writer setup
csvfile_deflocs = open('logger_def.csv','w')
csvfile_rewards = open('logger_rewards.csv','w')
csvfile_adversaries = open('logger_adversaries.csv','w')

# tensorflow setup
sess = tf.Session()
K.set_session(sess) # connects Keras models to this tf session

#setup initial def network - same as in the defenders.py
defender_model = Sequential()
defender_model.add(Dense(500, input_dim=num_cells*NUM_FEATURES,init='uniform', activation='sigmoid'))
defender_model.add(Dense(NUM_DEFENDERS*4, init='uniform', activation='sigmoid'))

# more tensorflow initialization
defender_output_tensor = defender_model.output
defender_param_tensor = defender_model.trainable_weights
optimizer = tf.train.GradientDescentOptimizer(0.1)
gradients = optimizer.compute_gradients(defender_output_tensor, defender_param_tensor)

sess.run(tf.global_variables_initializer())

#initialize the environment object 
env =  Env_simulation(num_defenders=NUM_DEFENDERS)
#print the parameters for the adversary for the optimization code

env.print_adversary_params()

chart = Reward_chart()

#create 5 episodes
for i in range(1,5):
	grid,animal_placements =  env.create_episode() #call create episode

	chart_rewards = []
	chart_timesteps = []

	for t in range(1,10): #run the process through 1000 timesteps **
		#get the list of mu_sigma from the current grid setup
		#reshape grid and call def_model.predict as in defenders.py

		grid_vec = env.grid_vector()
		defender_mu_sigma = defender_model.predict(grid_vec)[0]

		#print(defender_mu_sigma) # this becomes almost zero after the first update

		
		# csvfile_deflocs.write("Timestep: ")
		# csvfile_deflocs.write(str(t))
		# csvfile_rewards.write(str(t))
		# csvfile_rewards.write(",")
		# csvfile_deflocs.write("\n")
		# for e in defender_mu_sigma:
		# 	csvfile_deflocs.write(str(e))
		# 	csvfile_deflocs.write(",")
		# csvfile_deflocs.write("\n")
		

		reward_sum = 0 #reward accumulator
		for s in range(0,100): #sample 100 games with each mu_sigma distribution **
			def_loc = env.place_defenders(defender_mu_sigma)
			# csvfile_deflocs.write(str(def_loc[0]))
			# csvfile_deflocs.write("\n")
			adv_loc = env.place_adversaries()
			# csvfile_adversaries.write(str(adv_loc[0]))
			# csvfile_adversaries.write("\n")
			cur_reward = env.calc_reward_no_coverage()
			
			reward_sum += cur_reward
		
		avg_reward = reward_sum/100

		#print out reward_sum obtained also mention which timestep
		#check every 500 timestep
		# csvfile_rewards.write(str(avg_reward))
		# csvfile_rewards.write("\n")
		# csvfile_deflocs.write("-----------------------------------\n")
		
		#print("Timestep: ", t, " Avg reward: ", avg_reward)

		chart.add_timestep(t, avg_reward)
		
		# if t == 1 or t % 10 == 0:
		print("Timestep: ", t, " Avg reward: ", avg_reward)




		# # Seeing what's happening to the weights/biases over time
		# if t % 10 == 1:
		# 	weights1 = sess.run(tf.global_variables())[0]
		# 	bias1 = sess.run(tf.global_variables())[1]
		# 	weights2 = sess.run(tf.global_variables())[2]
		# 	bias2 = sess.run(tf.global_variables())[3]

		# 	weights1_avg = np.mean(weights1)
		# 	weights2_avg = np.mean(weights2)

		# 	bias1_avg = np.mean(bias1)
		# 	bias2_avg = np.mean(bias2)

		# 	print("Timestep: ", t, "\n")

		# 	# print("Weights 1: ", weights1)
		# 	print("Layer 1 Weights average: ", weights1_avg)
		# 	print("Layer 1 Bias average: ", bias1_avg)
		# 	print("\n")

		# 	# print("Weights 2: ", weights2)
		# 	print("Layer 2 Weights average: ", weights2_avg)
		# 	print("Layer 2 Bias average: ", bias2_avg)
		# 	print("\n")

		# 	print("Defender mu_sigma: ", defender_mu_sigma)
		# 	print("Average reward:", avg_reward)
		# 	print("\n")
		# 	print("---------------------------------------------------------------------")



		
		
		### *************** ####
		# The part below is the update to the optimizer ########
		
		# updating the gradient of the defender network
		scaled_grad_var = [(tf.multiply(g, -avg_reward), v) for g, v in gradients]

		update_weights = optimizer.apply_gradients(scaled_grad_var)
		

		sess.run(update_weights, feed_dict={
			defender_model.input: grid_vec
		})

	chart.show_episode()

# Had to add this to keep charts open
input("All episodes completed. Press [enter] to exit")
		
		
		
		
		
		
		
		
	
 