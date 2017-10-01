### implementing the simple network - gradient update #####
# ** means to be updated later
from Env_stackelberg import Env_simulation
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras import backend as K
from Reward_chart import Reward_chart
import tensorflow as tf
import numpy as np
import csv
import random

#basic algorithm ###
#constants
num_cells = 100
NUM_FEATURES = 1
NUM_DEFENDERS = 1
NUM_ADVERSARIES = 5

gradient_experience_buffer = []
def_network_update = 5 #update the network after _____ episodes

gradient_reward_buffer = [] # for testing the normalization of reward values

#csv writer setup
csvfile_deflocs = open('logger_def.csv','w')
csvfile_rewards = open('logger_rewards.csv','w')
csvfile_adversaries = open('logger_adversaries.csv','w')

# tensorflow setup
sess = tf.Session()
K.set_session(sess) # connects Keras models to this tf session

#setup initial def network - same as in the defenders.py
defender_model = Sequential()
defender_model.add(Dense(500, input_dim=num_cells*NUM_FEATURES,init='glorot_normal', activation='sigmoid'))
defender_model.add(Dense(NUM_DEFENDERS*4, init='glorot_normal', activation='sigmoid'))

# more tensorflow initialization
defender_output_tensor = defender_model.output
defender_param_tensor = defender_model.trainable_weights
optimizer = tf.train.AdamOptimizer(0.001)
#gradients = tf.gradients(defender_output_tensor, defender_param_tensor)


#this is used for accumulating the scaled gradients
processed_buffer = tf.placeholder(tf.float32)

print(tf.global_variables())

sess.run(tf.global_variables_initializer())

#initialize the environment object 
env =  Env_simulation(num_defenders=NUM_DEFENDERS, \
	num_features=NUM_FEATURES, num_adversaries = NUM_ADVERSARIES)

chart = Reward_chart()

# try placing animals/adversaries just once in beginning
# grid,animal_placements =  env.create_episode() #call create episode
# adv_loc, _ = env.place_adversaries()
# print("Adversary locations: ", adv_loc)

csvfile_deflocs.write("Episode,Defender_mu_x,Defender_sigma_x,Defender_mu_y,Defender_sigma_y,Avg_Reward\n")

#create 100 episodes
for i in range(1,1001):
	grid,animal_placements =  env.create_episode() #call create episode

	grid_vec = env.grid_vector()
	#get a prediction
	defender_mu_sigma = defender_model.predict(grid_vec)[0]

	# reward_sum = random.randint(-5,50) #reward accumulator
	reward_sum = 0


	#play 100 games to accumulate rewards
	for s in range(1,101): #sample 100 games with each mu_sigma distribution **
			
		def_loc = env.place_defenders(defender_mu_sigma)
		
		#more file handling - store sampled defender locations
		# csvfile_deflocs.write(str(def_loc[0]))
		# csvfile_deflocs.write("\n")
		
		#adversary placements		
		adv_loc = env.place_adversaries()
		#adversary file handling
		# csvfile_adversaries.write(str(adv_loc[0]))
		# csvfile_adversaries.write("\n")
		
		#reward computation for current game
		cur_reward = env.calc_reward_no_coverage()
			
		#accumulate rewards
		reward_sum += cur_reward
		
	avg_reward = reward_sum/100



	#file handling - storing defender locations			
	# csvfile_deflocs.write("Episode: ")
	csvfile_deflocs.write(str(i))
	csvfile_deflocs.write(",")
	#write the mu-sigma for the defender locations
	for e in defender_mu_sigma:
		csvfile_deflocs.write(str(e))
		csvfile_deflocs.write(",")
	csvfile_deflocs.write(str(avg_reward))
	csvfile_deflocs.write("\n")
	
	# #write rewards to a file - used for graphs
	# csvfile_rewards.write(str(i))
	# csvfile_rewards.write(",")
	# csvfile_rewards.write(str(avg_reward))
	# csvfile_rewards.write("\n")


	chart.add_timestep(i, avg_reward)
	
	
	
	# generate gradient tensor for current episode
	grad_vars_tensor = optimizer.compute_gradients(defender_output_tensor, defender_param_tensor)
	
	# evaluate gradient wrt. current weights for this episode
	# store in list of (grad, var) pairs where grad = Tf constant, var = reference to parameter variable
	# length 4: (W1 grads, W1 vars), (b1 grads, b1 vars), (W2 grads, W2 vars), (b2 grads, b2 vars)
	
	gradient_values = [(sess.run(grad, feed_dict={defender_model.input:grid_vec}), var) \
		for grad, var in grad_vars_tensor]
	evaluated_grad_vars = [(tf.constant(grad), var) for grad, var in gradient_values]

	# print(evaluated_grad_vars)

	# scale gradient values by -(reward)
	scaled_grad_vars = [(tf.multiply(g, -avg_reward), v) for g, v in evaluated_grad_vars]

	# print(scaled_grad_vars)

	# append scaled gradient list to buffer (represents one experience)
	gradient_experience_buffer.append(scaled_grad_vars)

	# For testing purposes: keep gradient and reward separate for now
	# Normalize the reward values later to encourage approximately 1/2 of the actions, discourage other 1/2
	gradient_reward_buffer.append((evaluated_grad_vars, avg_reward))

	# print("episode",i,"reward",avg_reward)
	
	#-------------end of episode computation---------------------#




	# Seeing what's happening to the weights/biases over time
	if i % def_network_update == 1:
		# weights1 = sess.run(tf.global_variables())[0]
		# bias1 = sess.run(tf.global_variables())[1]
		# weights2 = sess.run(tf.global_variables())[2]
		# bias2 = sess.run(tf.global_variables())[3]

		# weights1_avg = np.mean(weights1)
		# weights2_avg = np.mean(weights2)

		# bias1_avg = np.mean(bias1)
		# bias2_avg = np.mean(bias2)

		print("Episode: ", i, "\n")

		# print("Weights 1: ", weights1)
		# print("Layer 1 Weights average: %.29f" % weights1_avg)
		# print("Layer 1 Bias average: %.29f" % bias1_avg, "\n")

		# print("Weights 2: ", weights2)
		# print("Layer 2 Weights average: %.29f" % weights2_avg)
		# print("Layer 2 Bias average: %.29f" % bias2_avg, "\n")

		print("Defender mu_sigma: ", defender_mu_sigma)
		print("Average reward:", avg_reward)
		# print("\n")
		# print("---------------------------------------------------------------------")







	### *************** ####
	# The part below is the update to the optimizer ########
	# we change the policy gradient model a bit - 
	# take a sum of (output_tensor*r) and then take the derivative w.r.t weights of that
	
	#-------------------METHOD 1------------------------------------#
	#update the network every 10 times
		
	# if i%def_network_update == 0:
	# 	counter = 0
	# 	grad_list = []
	# 	for b in buffer:
	# 		input = b[0]
	# 		output_tensor = b[1]
	# 		reward = b[2]
	# 		if counter == 0:
	# 			processed_buffer = tf.scalar_mul(-reward,output_tensor)
	# 		else:
	# 			processed_buffer = tf.add(tf.scalar_mul(-reward,output_tensor), processed_buffer)
	# 		counter +=1
	# 	gradients = optimizer.compute_gradients(processed_buffer, defender_param_tensor)
	# 	update_weights = optimizer.apply_gradients(gradients)
	# 	input = np.asarray([b[0] for b in buffer])
	# 	input = np.reshape(input, (len(buffer),num_cells*NUM_FEATURES))
	# 	sess.run(update_weights,feed_dict={defender_model.input:input})
			
	#-----------------------------METHOD 2----------------------------------------#
	# alternate approach
	# we multiply the reward and the output and then take gradients using tf.gradients
	#d(constant*f(w)<-model.output) = constant*d(f(w))
	
	if i%def_network_update == 0:
		

		# PRINT WEIGHTS
		# weights1 = sess.run(tf.global_variables())[0]
		# bias1 = sess.run(tf.global_variables())[1]
		# weights2 = sess.run(tf.global_variables())[2]
		# bias2 = sess.run(tf.global_variables())[3]

		# weights1_avg = np.mean(weights1)
		# weights2_avg = np.mean(weights2)

		# bias1_avg = np.mean(bias1)
		# bias2_avg = np.mean(bias2)

		# print("Episode: ", i, "\n")

		# # print("Weights 1: ", weights1)
		# print("Layer 1 Weights average: %.29f" % weights1_avg)
		# print("Layer 1 Bias average: %.29f" % bias1_avg, "\n")

		# # print("Weights 2: ", weights2)
		# print("Layer 2 Weights average: %.29f" % weights2_avg)
		# print("Layer 2 Bias average: %.29f" % bias2_avg, "\n")

		# print("Defender mu_sigma: ", defender_mu_sigma, "\n")
		# print("Average reward:", avg_reward, "\n")

		# ----------------------------- NETWORK UPDATE -------------------------------------

		# NORMALIZE REWARD VALUES, SCALE GRADIENTS BY -NORMALIZED_REWARD
		raw_rewards = [reward for gradient, reward in gradient_reward_buffer]
		mean = np.mean(raw_rewards)
		stddev = np.std(raw_rewards)
		normalized_rewards = (raw_rewards - mean)/stddev

		# print("Raw: ", raw_rewards)
		# print("Normalized: ", normalized_rewards)

		# scale gradients by -normalized_reward
		normalized_experience_buffer = []
		grad_vars = [grad_var for grad_var, raw_reward in gradient_reward_buffer]
		for i in range(len(grad_vars)):
			normalized_reward = normalized_rewards[i]
			grad_var = grad_vars[i]
			scaled_normalized = [(tf.multiply(g, -normalized_reward), v) for g, v in grad_var]
			normalized_experience_buffer.append(scaled_normalized)

		


		# SAMPLE EXPERIENCES FROM BUFFER
		NUM_SAMPLES = 10
		if (NUM_SAMPLES < len(gradient_experience_buffer)):
			sampled_episode_nums = random.sample(range(len(gradient_experience_buffer)), NUM_SAMPLES)
			# sampled_experiences = [gradient_experience_buffer[i] for i in sampled_episode_nums]
			sampled_experiences = [normalized_experience_buffer[i] for i in sampled_episode_nums]

			print("Sampled episodes: ", sampled_episode_nums, "\n")

		else:
			sampled_experiences = normalized_experience_buffer
			print("Not enough experiences -- using entire buffer", "\n")

		# ACCUMULATE SCALED GRADIENTS FORM BUFFER
		episode_counter = 0
		for episode_grad_vars in sampled_experiences:
			if episode_counter == 0:
				summed_grad_vars = episode_grad_vars
			else:
				# Param_ind Indexing: 0 = W1, 1 = b1, 2 = W2, 3 = b2
				for param_ind in range(len(episode_grad_vars)): 
					# E.g.: add new episode's scaled W1 gradient to accumulated W1 gradient
					#[0] at the end specifies to add the gradient tensors, not the weight variable
					summed_grad_vars[param_ind][0] = \
						tf.add(summed_grad_vars[param_ind][0], episode_grad_vars[param_ind][0])



		
		# # PRINT GRADIENTS
		# final_grad_vars = sess.run(summed_grad_vars)
		# grad_vals_list = []

		# for (grad, weights) in final_grad_vars:
		# 	grad_vals_list.append(grad)

		# # print(len(grad_vals_list))

		# w1_gradients = grad_vals_list[0]
		# b1_gradients = grad_vals_list[1]
		# w2_gradients = grad_vals_list[2]
		# b2_gradients = grad_vals_list[3]

		# print("W1 mean gradient: ", np.mean(w1_gradients))
		# print("b1 mean gradient: ", np.mean(b1_gradients))
		# print("W2 mean gradient: ", np.mean(w2_gradients))
		# print("b2 mean gradient: ", np.mean(b2_gradients), "\n")

		


		# PERFORM UPDATE (and initialize AdamOptimizer variables if necessary)
		# Current list of variables
		temp = set(tf.global_variables())

		# GRADIENT UPDATE
		update_weights = optimizer.apply_gradients(summed_grad_vars)
		# no feed_dict necessary because gradients pre-calculated at time of episode

		uninitialized_vars_names = sess.run(tf.report_uninitialized_variables())
		if len(uninitialized_vars_names) > 0:
			# Initialize new variables: set of (all - old)
			sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))
			print("Initialized Adam variables")

		# uninitialized_vars = []

		# for var_name in uninitialized_vars_names:
		# 	var = tf.get_variable(var_name)
		# 	uninitialized_vars.append(var)

		# print("Uninitialized: ", uninitialized_vars)
		# print(len(uninitialized_vars))
		# if i == def_network_update:
		# 	init_new_vars = tf.variables_initializer(uninitialized_vars)
		# 	sess.run(init_new_vars)
		# 	print("Initialized new variables")

		# sess.run( tf.variables_initializer( list( tf.get_variable(name) \
		# 	for name in sess.run( tf.report_uninitialized_variables() ) ) ) )
		
		sess.run(update_weights) 






		print("---------------------------------------------------------------------")

chart.show_episode()
input("All episodes completed. Press [enter] to exit")


		


		
		
	
 