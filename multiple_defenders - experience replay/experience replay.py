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
import random

#basic algorithm ###
#constants
num_cells = 100
NUM_FEATURES = 7
NUM_DEFENDERS = 5
buffer = []
def_network_update = 5 #update the network after _____ episodes

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
optimizer = tf.train.GradientDescentOptimizer(0.1)
#gradients = tf.gradients(defender_output_tensor, defender_param_tensor)


#this is used for accumulating the scaled gradients
processed_buffer = tf.placeholder(tf.float32)


sess.run(tf.global_variables_initializer())

#initialize the environment object 
env =  Env_simulation(num_defenders=NUM_DEFENDERS)



#create 100 episodes
for i in range(1,101):
	grid,animal_placements =  env.create_episode() #call create episode

	grid_vec = env.grid_vector()
	#get a prediction
	defender_mu_sigma = defender_model.predict(grid_vec)[0]

	#file handling - storing defender locations			
	csvfile_deflocs.write("Episode: ")
	csvfile_deflocs.write(str(i))
	csvfile_deflocs.write("\n")
	#write the mu-sigma for the defender locations
	for e in defender_mu_sigma:
		csvfile_deflocs.write(str(e))
		csvfile_deflocs.write(",")
	csvfile_deflocs.write("\n")
		

	# reward_sum = random.randint(-5,50) #reward accumulator
	reward_sum = 0
	#play 100 games to accumulate rewards
	for s in range(0,101): #sample 100 games with each mu_sigma distribution **
			
		def_loc = env.place_defenders(defender_mu_sigma)
		
		#more file handling - store sampled defender locations
		csvfile_deflocs.write(str(def_loc[0]))
		csvfile_deflocs.write("\n")
		
		#adversary placements		
		adv_loc = env.place_adversaries()
		#adversary file handling
		csvfile_adversaries.write(str(adv_loc[0]))
		csvfile_adversaries.write("\n")
		
		#reward computation for current game
		cur_reward = env.calc_reward_no_coverage()
			
		#accumulate rewards
		reward_sum += cur_reward
		
	avg_reward = reward_sum/100
	
	#write rewards to a file - used for graphs
	csvfile_rewards.write(str(i))
	csvfile_rewards.write(",")
	csvfile_rewards.write(str(avg_reward))
	csvfile_rewards.write("\n")
	
	#store output tensor and net reward in a buffer
	buffer.append((grid_vec,defender_model.output,avg_reward))

	# print("episode",i,"reward",avg_reward)
	
	#-------------end of episode computation---------------------#




	# Seeing what's happening to the weights/biases over time
	# if i % def_network_update == 1:
		# weights1 = sess.run(tf.global_variables())[0]
		# bias1 = sess.run(tf.global_variables())[1]
		# weights2 = sess.run(tf.global_variables())[2]
		# bias2 = sess.run(tf.global_variables())[3]

		# weights1_avg = np.mean(weights1)
		# weights2_avg = np.mean(weights2)

		# bias1_avg = np.mean(bias1)
		# bias2_avg = np.mean(bias2)

		# print("Episode: ", i, "\n")

		# print("Weights 1: ", weights1)
		# print("Layer 1 Weights average: %.29f" % weights1_avg)
		# print("Layer 1 Bias average: %.29f" % bias1_avg, "\n")

		# print("Weights 2: ", weights2)
		# print("Layer 2 Weights average: %.29f" % weights2_avg)
		# print("Layer 2 Bias average: %.29f" % bias2_avg, "\n")

		# print("Defender mu_sigma: ", defender_mu_sigma)
		# print("Average reward:", avg_reward)
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
		weights1 = sess.run(tf.global_variables())[0]
		bias1 = sess.run(tf.global_variables())[1]
		weights2 = sess.run(tf.global_variables())[2]
		bias2 = sess.run(tf.global_variables())[3]

		weights1_avg = np.mean(weights1)
		weights2_avg = np.mean(weights2)

		bias1_avg = np.mean(bias1)
		bias2_avg = np.mean(bias2)

		print("Episode: ", i, "\n")

		# print("Weights 1: ", weights1)
		print("Layer 1 Weights average: %.29f" % weights1_avg)
		print("Layer 1 Bias average: %.29f" % bias1_avg, "\n")

		# print("Weights 2: ", weights2)
		print("Layer 2 Weights average: %.29f" % weights2_avg)
		print("Layer 2 Bias average: %.29f" % bias2_avg, "\n")

		# print("Defender mu_sigma: ", defender_mu_sigma)
		# print("Average reward:", avg_reward, "\n")
		# print("\n")




		# METHOD 1
		counter = 0
		grad_list = []
		for b in buffer:
			input_vec = b[0]
			output_tensor = b[1]
			
			reward = b[2]
			# reward = np.random.normal(0, 0.1) # testing randomly generated, zero-centered award

			if counter == 0:
				processed_buffer = tf.scalar_mul(-reward,output_tensor) # scaled output
			else:
				processed_buffer = tf.add(tf.scalar_mul(-reward,output_tensor), processed_buffer)
			counter +=1
			
			# display calculation process
			print("Counter: ", counter)
			print("Reward: ", reward)
			print ("Mean output: ", np.mean(sess.run(output_tensor, feed_dict={defender_model.input:input_vec})))
			print ("Mean scaled output: ", np.mean(sess.run(processed_buffer, \
				feed_dict={defender_model.input:input_vec})), "\n")

		method1_gradients = optimizer.compute_gradients(processed_buffer, defender_param_tensor)
		# print("Method 1 gradients: ", sess.run(method1_gradients, feed_dict={defender_model.input:input_vec}))
		# print("Method 1 gradients: ", np.array(sess.run(method1_gradients, feed_dict={defender_model.input:input_vec})).shape)
		# print("Method 1 gradients: ", method1_gradients)
		update_weights1 = optimizer.apply_gradients(method1_gradients)
		
		input_vecs = np.asarray([b[0] for b in buffer])
		input_vecs = input_vecs.reshape(len(buffer),num_cells*NUM_FEATURES)

		# testing gradient values
		gradient_vals1 = np.array(sess.run(method1_gradients, feed_dict={defender_model.input:input_vecs}))
		# print("Method 1 gradients: \n")
		# print("Method 1 gradient shape: ", method1_gradients)


		# PRINT MEAN GRADIENTS
		gradients_lists = [tf.reshape(g, [-1]) for g in tf.gradients(processed_buffer, defender_param_tensor)]
		# gradient_list = gradients_lists[0]+gradients_lists[1]+gradients_lists[2]+gradients_lists[3]
		# gradient_mean = sess.run(tf.reduce_mean(gradients_list), \
		# 	feed_dict={defender_model.input:input_vecs})
		print("W1 mean gradient: ", sess.run(tf.reduce_mean(gradients_lists[0]), \
			feed_dict={defender_model.input:input_vecs}))
		print("B1 mean gradient: ", sess.run(tf.reduce_mean(gradients_lists[1]), \
			feed_dict={defender_model.input:input_vecs}))
		print("W2 mean gradient: ", sess.run(tf.reduce_mean(gradients_lists[2]), \
			feed_dict={defender_model.input:input_vecs}))
		print("B2 mean gradient: ", sess.run(tf.reduce_mean(gradients_lists[3]), \
			feed_dict={defender_model.input:input_vecs}))

		# 1st weight and gradient
		print("\n1st layer 1st weight: %.29f" % weights1[0][0])
		print("1st layer 1st gradient: %.29f" % gradient_vals1[0][0][0][0], "\n")
		
		sess.run(update_weights1,feed_dict={defender_model.input:input_vecs})


		print("\n---------------------------------------------------------------------")




		# # METHOD 2
		# output_tensor_list = []
		# for b in buffer:
		# 	input_vec = b[0]
		# 	output_tensor = tf.scalar_mul(-b[2],b[1]) # (-reward)*output
		# 	output_tensor_list.append(output_tensor)
		# method2_gradients = tf.gradients(output_tensor_list,defender_param_tensor)
		# grad_vars = zip(method2_gradients,defender_param_tensor)
		# update_weights2 = optimizer.apply_gradients(grad_vars)
		
		# input_vecs = np.asarray([b[0] for b in buffer]) # list of all input vectors
		# input_vecs = input_vecs.reshape(len(buffer),num_cells*NUM_FEATURES)

		# # # testing gradient values
		# # gradient_vals2 = np.array(sess.run(method2_gradients, feed_dict={defender_model.input:input_vecs}))
		# # print("Method 2 gradients: \n")
		# # print("Method 2 gradient shape: ", method2_gradients, defender_param_tensor)
		# # print(gradient_vals2)

		# sess.run(update_weights2,feed_dict={defender_model.input:input_vecs})




		
	#--------------------Problems?-----------------------------#
	# -- As the size of the buffer grows so the input and does the whole update takes longer
	#-- the gaussians shrink for the defender which makes sampling difficult------------------
		
	#--------------More ways of extension------------------------#
	# restrict the no. of elements sampled from the buffer
		#smart sampling techniques that exploit similiarities
		# look into more literature of sampling techniques
			
			
			

	
		
		
		
		
		
		
		
		
	
 