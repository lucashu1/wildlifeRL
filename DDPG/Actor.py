### Actor class ####
from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import numpy as np

class Actor():
	def __init__(self,sess,no_states,no_actions,l_rate,tau):
		self.sess = sess
		self.l_rate = l_rate
		self.tau = tau
		
		self.model,self.weights,self.state = self.create_actor(no_states,no_actions)
		
		K.set_session(sess)
		
		self.target_model,self.target_weights,self.target_states = self.create_actor(no_states,no_actions)
		#this comes from the critic hence is initially empty
		self.action_gradient = tf.placeholder(tf.float32, [None, no_actions]) 
		# combining gradients
		self.actor_param_grads = tf.gradients(self.model.output,self.weights,-self.action_gradient) 
		#apply gradient to optimizer
		temp = zip(self.actor_param_grads,self.weights)
		self.optimize = tf.train.AdamOptimizer(l_rate).apply_gradients(temp) 
		self.sess.run(tf.global_variables_initializer())
		
		
		
	#MOST CRUCIAL STEP OF COMPUTING ACTOR GRADIENT	
	def train(self,input,action_grad):
		self.sess.run(self.optimize,feed_dict={self.state:input,self.action_gradient:action_grad})
	
	#update target weights
	def update_actor_target(self):
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in range(len(self.target_weights)):
			actor_target_weights[i] = self.tau*actor_weights[i]+(1-self.tau)*actor_target_weights[i];
		self.target_model.set_weights(actor_target_weights)
			
	
	def create_actor(self,no_states,no_actions):
		# print('now creating actor model')
		inputs = Input(shape=[no_states])
		layer_1 = Dense(4,init='glorot_uniform',activation='relu')(inputs)
		layer_2 = Dense(3,init = 'glorot_uniform',activation='relu')(layer_1)
		output_layer = Dense(no_actions,init='glorot_uniform',activation='sigmoid')(layer_2);
		m = Model(input=inputs,output=output_layer)
		return m,m.trainable_weights,inputs
		

# ----------------------------- ACTOR TEST -----------------------------
# sess = tf.Session()	
# actor = Actor(sess,100*7,10,0.0001,0.001)
# # NUM_GRIDS = 1
# GRID_LEN = 10
# GRID_HEIGHT = 10
# NUM_FEATURES = 7

# NUM_ANIMALS = 75
# NUM_ADVERSARIES = 15
# NUM_DEFENDERS = 5

# num_cells = GRID_LEN*GRID_HEIGHT
# animal_feature_index = NUM_FEATURES - 1 # Animal Presence is last feature
# grid = np.random.rand(GRID_HEIGHT, GRID_LEN, NUM_FEATURES)
# grid = grid.reshape(GRID_LEN*GRID_HEIGHT, NUM_FEATURES)
# grid_input_vector = grid.reshape(1, GRID_LEN*GRID_HEIGHT*NUM_FEATURES)

# defenders_mu_sigma = actor.model.predict(grid_input_vector)
# print("Defender mu/sigmas: size ", defenders_mu_sigma[0])
