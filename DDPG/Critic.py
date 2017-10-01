### Critic class ####

import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

from Actor import Actor

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class Critic(object):
    def __init__(self, sess, state_size, action_size, LEARNING_RATE, TAU):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        # Create main critic network
        self.model, self.action, self.state = self.create_critic(state_size, action_size)  

        # target critic network
        self.target_model, self.target_action, self.target_state = self.create_critic(state_size, action_size)

        # gradient of output with respect to action(s)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

        # initialize critic_target to match critic
        self.copy_weights_to_target()


    # returns evaluated gradient w.r.t. actions (action gradient)
    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def update_critic_target(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def copy_weights_to_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def create_critic(self, state_size,action_dim):
        # print("Now we build the model")
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu', init='glorot_uniform')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear', init='glorot_uniform')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear', init='glorot_uniform')(w1)
        h2 = merge([h1,a1],mode='sum')    
        h3 = Dense(HIDDEN2_UNITS, activation='relu', init='glorot_uniform')(h2)
        V = Dense(action_dim,activation='linear', init='glorot_uniform')(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 


# sess = tf.Session()	
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

# actor = Actor(sess,100*7, NUM_DEFENDERS*4,0.0001,0.001)

# defenders_mu_sigma = actor.model.predict(grid_input_vector)[0]
# print("Defender mu/sigmas: ", defenders_mu_sigma)

# critic = Critic(sess, 100*7, NUM_DEFENDERS*4, .0001, .0001)

# critic_output = critic.target_model.predict([grid_input_vector, defenders_mu_sigma.reshape(1, NUM_DEFENDERS*4)])[0]
# print("Critic output: ", critic_output)









# critic: how good or bad the actor network's prediction was

# Feed in predictions + reward from buffer into critic network
# Sample from buffer, feed in states + predictions --> 
	# get different critic/estimated reward (not same as actual reward)
	# compare difference between rewards (from actor/critic)
	# difference = our "loss" - aim to minimize this

# sum over (gradient_actor * gradient_critic) for each sample
	# like we're doing right now, but with extra multiplication
	# gradient_critic = "action_gradient"

# critic network input: state (700-dim) and action (num_defenders * 4)
# output: 1 value (b/w 0-1)

# when doing update: use target network to do update
# target networks are doing the predictions
# target networks ~ mimic original networks
# weights are updated in the original network
# predictions (outputting mu_sigma for actor, or est. reward for critic) done via target networks

# target-train: updates the weights of the target network

# https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

# try feeding in random output for grid_vec + mu_sigmas