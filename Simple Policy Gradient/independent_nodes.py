# testing a separate grid creation module
from Env_simulation import Env_simulation
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras import backend as K
import tensorflow as tf
import numpy as np
import csv

env =  Env_simulation(num_defenders=1)
env.create_episode()

def distribute_grid():
	dis_grid = env.grid_vector_cellwise()
	for row in dis_grid:
		

