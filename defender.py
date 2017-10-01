import numpy as np 
from scipy.signal import convolve2d
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge
from keras import backend as K
from random import randint
from random import sample

# Input features:
# 1. Habitat (0-1)
# 2. NPP (0-1)
# 3. Slope (0-1)
# 4. Road distance (0-1)
# 5. Town distance (0-1)
# 6. Water distance (0-1)
# 7. Animal presence (binary)

# NUM_GRIDS = 1
GRID_LEN = 10
GRID_HEIGHT = 10
NUM_FEATURES = 7

NUM_ANIMALS = 75
NUM_ADVERSARIES = 15
NUM_DEFENDERS = 5

num_cells = GRID_LEN*GRID_HEIGHT
animal_feature_index = NUM_FEATURES - 1 # Animal Presence is last feature

# generate random grid
# Initial grid dimensions: GRID_HEIGHT X GRID_LEN X NUM_FEATURES
grid = np.random.rand(GRID_HEIGHT, GRID_LEN, NUM_FEATURES)

# Flatten grid (so 10 x 10 grid turns into 1 x 100)
grid = grid.reshape(GRID_LEN*GRID_HEIGHT, NUM_FEATURES)

# cell numbers of which cells have animals, adversaries, defenders
animal_cells = []
adversary_cells = []
defender_cells = []
animal_adversary_buffer = []

# make Animal Presence feature 0
for cell in grid:
	cell[animal_feature_index] = 0 # (NUM_FEATURES - 1) = index of Animal Presence feature


# ----------------------------------------------------------------------------------


# 1. PLACE ANIMALS

# random values between 0 and 1
adversary_mu = np.random.rand()
adversary_sigma = np.random.rand()

print("Adversary mu/sigma: ", adversary_mu, " ", adversary_sigma)

# generate animals
# place_animals: returns list of animal locations (cell numbers), sets animal feature in grid
def place_animals(count):
	cells = []
	# randomly pick some cell numbers to have animals
	animal_locations = sample(range(num_cells-1), count)
	
	# set "Animal Presence" feature selected cells to be 1
	for cell_index in animal_locations:
		grid[cell_index][animal_feature_index] = 1
		cells.append(cell_index)

	return cells


animal_cells = place_animals(NUM_ANIMALS)
print("Animal cells: ", animal_cells)

# randomly pick some cells with animals to have adversaries as well
# adversary_locations = sample(animal_locations, NUM_ADVERSARIES)
# TODO: add adversaries as a feature in the grid? (keep it hidden to the defender_model)




# ----------------------------------------------------------------------------------

# 2. PLACE ADVERSARIES AND DEFENDERS

# flatten grid into a vector to feed into model
grid_input_vector = grid.reshape(1, GRID_LEN*GRID_HEIGHT*NUM_FEATURES)

# Create defender model with 1 hidden layer
# Input: flattened 1D vector of grid with features
# Output: mu and sigma between 0-1 for each defender
defender_model = Sequential()
defender_model.add(Dense(500, input_dim=num_cells*NUM_FEATURES,init='uniform', activation='sigmoid'))
defender_model.add(Dense(NUM_DEFENDERS*2, init='uniform', activation='sigmoid'))

# Save defender mu/sigmas
defenders_mu_sigma = defender_model.predict(grid_input_vector)[0]
print("Defender mu/sigmas: ", defenders_mu_sigma)


# generate adversaries
# place_adversaries: returns list of animal locations (cell numbers)
def place_adversaries(mu, sigma, count):
	cells = []
	one_hot = np.zeros(num_cells)
	current_adversaries = 0
	
	while current_adversaries < count:
		adversary_loc = np.random.normal(mu, sigma)
		cell_index = int(adversary_loc * num_cells) # value between 0 and num_cells - 1
		
		# check if cell_index is in bounds. If not, try again
		if cell_index < 0 or cell_index >= num_cells:
			continue

		# if cell has animal and no adversary, add adversary. Otherwise, try again
		if grid[cell_index][animal_feature_index] == 1 and cell_index not in cells:
			cells.append(cell_index)
			one_hot[cell_index] = 1
			current_adversaries += 1
		else:
			continue
	
	return [cells, one_hot]

[adversary_cells, adversary_one_hot] = place_adversaries(adversary_mu, adversary_sigma, NUM_ADVERSARIES)
print("Adversary cells: ", adversary_cells)

animal_adversary_buffer = [animal_cells, adversary_cells]
print("Non-defender buffer: ", animal_adversary_buffer)


# generate defenders
# place_defenders: return list of defender locations (cell numbers)
def place_defenders(mu_sigmas, count):
	defender_num = 0
	one_hot = np.zeros(num_cells)
	cells = []

	while defender_num < count:
		# get mu/sigma from defender network output
		defender_mu = mu_sigmas[defender_num*2]
		defender_sigma = mu_sigmas[defender_num*2 + 1]

		# sample and scale up
		defender_loc = np.random.normal(defender_mu, defender_sigma)
		cell_index = int(defender_loc * num_cells)

		# check if index is out of bounds
		if cell_index < 0 or cell_index >= num_cells:
			continue

		# if cell does not have defender, add defender. Otherwise, try again
		if cell_index not in cells:
			cells.append(cell_index)
			one_hot[cell_index] = 1
			defender_num += 1
		else:
			continue
	
	return [cells, one_hot]

[defender_cells, defender_one_hot] = place_defenders(defenders_mu_sigma, NUM_DEFENDERS)
print("Defender cells: ", defender_cells)

# 2D grids with 1s where adversaries/defenders are
adversary_grid = adversary_one_hot.reshape(GRID_HEIGHT, GRID_LEN)
defender_grid = defender_one_hot.reshape(GRID_HEIGHT, GRID_LEN)

print("Adversary grid:", "\n", adversary_grid)
print("Defender grid:", "\n", defender_grid)


# ----------------------------------------------------------------------------------


# 3. REWARD CALCULATIONS
# http://stackoverflow.com/questions/26363579/how-to-find-neighbors-of-a-2d-list-in-python


# input: one-hot adversary grid, int array of defender locations
# output: net defender reward
def defender_reward(adversary_grid, defender_cells):
	total_reward = 0
	kernel = np.ones((3, 3))
	nearby_adversaries = convolve2d(adversary_grid, kernel, mode='same', boundary='fill')
	for defender_location in defender_cells:
		defender_row = int(defender_location / GRID_LEN)
		defender_col = int(defender_location % GRID_LEN)
		total_reward += nearby_adversaries[defender_row][defender_col]
	return total_reward

# input: one-hot defender grid, int array of adversary locations
# output: net adversary reward
def adversary_reward(defender_grid, adversary_cells):
	total_reward = 0
	kernel = np.ones((3, 3))
	nearby_defenders = convolve2d(defender_grid, kernel, mode='same', boundary='fill')
	for adversary_location in adversary_cells:
		adversary_row = int(adversary_location / GRID_LEN)
		adversary_col = int(adversary_location % GRID_LEN)
		num_nearby_defenders = nearby_defenders[adversary_row][adversary_col]
		if num_nearby_defenders == 0:
			total_reward += 1
		else:
			total_reward -= num_nearby_defenders
	return total_reward

# if want to change function implementations to using all one-hot grids:
# http://stackoverflow.com/questions/27175400/how-to-find-the-index-of-a-value-in-2d-array-in-python

d_reward = defender_reward(adversary_grid, defender_cells)
a_reward = adversary_reward(defender_grid, adversary_cells)
total_reward = (d_reward - a_reward) / (NUM_DEFENDERS + NUM_ADVERSARIES)

print("Defender reward: ", d_reward)
print("Adversary reward: ", a_reward)
print("Total reward: ", total_reward)


# ----------------------------------------------------------------------------------

# 5. CRITIC NETWORK
# for more critic network model: https://keras.io/getting-started/functional-api-guide/

# define network layers
grid_input = Input(shape=(num_cells*NUM_FEATURES,), name='grid_input')
actor_input = Input(shape=(NUM_DEFENDERS*2,), name='actor_input')

grid_h1 = Dense(500, activation='sigmoid')(grid_input)
actor_h1 = Dense(20, activation='sigmoid')(actor_input)
grid_h2 = Dense(500, activation='sigmoid')(grid_h1)
actor_h2 = Dense(20, activation='sigmoid')(actor_h1)

merged_layer = merge([grid_h2, actor_h2], mode='concat')
critic_output = Dense(10, activation='linear', name='critic_output')(merged_layer)

# initialize model
critic_model = Model(input=[grid_input, actor_input], output=critic_output)

# test out critic model
defender_input_vector = defenders_mu_sigma.reshape(1, NUM_DEFENDERS*2)
critic_mu_sigma = critic_model.predict([grid_input_vector, defender_input_vector])[0]
print("Critic network output: ", critic_mu_sigma)



# ----------------------------------------------------------------------------------


# needed this to prevent some Tensorflow bug from popping up randomly
K.clear_session()