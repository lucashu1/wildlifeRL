import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
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
# (8. Adversary presence - HIDDEN TO DEFENDER MODEL NETWORK)

NUM_GRIDS = 1
GRID_LEN = 10
GRID_HEIGHT = 10
NUM_FEATURES = 7

NUM_ANIMALS = 75
NUM_ADVERSARIES = 15
NUM_DEFENDERS = 5

num_cells = GRID_LEN*GRID_HEIGHT
animal_feature_index = NUM_FEATURES - 1 # Animal Presence is last feature

# generate array of training grids
# Each grid's dimensions are: (GRID_HEIGHT X GRID_LEN X NUM_FEATURES)
grids = np.random.rand(NUM_GRIDS, GRID_HEIGHT, GRID_LEN, NUM_FEATURES)

# Flatten each grid (so a 10 x 10 grid turns into 1 x 100)
grids = grids.reshape(NUM_GRIDS, GRID_LEN*GRID_HEIGHT, NUM_FEATURES)

# make Animal Presence feature 0
for grid in grids:
	for cell in grid:
		cell[animal_feature_index] = 0 # (NUM_FEATURES - 1) = index of Animal Presence feature

# one-hot array showing which cells have adversaries
adversaries = np.zeros((NUM_GRIDS, GRID_LEN*GRID_HEIGHT))

for grid in grids:
	# randomly pick some cell numbers to have animals
	animal_locations = sample(range(num_cells-1), NUM_ANIMALS)

	# print("Number of animals:", len(animal_locations))

	# set "Animal Presence" feature selected cells to be 1
	for cell_num in animal_locations:
		# print(cell_num)
		grid[cell_num][animal_feature_index] = 1

	# randomly pick some cells with animals to have adversaries as well
	adversary_locations = sample(animal_locations, NUM_ADVERSARIES)


# flatten each grid into a 1D vector
grids = grids.reshape(NUM_GRIDS, GRID_LEN*GRID_HEIGHT*NUM_FEATURES)

# Create defender model with 1 hidden layer
# Output: likelihood score of placing a defender in each cell
defender_model = Sequential()
defender_model.add(Dense(500, input_dim=num_cells*NUM_FEATURES,init='uniform', activation='sigmoid'))
defender_model.add(Dense(num_cells, init='uniform', activation='softmax'))

# Scores will add up to NUM_DEFENDERS
scores = defender_model.predict(grids) * NUM_DEFENDERS

# Some confirmation tests
print(scores)
print(np.sum(scores))