### Environment setup class ####
import numpy as np
from scipy.signal import convolve2d
from random import randint
from random import sample

class Env_simulation():

	def __init__(self, grid_len=10, num_features=7, num_animals=75, \
		num_adversaries=15, num_defenders=5):
		self.grid_len = grid_len
		self.num_features = num_features
		self.num_animals = num_animals
		self.num_adversaries = num_adversaries
		self.num_defenders = num_defenders

		self.animal_feature_index = self.num_features - 1
		self.num_cells = grid_len * grid_len
		
		
		# randomly init adversary mu/sigmas between 0-1
		self.adversary_mu_x = np.random.rand()
		self.adversary_sigma_x = np.random.rand()
		self.adversary_mu_y = np.random.rand()
		self.adversary_sigma_y = np.random.rand()


		# randomly init animal mu/sigmas between 0-1
		self.animal_mu_x = np.random.rand()
		self.animal_sigma_x = np.random.rand()
		self.animal_mu_y = np.random.rand()
		self.animal_sigma_y = np.random.rand()

		# lists of coordinates (row, col)
		self.animal_coords = [] # coordinates of cells with animals (no repeats!)
		self.defender_coords = []
		self.adversary_coords = []

		# grids of animals, adversaries, defenders
		self.animal_counts = np.zeros((self.grid_len, self.grid_len))
		self.adversary_one_hot = np.zeros((self.grid_len, self.grid_len))
		self.defender_one_hot = np.zeros((self.grid_len, self.grid_len))

	# cell number (0-99) to (row, column)
	def cell_to_coord(self, cell_number):
		row = int(cell_number / self.grid_len)
		col = int(cell_number % self.grid_len)
		return (row, col)

	def place_animals(self):
		# clear existing animals
		self.animal_coords.clear()

		# set animal feature to 0
		for row in self.grid:
			for cell in row:
				cell[self.animal_feature_index] = 0

		current_animal_count = 0

		# place animals based on random x, y distribution
		while current_animal_count < self.num_animals:
			animal_x = np.random.normal(self.animal_mu_x, self.animal_sigma_x)
			animal_y = np.random.normal(self.animal_mu_y, self.animal_sigma_y)

			scaled_x = int(animal_x * self.grid_len)
			scaled_y = int(animal_y * self.grid_len)
			
			# check if chosen location is out of bounds. If it is, try again
			if scaled_x < 0 or scaled_x >= self.grid_len:
				continue

			if scaled_y < 0 or scaled_y >= self.grid_len:
				continue

			row = scaled_y
			col = scaled_x

			# add animal
			self.grid[row][col][self.animal_feature_index] += 1
			self.animal_counts[row][col] += 1

			if (row, col) not in self.animal_coords:
				self.animal_coords.append((row, col))

			current_animal_count += 1

		# normalize animal counts to get animal density
		for row in self.grid:
			for cell in row:
				cell[self.animal_feature_index] /= self.num_animals


		# returns list of animal coordinates (row, col)
		return self.animal_coords

	def create_episode(self):
		# randomly initialize grid features between 0-1
		self.grid = np.random.rand(self.grid_len, self.grid_len, self.num_features)

		# set animal feature to 0
		for row in self.grid:
			for cell in row:
				cell[self.animal_feature_index] = 0

		# clear existing animals
		self.animal_coords.clear()
		self.animal_counts = np.zeros((self.grid_len, self.grid_len))

		# place animals in grid
		self.place_animals()

		# clear existing adversaries
		self.adversary_coords.clear()
		self.adversary_one_hot = np.zeros((self.grid_len, self.grid_len))

		# clear existing defenders
		self.defender_coords.clear()
		self.defender_one_hot = np.zeros((self.grid_len, self.grid_len))

		return (self.grid, self.animal_coords)

	# generate adversaries
	# returns adversary coordinates and one-hot grid
	def place_adversaries(self):
		# clear existing adversaries
		self.adversary_coords.clear()
		self.adversary_one_hot = np.zeros((self.grid_len, self.grid_len))
		adversary_count = 0
		
		while adversary_count < self.num_adversaries:
			adversary_x = np.random.normal(self.adversary_mu_x, self.adversary_sigma_x)
			adversary_y = np.random.normal(self.adversary_mu_y, self.adversary_sigma_y)

			scaled_x = int(adversary_x * self.grid_len)
			scaled_y = int(adversary_y * self.grid_len)
			
			# check if chosen location is out of bounds. If it is, try again
			if scaled_x < 0 or scaled_x >= self.grid_len:
				continue

			if scaled_y < 0 or scaled_y >= self.grid_len:
				continue

			row = scaled_y
			col = scaled_x

			# if cell has animal and no adversary, add adversary. Otherwise, try again
			if self.grid[row][col][self.animal_feature_index] > 0 \
					and (row, col) not in self.adversary_coords:
				self.adversary_coords.append((row, col))
				self.adversary_one_hot[row][col] = 1
				adversary_count += 1
			else:
				continue
		
		return (self.adversary_coords, self.adversary_one_hot)

	# generate defenders
	# place_defenders: return list of defender locations (cell numbers)
	def place_defenders(self, mu_sigmas):
		# clear existing defenders
		self.defender_coords.clear()
		self.defender_one_hot = np.zeros((self.grid_len, self.grid_len))
		defender_count = 0

		while defender_count < self.num_defenders:
			# get mu/sigma from defender network output
			defender_mu_x = mu_sigmas[defender_count*4]
			defender_sigma_x = mu_sigmas[defender_count*4 + 1]
			defender_mu_y = mu_sigmas[defender_count*4 + 2]
			defender_sigma_y = mu_sigmas[defender_count*4 + 3]

			# sample and scale up
			defender_x = np.random.normal(defender_mu_x, defender_sigma_x)
			defender_y = np.random.normal(defender_mu_y, defender_sigma_y)

			scaled_x = int(defender_x * self.grid_len)
			scaled_y = int(defender_y * self.grid_len)

			#print("Defender count: ", defender_count, " X:", scaled_x, " Y: ", scaled_y)

			# check if location is out of bounds
			if scaled_x < 0 or scaled_x >= self.grid_len:
				continue
			if scaled_y < 0 or scaled_y >= self.grid_len:
				continue

			row = scaled_y
			col = scaled_x

			# if cell does not have defender, add defender. Otherwise, try again
			if (row, col) not in self.defender_coords:
				self.defender_coords.append((row, col))
				self.defender_one_hot[row][col] = 1
				defender_count += 1
			else:
				continue
		
		return (self.defender_coords, self.defender_one_hot)

	# input: one-hot adversary grid, int array of defender locations
	# output: net defender reward
	def defender_reward(self):
		reward = 0
		kernel = np.ones((3, 3))
		nearby_adversaries = convolve2d(self.adversary_one_hot, kernel, \
			mode='same', boundary='fill')
		for defender_coord in self.defender_coords:
			row = defender_coord[0]
			col = defender_coord[1]
			reward += nearby_adversaries[row][col]
		return reward

	# input: one-hot defender grid, int array of adversary locations
	# output: net adversary reward
	def adversary_reward(self):
		reward = 0
		kernel = np.ones((3, 3))
		nearby_defenders = convolve2d(self.defender_one_hot, kernel, \
			mode='same', boundary='fill')
		for adversary_coord in self.adversary_coords:
			row = adversary_coord[0]
			col = adversary_coord[1]
			num_nearby_defenders = nearby_defenders[row][col]
			if num_nearby_defenders == 0:
				reward += 1
			#else:
				#reward -= num_nearby_defenders
		return reward

	# Coverage Reward (R_c)
	# +1 if an animal is defended (there is a nearby defender)
	# -1 if not defended (no nearby defender)
	def coverage_reward(self):
		reward = 0
		kernel = np.ones((3, 3))
		nearby_defenders = convolve2d(self.defender_one_hot, kernel, \
			mode='same', boundary='fill')
		for animal_coord in self.animal_coords:
			row = animal_coord[0]
			col = animal_coord[1]
			num_nearby_defenders = nearby_defenders[row][col]
			
			if num_nearby_defenders == 0: # not defended
				reward -= 1
			elif num_nearby_defenders > 0: # defended
				reward += 1

		return reward

	def calc_reward_no_coverage(self):
		d_reward = self.defender_reward()
		a_reward = self.adversary_reward()
		# c_reward = self.coverage_reward()

		total_reward = (d_reward - a_reward)/self.num_adversaries
		return total_reward

	def calc_reward_coverage(self):
		r_a = self.calc_reward_no_coverage()
		r_c = self.coverage_reward()/self.num_animals
		total_reward = r_a + r_c
		return total_reward

	def calc_reward(self):
		d_reward = self.defender_reward()
		a_reward = self.adversary_reward()

		total_reward = (d_reward - a_reward) / (self.num_defenders + self.num_adversaries)
		# total_reward = (d_reward / self.num_defenders) - (a_reward / self.num_adversaries)
		return total_reward

	def grid_vector(self):
		return self.grid.reshape(1, self.grid_len*self.grid_len*self.num_features)
		
	
	def grid_vector_cellwise(self):
		return self.grid.reshape(self.grid_len*self.grid_len, self.num_features) 
		
	def print_adversary_params(self):
		print(self.adversary_mu_x)
		print(self.adversary_sigma_x)
		print(self.adversary_mu_y)
		print(self.adversary_sigma_x)
		print(self.grid.shape)

	

	# ----------------- 3.10.17 -----------------

	# New reward (R_a):
	# For each defender: Iterate over all adversaries
		# Num. adversaries close to that defender - Num. adversaries not close to it
		# cell's reward (or penalty) = number of animals in that cell w/ adversary
		# Sum divided by number of adversaries

	# R_c:
		# For each defender: Iterate over all targets
		# If covering target, +1
		# If not covering target, -1
		# Sum divided by number of targets

	# PLots: matplotlib
		# Avg reward per timestep within one episode
			# 5 episodes --> 5 graphs
		# Make this a new class?

	# Copy folder --> "Simple Gradient Update Multiple Defenders"
	# Try same thing with 5 defenders
	# Reward: (R_a1 + R_c1 + R_a1 + R_c2 + ... R_a5 + R_c5) / num_defenders
	# R_a is divided by num. adversaries
	# R_c is divided by num. targets

	# DOCUMENTATION: separate file describing what functions do what
		# input, output, how it works, etc.

	# ACTION ITEMS:
		# Change reward (mainly R_a)
		# Add multiple defender case (separate folder)
		# Generate plots


