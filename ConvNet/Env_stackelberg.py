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
		self.logger = open("log_env_details.csv",'w')

		self.animal_feature_index = self.num_features - 1
		self.num_cells = grid_len * grid_len
		

		# lists of coordinates (row, col)
		self.animal_coords = [] # coordinates of cells with animals (no repeats!)
		self.defender_coords = []
		self.adversary_coords = []

		# grids of animals, adversaries, defenders
		self.animal_counts = np.zeros((self.grid_len, self.grid_len))
		self.adversary_one_hot = np.zeros((self.grid_len, self.grid_len))
		self.defender_one_hot = np.zeros((self.grid_len, self.grid_len))
		
	#simple helper to convert a sample 	to something between 0,10
	def convert_decimal(self,number):
		str_num = str(number)
	
		
		#REMOVE - SIGN IF ANY
		if	str_num[0]=='-':
			str_num = str_num[1:]
		
		deci_pos = str_num.index('.')
		
		
		#PROCESS THE LEFT SIDE OF THE DECIMAL
		str_deci_left = str_num[0:deci_pos]
		deci_left = int(str_deci_left)
		
		if deci_left > 0:
			return deci_left
		
		else:
			# #PROCESS THE RIGHT SIDE OF THE DECIMAL
			# #GET THE FIRST NON-ZERO DIGIT ON THE RIGHT
			deci_right = 0
			for i in range(deci_pos+1,len(str_num)):
				if int(str_num[i]) > 0:
					deci_right = int(str_num[i])
					break
				
			return deci_right

	def clear_animals(self):
		self.animal_coords.clear()
		self.animal_counts = np.zeros((self.grid_len, self.grid_len))

		# set animal feature to 0
		for row in self.grid:
			for cell in row:
				cell[self.animal_feature_index] = 0

	def clear_defenders(self):
		self.defender_coords.clear()
		self.defender_one_hot = np.zeros((self.grid_len, self.grid_len))

	def clear_adversaries(self):
		self.adversary_coords.clear()
		self.adversary_one_hot = np.zeros((self.grid_len, self.grid_len))


	# reset environment - clear animals, adversaries, defenders
	# keep original mu/sigmas the same
	def reset(self):
		self.clear_animals()
		self.clear_defenders()
		self.clear_adversaries()

	# # cell number (0-99) to (row, column)
	# def cell_to_coord(self, cell_number):
	# 	row = int(cell_number / self.grid_len)
	# 	col = int(cell_number % self.grid_len)
	# 	return (row, col)

	def place_animals(self):
		self.clear_animals()

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
			
			#-------------- TO BE UNCOMMENTED LATER --------------------------#
			# #first check that there are as many open spots for adversaries
			# if	len(self.animal_coords) < self.num_adversaries:
				# if (row,col) not in self.animal_coords:
					# # add animal
					# self.grid[row][col][self.animal_feature_index] += 1
					# self.animal_counts[row][col] += 1
					# self.animal_coords.append((row, col))
					# current_animal_count += 1
				# else:	
					# continue
			# else:
				# # add animal
				# self.grid[row][col][self.animal_feature_index] += 1
				# self.animal_counts[row][col] += 1
				# if(row,col) not in self.animal_coords:
					# self.animal_coords.append((row, col))
				# current_animal_count += 1
				
			#----CURRENT SIMULATION OF ANIMALS---------------------# 	
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
		#print(self.animal_counts)
		return self.animal_coords

	def create_episode(self):
		# reset environment features
		#self.reset()

		# randomly initialize grid features between 0-1
		self.grid = np.random.rand(self.grid_len, self.grid_len, self.num_features)
		
		# randomly init adversary mu/sigmas between 0-1
		self.adversary_mu_x = np.random.rand()
		self.adversary_sigma_x = np.random.rand()
		self.adversary_mu_y = np.random.rand()
		self.adversary_sigma_y = np.random.rand()
		
		self.logger.write(str(self.adversary_mu_x))
		self.logger.write(",")
		self.logger.write(str(self.adversary_sigma_x))
		self.logger.write(",")
		self.logger.write(str(self.adversary_mu_y))
		self.logger.write(",")
		self.logger.write(str(self.adversary_sigma_y))
		self.logger.write("\n")
		self.logger.flush()

	
		# randomly init animal mu/sigmas between 0-1
		self.animal_mu_x = np.random.rand()
		self.animal_sigma_x = np.random.rand()
		self.animal_mu_y = np.random.rand()
		self.animal_sigma_y = np.random.rand()
		
		# place animals in grid
		self.place_animals()

		return (self.grid, self.animal_coords)

	# generate adversaries
	# returns adversary coordinates and one-hot grid
	def place_adversaries(self):
		# clear existing adversaries
		self.clear_adversaries()
		adversary_count = 0
		#print("In the place adversaries function adversaries mu_sigmas\n")
		
			
		while adversary_count < self.num_adversaries:
			
		
			adversary_x = np.random.normal(self.adversary_mu_x, self.adversary_sigma_x)
			adversary_y = np.random.normal(self.adversary_mu_y, self.adversary_sigma_y)

			
			
			scaled_x = self.convert_decimal(adversary_x)
			scaled_y = self.convert_decimal(adversary_y)
			
			self.logger.write("actual_x:,")
			self.logger.write(str(adversary_x))
			self.logger.write(" ,")
			
			self.logger.write("actual_y:,")
			self.logger.write(str(adversary_y))
			self.logger.write("\n")
			self.logger.flush()
			# check if chosen location is out of bounds. If it is, try again
			if scaled_x < 0 or scaled_x >= self.grid_len:
				continue

			if scaled_y < 0 or scaled_y >= self.grid_len:
				continue

			row = scaled_x
			col = scaled_y
			
			self.logger.write("row:,")
			self.logger.write(str(row))
			self.logger.write(" ,")
			self.logger.write("col:,")
			self.logger.write(str(col))
			self.logger.write("\n")
			self.logger.flush()
			
			# if cell has animal and no adversary, add adversary. Otherwise, try again
			if self.animal_counts[row][col] > 0: #and (row, col) not in self.adversary_coords:
				if (row,col) not in self.adversary_coords:
					self.adversary_coords.append((row, col))
					self.adversary_one_hot[row][col] = 1
				adversary_count += 1
				
			else:
				continue
			
		self.logger.write("---------------------------------------------------------------\n")	
		self.logger.flush()
		return (self.adversary_coords, self.adversary_one_hot)

	# generate defenders
	# place_defenders: return list of defender locations (cell numbers)
	def place_defenders(self, mu_sigmas):
		# clear existing defenders
		self.clear_defenders()
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
			
			#print('defender row:',row,'defender col:',col)
			# if cell does not have defender, add defender. Otherwise, try again
			if (row, col) not in self.defender_coords:
				self.defender_coords.append((row, col))
				self.defender_one_hot[row][col] = 1
			defender_count += 1
		
		return (self.defender_coords, self.defender_one_hot)

	# -------------------------------------------------------------------

	# OLD REWARD FUNCTIONS

	# # input: one-hot adversary grid, int array of defender locations
	# # output: net defender reward
	# def defender_reward(self):
	# 	reward = 0
	# 	kernel = np.ones((3, 3))
	# 	nearby_adversaries = convolve2d(self.adversary_one_hot, kernel, \
	# 		mode='same', boundary='fill')
	# 	for defender_coord in self.defender_coords:
	# 		row = defender_coord[0]
	# 		col = defender_coord[1]
	# 		reward += nearby_adversaries[row][col]
	# 	return reward

	# # input: one-hot defender grid, int array of adversary locations
	# # output: net adversary reward
	# def adversary_reward(self):
	# 	reward = 0
	# 	kernel = np.ones((3, 3))
	# 	nearby_defenders = convolve2d(self.defender_one_hot, kernel, \
	# 		mode='same', boundary='fill')
	# 	for adversary_coord in self.adversary_coords:
	# 		row = adversary_coord[0]
	# 		col = adversary_coord[1]
	# 		num_nearby_defenders = nearby_defenders[row][col]
	# 		if num_nearby_defenders == 0:
	# 			reward += 1
	# 		#else:
	# 			#reward -= num_nearby_defenders
	# 	return reward

	# ------------------------------------------------------------------


	# R_a: reward based on adversaries
	# For each defender: num. of adversaries nearby - num. adversaries not nearby
		# scaled by num. animals in adversary cell
	def R_a(self):
		total_reward = 0

		# number of animals in cells with adversaries only
		animals_in_adversary_cells = np.multiply(self.adversary_one_hot, self.animal_counts)
		animals_attacked = np.sum(animals_in_adversary_cells)

		for defender in self.defender_coords:
			row = defender[0]
			col = defender[1]
			animals_defended = animals_in_adversary_cells[row][col]
			animals_not_defended = self.num_animals - animals_defended
			total_reward += animals_defended - animals_attacked

		return (total_reward / self.num_adversaries)

	# R_c: reward based on target coverage
	# For each defender:
		# +1 if a cell w/ animal(s) is defended
		# -1 if not defended
	# def R_c(self):
	# 	total_reward = 0
	# 	kernel = np.ones((3, 3))

	# 	animal_one_hot = animal_counts.clip(max=1)
	# 	num_animal_cells = np.sum(animal_one_hot)
	# 	nearby_animal_cells = convolve2d(self.animal_one_hot, kernel, \
	# 		mode='same', boundary='fill')
	# 	for defender in defender_coords:
	# 		reward = 0
	# 		row = defender[0]
	# 		col = defender[1]
			
	# 		covered_cells = nearby_animal_cells[row][col]
	# 		reward += covered_cells
			
	# 		non_covered_cells = num_animal_cells - covered_cells
	# 		reward -= non_covered_cells

	# 		total_reward += reward

	# 	return (total_reward / self.num_animals)


	# OLD METHOD
	# def R_c_old(self):
	# 	reward = 0
	# 	kernel = np.ones((3, 3))
	# 	nearby_defenders = convolve2d(self.defender_one_hot, kernel, \
	# 		mode='same', boundary='fill')
	# 	for animal_coord in self.animal_coords:
	# 		row = animal_coord[0]
	# 		col = animal_coord[1]
	# 		num_nearby_defenders = nearby_defenders[row][col]
			
	# 		if num_nearby_defenders == 0: # not defended
	# 			reward -= 1
	# 		elif num_nearby_defenders > 0: # defended
	# 			reward += 1

	# 	return reward

	# def calc_reward_with_coverage(self):
	# 	R_adversary = self.R_a()
	# 	R_coverage = self.R_c()
	# 	return ((R_adversary + R_coverage) / self.num_defenders)
	
	def calc_reward_no_coverage(self):
		R_adversary = self.R_a()
		return ((R_adversary) / self.num_defenders)

	def grid_vector(self):
		return self.grid.reshape(1, self.grid_len*self.grid_len*self.num_features)
		
	def count_animal_cells(self):
		count_cells = 0
		for row in self.grid:
			for cell in row:
				if cell[self.animal_feature_index] > 0:
					count_cells+=1
		return count_cells
		
		
	

			

