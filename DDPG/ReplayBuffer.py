### buffer class ####
from collections import deque
import random
import numpy as np 

class ReplayBuffer():
	
	def __init__(self, buffer_size=5000, random_seed=123):
		# The right side of the deque contains the most recent experiences 
		self.buffer_size = buffer_size
		self.count = 0
		self.buffer = deque()
		random.seed(random_seed)

	def buffer_add(self, state, action, reward, new_state):
		experience = (state, action, reward, new_state)
		if self.count < self.buffer_size: 
			self.buffer.append(experience)
			self.count += 1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def size(self):
		return self.count

	def get_elements(self, n):
		batch = []

		if self.count < n:
			batch = random.sample(self.buffer, self.count)
		else:
			batch = random.sample(self.buffer, n)

		return batch

	def clear(self):
		self.deque.clear()
		self.count = 0