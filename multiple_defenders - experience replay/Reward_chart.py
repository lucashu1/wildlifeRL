import matplotlib.pyplot as plt

class Reward_chart:

	def __init__(self):
		self.timesteps = []
		self.reward_vals = []
		self.current_episode = 1

		plt.ion()
		plt.show()

	def add_timestep(self, timestep, reward):
		self.timesteps.append(timestep)
		self.reward_vals.append(reward)

	def show_episode(self):
		# initialize new chart figure
		plt.figure(self.current_episode)
		plt.title("Episode: " + str(self.current_episode))
		plt.xlabel("Timestep")
		plt.ylabel("Average reward")

		# chart data
		plt.scatter(self.timesteps, self.reward_vals)
		plt.draw()
		plt.pause(0.001)

		# reset and increment episode so that another figure can be added
		self.current_episode += 1
		self.timesteps.clear()
		self.reward_vals.clear()

		plt.show()