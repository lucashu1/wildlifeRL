# wildlifeRL

An attempt at training RL agents to find optimal anti-poaching patrol
routes in a simulated wildlife security game. Created during
the 2016-2017 school year for [USC Teamcore AI Lab](http://teamcore.usc.edu/people/Gsg/index.html).

### Overall Objective:
Given a park grid with a certain spatial distribution of animals to protect,
train the (anti-poaching) patroller/defender to take in the game state (i.e. park grid)
as input, and produce some action (i.e. selected patrol locaitons) as output.
If the defender picks locations that are close to poachers' locations, 
then the defender gets a reward.

### Methods Tested:
* **ConvNet:** Model the defender network using a ConvNet from 2D park grid to action vector
* **Vanilla Policy Gradient:** Update the defender network using the game reward as a gradient signal
* **DDPG** (Deep Deterministic Policy Gradient): Model the defender using two complementary neural networks:
one actor network (to map from game state to action), and one critic network (to judge the goodness of the action)
* **Multi-Agent RL**: Train reinforcement learning models for both the defenders (i.e. anti-poaching patrollers) and attackers (i.e. poachers), then see what happens

### Built With:
* Python 3.5 (Anaconda build)
* Numpy/Scipy
* Tensorflow
* Keras

### Relevant Papers/Links:
* [Continuous control with deep reinforcement learning - Lillicrap, et al.](https://arxiv.org/abs/1509.02971)
* [Using Keras and Deep Deterministic Policy Gradient to play TORCS - Ben Lau](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)
* [Pong from pixels - Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)