import numpy as np

# Code based on:
# https://github.com/sfujim/TD3/blob/master/utils.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self, env):
		self.buffer = [] # No size limitation, similar to the paper
		self.zeroAction = np.zeros_like(env.action_space.sample(), dtype=np.float32)
		self.absorbingState = np.zeros((env.observation_space.shape[0]),dtype=np.float32)

	# data = (state, action, next_state)
	def add(self, data, done):
		if done:
			self.buffer.append((data[0], data[1], self.absorbingState))
		else:
			self.buffer.append(data)

	def addAbsorbing(self):
		self.buffer.append((self.absorbingState, self.zeroAction, self.absorbingState))

	def sample(self, batch_size=100):
		ind = np.random.randint(0, len(self.buffer), size=batch_size)
		s, a, ns = [], [], [], [], []

		for i in ind:
			S,A,nS = self.storage[i]
			s.append(np.array(S, copy=False))
			a.append(np.array(A, copy=False))
			ns.append(np.array(nS, copy=False))

		return np.array(s), np.array(a), np.array(ns)