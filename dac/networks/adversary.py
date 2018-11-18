import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dac.util.learning_rate import LearningRate
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
	def __init__(self, num_inputs, hidden_size=100):
		super(Discriminator, self).__init__()

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)
		self.linear3.weight.data.mul_(0.1)
		self.linear3.bias.data.mul_(0.0)
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.parameters())

	def forward(self, x):
		x = F.tanh(self.linear1(x))
		x = F.tanh(self.linear2(x))
		prob = F.sigmoid(self.linear3(x))
		return prob

	def adjust_adversary_learning_rate(self, lr):
		for param_group in self.optimizer:
			param_group['lr'] = lr

	def train(self, replay_buf, expert_buf, iterations, batch_size=100):
		lr = LearningRate.getInstance().getLR()
		self.adjust_adversary_learning_rate(lr)

		for it in range(iterations):
			# Sample replay buffer
			x, y, u = replay_buf.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)

			# Sample expert buffer
			expert_obs, expert_act = expert_buf.get_next_batch(batch_size)
			expert_obs = torch.FloatTensor(expert_obs).to(device)
			expert_act = torch.FloatTensor(expert_act).to(device)

			# Predict
			state_action = torch.cat([state, action], 1)
			expert_state_action = torch.cat([expert_obs, expert_act], 1)

			fake = self(state_action)
			real = self(expert_state_action)

			self.optimizer.zero_grad()
			loss = self.criterion(fake, torch.ones((state_action.size(0), 1)).to(device)) + self.criterion(real, torch.zeros((expert_state_action.size(0), 1)).to(device))
			loss.backward()
			self.optimizer.step()



