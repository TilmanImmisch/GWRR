import numpy as np
import torch
from torch._C import device
#from torch._C import long

#has batch processing capabilities, unlike utils.py
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		#size is used by the DDPG to have acces to all stored samples in the short term buffer, which are being overwritten sample by sample
		#ep_size is used by the GWR to learn just the new samples in batches
		#It is NOT possible to use mem_batch_size, as the number of stored samples rises by one additional sample each time the ep ends, 
		#which can push up the size of the ReplayBuffer somewhat (depending on num of eps)
		self.size = 0
		self.ep_size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, _, state, action, next_state, reward, done):
		if not(hasattr(reward, '__iter__')):
			batch_size = 1
		else: #if it's a list
			batch_size = len(reward)
		e_ptr = self.ptr + batch_size
		self.state[self.ptr:e_ptr] = np.copy(state)
		self.action[self.ptr:e_ptr] = action
		self.next_state[self.ptr:e_ptr] = np.copy(next_state)
		self.reward[self.ptr:e_ptr] = reward
		self.not_done[self.ptr:e_ptr] = 1. - done

		self.ptr = (self.ptr + batch_size) % self.max_size
		self.size = min(self.size + batch_size, self.max_size)
		self.ep_size = min(self.ep_size + batch_size, self.max_size)
	


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			ind
		)

	#Not using this now, as for training samples are needed
	def reset(self):
		self.size = 0
		self.ptr =0
		self.state[:] = 0
		self.action[:] = 0
		self.next_state[:] = 0
		self.reward[:] = 0
		self.not_done[:] =0


