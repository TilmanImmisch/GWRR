import numpy as np
import torch
from torch._C import device
#from torch._C import long

#own imports
from .gwrr_backend import GWRR

class GWR_replay(GWRR):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		#cache nonzero samples after training batches for faster sampling
		self.temporal_count_nonzero = 0

	
	def add(self, episode, states, actions, next_states, rewards, dones):
		states = torch.from_numpy(states)
		next_states = torch.from_numpy(next_states)
		_, self.temporal_count_nonzero = self.forward(episode, states, actions, next_states, rewards, dones)
	
	def sample(self, batch_size):
		p = np.squeeze(self.temporal_count[self.temporal_count_nonzero].toarray() / np.sum(self.temporal_count))

		##ALTERNATIVE WEIGHTING BY TD-ERROR
		#td_errors = np.abs(self.td_error[self.temporal_count_nonzero].toarray())
		#td_errors[td_errors == 0] = 0.2
		#td_count = td_errors * self.temporal_count[self.temporal_count_nonzero].toarray()
		#p = np.squeeze(td_count/ np.sum(td_count))
		#replace is True like in original implementation

		tuple_inds = np.random.choice(len(self.temporal_count_nonzero[0]),size=batch_size, replace=True, p=p)
		sns_inds = [(self.temporal_count_nonzero[0][tuple_inds]),(self.temporal_count_nonzero[1][tuple_inds])]

		r = np.arange(self.temporal_action.shape[1])
		col_mask = ([sns_inds[1]*self.action_dim][0][:,None] <= r) & ([sns_inds[1]*self.action_dim+self.action_dim-1][0][:,None] >= r)
		actions = self.temporal_action[sns_inds[0]][col_mask].todense().transpose()
		actions = np.array(np.split(actions, batch_size)).reshape(batch_size,-1)
		rewards = self.temporal_reward[sns_inds[0], sns_inds[1]].todense().transpose()

		return (
			self.V[sns_inds[0]].float().to(self.device), #full states
			torch.FloatTensor(actions).to(self.device), #actions
			self.V[sns_inds[1]].float().to(self.device), #full next_states
			torch.FloatTensor(rewards).to(self.device), #rewards
			torch.FloatTensor(1. - self.done[sns_inds[1]]).to(self.device), #done state
			sns_inds #inds to store td-error later
		)
	def store_td(self, inds, td_error):
		for i in range(inds[0].shape[0]):
			self.td_error[inds[0][i], inds[1][i]] =  td_error[i]