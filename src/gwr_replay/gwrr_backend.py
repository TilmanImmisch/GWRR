import logging
import importlib
from turtle import update  

import torch
from scipy import sparse
import numpy as np

from .base_models.gwr.ggwr import GGWR
logger = logging.getLogger('EGGWR-Log')

def delete_row_lil(mat, i):
    if not isinstance(mat, sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

def delete_column_lil(mat, i):
    if not isinstance(mat, sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

class GWRR(GGWR):
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neighborhood = kwargs['neighborhood']
        self.td_modulate = kwargs['td_modulate']

        self.test_counter = 0 
        self.td_discount = 0.5

        super().__init__(**kwargs)
        self.action_dim = kwargs.get("action_dim")
        self.name = 'EGGWR'
        self.phi = kwargs["phi"]
        self.a_t_max = kwargs["a_t_max"]
        self.h_t_max = kwargs["h_t_max"]
        #self.iota = .2

        self.matrix_cat_size = 2000
        #storing done with nodes
        self.done = torch.full([self.size,1],-1)
        # Nodes and edges

        #self.E = torch.full((self.matrix_cat_size, self.matrix_cat_size), -1, dtype=torch.long)
        #self.h = torch.ones(self.size)
        #self.V = torch.tensor(start_state.reshape(1,-1))
        self.E = sparse.lil_matrix((self.matrix_cat_size, self.matrix_cat_size))

        #adding four temporal matrices to store average count, reward, action and done along temporal edges

        self.temporal_count = sparse.lil_matrix((self.matrix_cat_size,self.matrix_cat_size))
        self.temporal_reward = sparse.lil_matrix((self.matrix_cat_size,self.matrix_cat_size))
        self.td_error = sparse.lil_matrix((self.matrix_cat_size,self.matrix_cat_size))
        #mapping 3D matrix (size x size x action to 2D matrix)
        self.temporal_action = sparse.lil_matrix((self.matrix_cat_size, self.matrix_cat_size* self.action_dim))

        self.num_neighbors = 0

        self.prev_bmu = -1
        self.prev_act = np.zeros((self.action_dim))
        self.prev_rew = 0
        self.prev_done = 0

    #Activation with taking into account done
    # def activate_bmu(self, sample, done):
    #     """
    #     Calculate pairwise distance of sample and network nodes, determine BMU and sBMU and calculate activity
    #     Taking into account the euclidian distance, context distance and done state
    #     :param sample: Feature values of current observation
    #     :return b: Index of BMU in 2D tensor V of network nodes
    #     :return s: Index of sBMU in 2D tensor V of network nodes
    #     :return a: Activity of the BMU
    #     """
    #     dists = self.pdist(sample, self.V)
    #     c_dists = 0
    #     global_C = torch.cat([self.global_C.view(1, self.n_context, int(self.global_C.shape[-1]))] * self.size)
    #     for i in range(self.n_context):
    #         c_dists += self.alpha[i+1] * self.pdist(global_C[:, i], self.C[:, i])
        

    #     d_dists = (done, self.done)

    #     #new_dists = self.alpha[0] * dists + c_dists + self.iota * d_dists
    #     (b_dist, _), (b, s) = torch.topk(dists, k=2, largest=False, sorted=True)
    #     a = torch.exp(-b_dist)
    #     return b, s, a 



    def insert_node(self, sample, b):
        """
        Insert new node into network at highest index in node list V, edge list E, context C, temporal connection list
        and, habituation list h.
        :param sample: Feature values of current observation
        :param b: Index of BMU in 2D tensor V of network nodes
        """
        self.h = torch.cat((self.h, torch.ones(1)))
        ##WOAH THERE, SUBTLE CHANGE: inserting node at position (sample), not in-between (sample + self.V[b]) / 2) 
        self.V = torch.cat((self.V, torch.unsqueeze(sample,0)), dim=0)
        context = self.global_C.view(1, self.n_context, -1)
        #context = (0.5 * (self.global_C + self.C[b])).view(1, self.n_context, -1)
        self.C = torch.cat((self.C, context), dim=0)
        self.done = torch.cat((self.done, torch.full([1,1],-1)))
        
        #resizing lil_matrices for count and reward
        if self.size % self.matrix_cat_size ==0:
            temporal_size = np.array(self.temporal_count.shape)
            temporal_size[:] += self.matrix_cat_size
            self.E.resize(temporal_size)
            self.temporal_count.resize(temporal_size)
            self.temporal_reward.resize(temporal_size)
            self.td_error.resize(temporal_size)
            self.temporal_action.resize((temporal_size[0], temporal_size[1]*self.action_dim))

        self.size += 1

    def delete_node(self, idx):
        """
        Delete node without edge connections in node list V, edge list E, context C , temporal connection list,
        and habituation list h.
        :param idx: Index of node to be deleted
        """
        self.h = torch.cat((self.h[:idx], self.h[idx + 1:]), dim=0)
        self.V = torch.cat((self.V[:idx], self.V[idx + 1:]), dim=0)
        self.C = torch.cat((self.C[:idx], self.C[idx + 1:]), dim=0)
        self.done = torch.cat((self.done[:idx], self.done[idx+1:]), dim=0)

        delete_row_lil(self.E, idx)
        delete_column_lil(self.E, idx)
        #resizing so that temporal_cat_size still fits 
        self.E.resize(self.E.shape+1)

        delete_row_lil(self.temporal_count, idx)
        delete_column_lil(self.temporal_count, idx)
        self.temporal_count.resize(self.temporal_count.shape +1 )

        delete_row_lil(self.temporal_action, idx)
        delete_column_lil(self.temporal_action, np.arange(idx, idx+self.action_dim))
        self.temporal_action.resize((self.temporal_action.shape[0]+1, self.temporal_action.shape[1]+self.action_dim))

        delete_row_lil(self.temporal_reward, idx)
        delete_column_lil(self.temporal_reward, idx)
        self.temporal_reward.resize(self.temporal_reward +1)

        delete_row_lil(self.td_error, idx)
        delete_column_lil(self.td_error, idx)
        self.td_error.resize(self.td_error +1)

        self.size -= 1

    def update_bmu(self, b, sample):
        """
        Adapt position of BMU (in case of no new insertion)
        :param b: Index of BMU in 2D tensor V of network nodes
        :param sample: Feature values of current observation
        """
        self.V[b] += self.eps_b * self.h[b] * (sample - self.V[b])
        self.C[b] += self.eps_b * self.h[b] * (self.global_C - self.C[b])
        self.h[b] += self.tau_b * self.kappa * (1 - self.h[b]) - self.tau_b

    def update_neighbor(self, n, b, sample):
        """
        Adapt position of BMU neighbor, update its edge connection to the BMU, and check if it has to be deleted
        :param n: Index of BMU neighbor in 2D tensor V of network nodes
        :param b: Index of BMU in 2D tensor V of network nodes
        :param sample: Feature values of current observation
        :return delete: True if current node has to be deleted (no outgoing edges), False otherwise
        """
 
        self.V[n] += self.eps_n * self.h[n] * (sample - self.V[n])
        #why by eps_b ???
        self.C[n] += self.eps_n * self.h[n] * (self.global_C - self.C[n])
        self.h[n] += self.tau_n * self.kappa * (1 - self.h[n]) - self.tau_n

        delete = False
        if self.E[b, n] < self.max_age:
            self.E[b, n] += 1
            self.E[n, b] += 1
        else:
            self.E[b, n], self.E[n, b] = 0, 0
            if all(torch.eq(self.E[n], 0)):
                delete = True
        return delete

    #Updating the mean done-ness of a bmu
    def update_done(self, current_bmu, done):
        if self.done[current_bmu] == -1:
            self.done[current_bmu] = done
        else:
            self.done[current_bmu] =  self.phi * done + (1-self.phi) * self.done[current_bmu]


    def update_temporal(self, current_bmu, prev_bmu, action, reward):
        """
        Update the temporal connection list.
        :param current_bmu: Index of current BMU
        :param prev_bmu: Index of the previous BMU
        """

        #update connection counter
        self.temporal_count[prev_bmu, current_bmu] += 1

        #action_dim = 1, using lil_matrix
        act_idx = current_bmu*self.action_dim
        if int(self.temporal_count[prev_bmu, current_bmu]) == 1:
            self.temporal_action[prev_bmu, act_idx:act_idx+ self.action_dim] = action
            self.temporal_reward[prev_bmu, current_bmu] = reward
        else:
            #tensors can't be added to lil matrices ...
            action_delta =  self.phi * action + (1-self.phi) * self.temporal_action[prev_bmu, act_idx:act_idx+self.action_dim].toarray().flatten()
            #to making indexing work
            if self.action_dim == 1:
                action_delta = torch.tensor([action_delta])
            for i in range(self.action_dim):
                self.temporal_action[prev_bmu, act_idx+i] = action_delta[i]
            self.temporal_reward[prev_bmu, current_bmu] = self.phi * reward + (1-self.phi) * self.temporal_reward[prev_bmu, current_bmu]

    def sigmoid(self, value):
        return np.abs((2/(1+np.e**(-value)))-1)

    @torch.no_grad()
    def forward(self, it, states, actions, next_states,rewards, dones):
        """
        Original Episodic Gamma-GWR algorithm as in Parisi et al. (2018)
        b: Index of BMU | s: Index of second BMU | a: BMU activity
        :param it: Number of batch/iteration
        :param data: List of mini-batch samples (contains just a single sample for continuous data stream)
        """
        bmus = torch.zeros((1, self.V[0].shape[0]))
        prev_bmu = self.prev_bmu
        iter = 0
        self.temporal_reward.tolil()
        self.temporal_action.tolil()

        test_context = torch.zeros(self.global_C.shape)
        if not self.training:
            old_context = self.global_C
            self.global_C = test_context

        #adding env reset state on front, so that it's also learned, when new episode start
        if self.action_dim == 1:
            actions = np.append(self.prev_act,actions)  
        else:
            actions = np.vstack((self.prev_act,actions))  
        rewards= np.append(self.prev_rew,rewards)   
        dones = np.append(self.prev_done,dones)
        self.prev_act = actions[-1]
        self.prev_rew = rewards[-1]
        self.prev_done = dones[-1]

        #using next_states instead of states, so that action, reward, dones are assigned correctly
        for ptr, state in enumerate(states):
            b, s, a = self.activate_bmu(state)

            bmus = torch.cat((bmus, self.V[b].view(1, -1)), dim=0)
            #new line
            if self.training:
                #changed < to <= for h[b] <= h_t
                a_t = self.a_t
                h_t = self.h_t
                if self.td_modulate:
                    td_error = self.td_error[prev_bmu, b]
                    a_t = self.a_t + (self.a_t_max - self.a_t) * self.sigmoid(td_error)
                    h_t = self.h_t + (self.h_t_max - self.h_t) * self.sigmoid(td_error)
                if a < a_t and self.h[b] <= h_t:
                    self.insert_node(state, b)
                    self.update_edges(b, s, self.size - 1)
                    #link prev bmu to the newly created node
                    b = self.size -1
                    if self.td_modulate:
                        self.td_error[prev_bmu, b] = td_error * self.td_discount
                    #logger.info('Iteration {}. Inserted new node at position (first dimensions): {}.'
                    #            'BMU index: {}. Updated {} size: {}.'.format(it, 4, b, self.name,  self.size))
                else: 
                    self.update_edges(b, s) 
                    self.update_bmu(b, state)
                    if self.neighborhood:
                        neighbors = self.E[b].nonzero()[1]
                        if len(neighbors) > self.num_neighbors:
                            self.num_neighbors = len(neighbors)
                        n_deleted = 0
                        for n in neighbors:
                            n = int(n)
                            if self.update_neighbor(n - n_deleted, b, state):
                                logger.info('Iteration {}. Deleted node at index: {}. size: {}.'.format(it, n - n_deleted, self.size))
                                self.delete_node(n - n_deleted)
                                b -= 1 if n < b else b
                                n_deleted += 1        
                self.update_done(b, dones[ptr])
                 #if episode ended don't link end of last ep and start of new one
                if prev_bmu != -1:
                    self.update_temporal(b, prev_bmu, actions[ptr], rewards[ptr])
                prev_bmu = b
                if dones[ptr] == 1:
                    prev_bmu = -1 
                # update Context for the next step
                self.update_global_context(b)
            #TODO: Look at this
            else:
                test_context[0] = state
                for i in range(1, self.n_context):
                    test_context[i] = test_context[i-1]
                self.global_C = test_context
            iter += 1
        self.prev_bmu = prev_bmu
        if not self.training:
            self.global_C = old_context
        self.temporal_reward.tocsr()
        self.temporal_action.tocsr()
        return bmus[1,:], self.temporal_count.nonzero()
