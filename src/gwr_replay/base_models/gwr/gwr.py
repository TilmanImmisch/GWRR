import logging

import torch
import torch.nn as nn

logger = logging.getLogger('GWR-Log')


class GWR(nn.Module):
    def __init__(self, **kwargs):
        super(GWR, self).__init__()
        self.size = 2
        self.pdist = nn.PairwiseDistance(p=2)

        # Thresholds
        self.a_t = torch.tensor(kwargs.get('a_t'))
        self.h_t = torch.tensor(kwargs.get('h_t'))
        self.max_age = torch.tensor(kwargs.get('max_age'))

        # Hyperparameters for optimization
        self.tau_b = torch.nn.Parameter(torch.tensor(kwargs.get('tau_b')))
        self.tau_n = torch.nn.Parameter(torch.tensor(kwargs.get('tau_n')))
        self.eps_b = torch.nn.Parameter(torch.tensor(kwargs.get('eps_b')))
        self.eps_n = torch.nn.Parameter(torch.tensor(kwargs.get('eps_n')))
        self.kappa = torch.nn.Parameter(torch.tensor(kwargs.get('kappa')))

        # Nodes and edges
        self.h = torch.ones(self.size)
        self.V = torch.rand(self.size, kwargs.get('dim'))
        self.E = torch.full((self.size, self.size), -1, dtype=torch.long)

    def activate_bmu(self, sample):
        """
        Calculate pairwise distance of sample and network nodes, determine BMU and sBMU and calculate activity
        :param sample: Feature values of current observation
        :return b: Index of BMU in 2D tensor V of network nodes
        :return s: Index of sBMU in 2D tensor V of network nodes
        :return a: Activity of the BMU
        """
        dists = self.pdist(sample, self.V)
        (b_dist, _), (b, s) = torch.topk(dists, k=2, largest=False, sorted=True)
        a = torch.exp(-b_dist)
        return b, s, a

    def insert_node(self, sample, b):
        """
        Insert new node into network at highest index in node list V, edge list E, and habituation list h.
        :param sample: Feature values of current observation
        :param b: Index of BMU in 2D tensor V of network nodes
        """
        self.h = torch.cat((self.h, torch.ones(1)))
        self.V = torch.cat((self.V, torch.unsqueeze((sample + self.V[b]) / 2, 0)), dim=0)
        self.E = torch.cat((self.E, torch.full((1, self.size), -1)), dim=0)
        self.E = torch.cat((self.E, torch.full((self.size + 1, 1), -1)), dim=1)
        self.size += 1

    def delete_node(self, idx):
        """
        Delete node without edge connections in node list V, edge list E, and habituation list h.
        :param idx: Index of node to be deleted
        """
        self.h = torch.cat((self.h[:idx], self.h[idx + 1:]), dim=0)
        self.V = torch.cat((self.V[:idx], self.V[idx + 1:]), dim=0)
        self.E = torch.cat((self.E[:idx], self.E[idx + 1:]), dim=0)
        self.E = torch.cat((self.E[:, :idx], self.E[:, idx + 1:]), dim=1)
        self.size -= 1

    def update_bmu(self, b, sample):
        """
        Adapt position of BMU (in case of no new insertion)
        :param b: Index of BMU in 2D tensor V of network nodes
        :param sample: Feature values of current observation
        """
        self.V[b] += self.eps_b * self.h[b] * (sample - self.V[b])
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
        self.h[n] += self.tau_n * self.kappa * (1 - self.h[n]) - self.tau_n

        delete = False
        if self.E[b, n] < self.max_age:
            self.E[b, n] += 1
            self.E[n, b] += 1
        else:
            self.E[b, n], self.E[n, b] = -1, -1
            if all(torch.eq(self.E[n], -1)):
                delete = True

        return delete

    def update_edges(self, b, s, r=None):
        """
        If new node inserted (r not None), create edge connections with BMU and sBMU and delete their common edge,
        otherwise reset/create edge connection between BMU and sBMU
        :param b: Index of BMU in 2D tensor V of network nodes
        :param s: Index of sBMU in 2D tensor V of network nodes
        :param r: Index of new node in 2D tensor V of network nodes
        """
        if r is None:
            self.E[b, s], self.E[s, b] = 0, 0
        else:
            self.E[b, s], self.E[s, b] = -1, -1
            self.E[b, r], self.E[r, b], self.E[s, r], self.E[r, s] = 0, 0, 0, 0

    @torch.no_grad()
    def forward(self, it, data):
        """
        Original unsupervised GWR algorithm as in Marsland et al. (2002)
        b: Index of BMU | s: Index of second BMU | a: BMU activity
        :param it: Number of batch/iteration
        :param data: List of mini-batch samples (contains just a single sample for continuous data stream)
        """
        for sample, _ in data:

            b, s, a = self.activate_bmu(sample)
            if self.training:
                if a < self.a_t and self.h[b] < self.h_t:
                    self.insert_node(sample, b)
                    self.update_edges(b, s, self.size - 1)
                    logger.info('Iteration {}. Inserted new node at position (first dimensions): {}. BMU index: {}. '
                                'Updated network size: {}.'.format(it, sample.numpy()[:4].round(3), b, self.size))

                else:
                    self.update_edges(b, s)
                    self.update_bmu(b, sample)

                    neighbors = [n for n in range(self.size) if self.E[b, n] > -1]
                    n_deleted = 0
                    for n in neighbors:
                        if self.update_neighbor(n - n_deleted, b, sample):
                            self.delete_node(n - n_deleted)
                            logger.info('Iteration {}. Deleted node at index: {}. Updated network size: {}.'.format(
                                it, n - n_deleted, self.size))
                            b -= 1 if n < b else b
                            n_deleted += 1
