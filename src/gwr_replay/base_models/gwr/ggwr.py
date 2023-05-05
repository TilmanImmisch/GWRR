import logging

import torch

from .gwr import GWR
logger = logging.getLogger('GGWR-Log')


class GGWR(GWR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_context = kwargs.get('n_context')

        self.beta = torch.nn.Parameter(torch.tensor(kwargs.get('beta')))

        # Context
        self.C = torch.zeros((self.size, self.n_context, kwargs.get('dim')))
        # Global Context
        self.global_C = torch.zeros(self.n_context, kwargs.get('dim'))

        """
         calculate weights for the recurrent activation of the Gamma-GWR.
         According to the GDM paper.
        """
        self.alpha = torch.zeros(self.n_context + 1)
        for i in range(len(self.alpha)):
            self.alpha[i] = torch.exp(torch.tensor(float(-i)))
        self.alpha[:] = self.alpha[:] / sum(self.alpha)

    def activate_bmu(self, sample):
        """
        Calculate pairwise distance of sample and network nodes, determine BMU and sBMU and calculate activity
        :param sample: Feature values of current observation
        :return b: Index of BMU in 2D tensor V of network nodes
        :return s: Index of sBMU in 2D tensor V of network nodes
        :return a: Activity of the BMU
        """
        dists = self.pdist(sample, self.V)
        c_dists = 0
        global_C = torch.cat([self.global_C.view(1, self.n_context, int(self.global_C.shape[-1]))] * self.size)
        for i in range(self.n_context):
            c_dists += self.alpha[i+1] * self.pdist(global_C[:, i], self.C[:, i])

        dists = self.alpha[0] * dists + c_dists
        (b_dist, _), (b, s) = torch.topk(dists, k=2, largest=False, sorted=True)
        a = torch.exp(-b_dist)
        return b, s, a

    def update_global_context(self, b):
        """
        Update global context for next time step by merging weight and context of BMU.
        :param b: Index of BMU in 2D tensor V of network nodes
        """
        for k in range(self.n_context):
            if k == 0:
                self.global_C[k] = self.beta * self.C[b][k] + (1 - self.beta) * self.V[b]
            else:
                self.global_C[k] = self.beta * self.C[b][k] + (1 - self.beta) * self.C[b][k - 1]

    def insert_node(self, sample, b):
        """
        Insert new node into network at highest index in node list V, edge list E, context C, and habituation list h.
        :param sample: Feature values of current observation
        :param b: Index of BMU in 2D tensor V of network nodes
        """
        self.h = torch.cat((self.h, torch.ones(1)))
        self.V = torch.cat((self.V, torch.unsqueeze((sample + self.V[b]) / 2, 0)), dim=0)
        self.E = torch.cat((self.E, torch.full((1, self.size), -1)), dim=0)
        self.E = torch.cat((self.E, torch.full((self.size + 1, 1), -1)), dim=1)
        context = (0.5 * (self.global_C + self.C[b])).view(1, self.n_context, -1)
        self.C = torch.cat((self.C, context), dim=0)
        self.size += 1

    def delete_node(self, idx):
        """
        Delete node without edge connections in node list V, edge list E, context C ,and habituation list h.
        :param idx: Index of node to be deleted
        """
        self.h = torch.cat((self.h[:idx], self.h[idx + 1:]), dim=0)
        self.V = torch.cat((self.V[:idx], self.V[idx + 1:]), dim=0)
        self.E = torch.cat((self.E[:idx], self.E[idx + 1:]), dim=0)
        self.E = torch.cat((self.E[:, :idx], self.E[:, idx + 1:]), dim=1)
        self.C = torch.cat((self.C[:idx], self.C[idx + 1:]), dim=0)
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
        self.C[n] += self.eps_b * self.h[n] * (self.global_C - self.C[n])
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

    @torch.no_grad()
    def forward(self, it, data):
        """
        Original unsupervised GammaGWR algorithm as in Parisi et al. (2017)
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
                # update Context for the next step
                self.update_global_context(b)
