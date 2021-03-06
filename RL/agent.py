from .utils import ReplayBuffer, OrnsteinUhlenbeckNoise

import torch
import torch.nn.functional as F
import numpy as np

from copy import deepcopy


class DDPGAgent:
    def __init__(self, env, pi, q, gamma, tau, batch_size, lr_pi, lr_q, warmup_step, mode='train', render=False):
        # Initialize environment
        self.env = env

        # Initialize networks and optimizer
        self.pi = pi
        self.q = q
        self.pi_target = deepcopy(pi)
        self.q_target = deepcopy(q)

        # Attach networks to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pi.to(self.device)
        self.q.to(self.device)
        self.pi_target.to(self.device)
        self.q_target.to(self.device)

        # Initialize other hyperparameters
        self.gamma = gamma  # The discounted rate
        self.tau = tau  # hyperparameter for soft updating parameters of target networks
        self.batch_size = batch_size
        self.warmup_step = warmup_step
        self.render = render
        self.buffer = ReplayBuffer()
        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.env.action_space.shape[0]))

        # If training mode, set networks to training mode and define optimizers.
        self.mode = mode
        if self.mode == "train":
            self.pi.train()
            self.q.train()
            self.pi_target.train()
            self.q_target.train()
            self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=lr_pi)
            self.optimizer_q = torch.optim.Adam(self.q.parameters(), lr=lr_q)

        else:
            self.pi.eval()
            self.q.eval()
            self.pi_target.eval()
            self.q_target.eval()

    def train(self):
        s, a, r, s_prime, done = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        s_prime = torch.tensor(s_prime, device=self.device)
        done_mask = 1.0 - torch.tensor(done, device=self.device)  # done_mask takes 0 if done==1 otherwise 1

        # Compute losses
        q_target = r + self.gamma * self.q_target(s_prime, self.pi_target(s_prime).detach()) * done_mask
        q_loss = F.smooth_l1_loss(self.q(s, a), q_target)
        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        pi_loss = -self.q(s, self.pi(s)).mean()
        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        # Soft-update parameters of target networks
        for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def run_episode(self):
        s = self.env.reset()
        if self.render:
            self.env.render()
        done = False

        score = 0.0
        while not done:
            a = self.pi(torch.tensor(s, dtype=torch.float, device=self.device))
            a = a.item() # + self.noise()[0]
            s_prime, r, done, _ = self.env.step([a])
            if self.render:
                self.env.render()
            self.buffer.push(transition=(s, a, r / 100.0, s_prime, done))
            s = s_prime
            score += r

            if self.mode == 'train' and self.buffer.size() > self.warmup_step:
                self.train()

        return score
