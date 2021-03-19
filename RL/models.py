import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_(size):
    """
    Take a look "Experiment Details" in the DDPG paper
    code from: https://blog.paperspace.com/physics-control-tasks-with-deep-reinforcement-learning/
    """
    fan_in = size[0]
    weight = 1./math.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)


class Pi(nn.Module):
    def __init__(self, dim_state, dim_hidden1, dim_hidden2, dim_action):
        super(Pi, self).__init__()
        self.fc1 = nn.Linear(dim_state, dim_hidden1)
        self.fc1.weight.data = fanin_(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc2.weight.data = fanin_(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(dim_hidden2, dim_action)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)  # Take a look "Experiment Details" in the DDPG paper

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.tanh(x)

        return output


class Q(nn.Module):
    def __init__(self, dim_state, dim_hidden1, dim_hidden2, dim_action):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(dim_state, dim_hidden1)
        self.fc1.weight.data = fanin_(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(dim_hidden1 + dim_action, dim_hidden2)
        self.fc2.weight.data = fanin_(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(dim_hidden2, dim_action)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)  # Take a look "Experiment Details" in the DDPG paper

    def forward(self, s, a):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(torch.cat([x, a], dim=1))
        x = F.relu(x)
        output = self.fc3(x)

        return output
