import torch
import torch.nn as nn
import torch.nn.functional as F


class Pi(nn.Module):
    def __init__(self, dim_state, dim_hidden1, dim_hidden2, dim_action):
        super(Pi, self).__init__()
        self.fc1 = nn.Linear(dim_state, dim_hidden1)
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, dim_action)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = 2 * torch.tanh(x)  # The range of action space is from -2 to 2

        return output


class Q(nn.Module):
    def __init__(self, dim_state, dim_hidden1, dim_hidden2, dim_action):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(dim_state, dim_hidden1)
        self.fc2 = nn.Linear(dim_hidden1 + dim_action, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, dim_action)

    def forward(self, s, a):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(torch.cat([x, a], dim=1))
        x = F.relu(x)
        output = self.fc3(x)

        return output
