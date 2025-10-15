import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, output_dim=1):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)

        if target.size(0) == 0:
            return 0

        target = target.view(-1, 1)
        # print(target.shape)
        output = output.view(-1, 1)
        pt = output * target + (1 - output) * (1 - target)
        # print(pt.shape)
        ce_loss = -torch.log(pt)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


if __name__ == "__main__":

    mlp = MLP(hidden_dim=128)
