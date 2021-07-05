"""
net 
"""

import torch 
from torch import nn
import torch.nn.functional as F


class AttentiveNet(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()

        # self.liner1 = nn.Linear(input_size, hidden_size)
        self.cov2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.cov1 = nn.Conv1d(input_size, hidden_size, kernel_size=1, padding=0)
        # self.cov3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)

        # self.w = nn.Parameter(torch.Tensor(hidden_size*2, 1))

        self.dense = nn.Linear(hidden_size, 1)
        

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        x1 = self.cov1(x)
        x = self.cov2(x1)

        x = x.permute(0,2,1).contiguous()

        out = torch.mean(x, dim=1)  # [batch, 64]

        out = F.relu(out)
        out = F.sigmoid(self.dense(out))# [batch, 1]
        return out.squeeze()



