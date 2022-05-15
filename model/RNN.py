import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import pdb
import math

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.rnn = nn.RNN(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        # self.u = nn.Linear(input_dim, hidden_dim)
        # self.u = nn.Linear(input_dim, hidden_dim)
        # self.w = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(2 *hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Tensor(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Tensor(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim))

        # if torch.cuda.is_available():
        #     h0 = Tensor(torch.zeros(x.size(0), self.hidden_dim).cuda())
        # else:
        #     h0 = Tensor(torch.zeros(x.size(0), self.hidden_dim))

        # outs = []
        # hn = h0  # [B,N]
        #
        # for seq in range(x.size(1)):
        #     xt = self.u(x[:, seq, :])
        #     s0 = self.w(hn)  # 实为s_{t-1}
        #     hn = torch.relu(xt + s0)
        #     outs.append(hn)

        # out = outs[-1]
        # out_all = torch.stack(outs, dim=1)
        # out = self.v(out_all)
        # out = torch.mean(out_all,dim=-1)
        out, _ = self.rnn(x, h0)
        # out = self.u(x[:, -1, :])
        # out = self.u(x.view(-1, 24 * 7))
        out = self.v(out[:, -3:, :])
        # out = torch.relu(out)
        # out = self.w(out)
        # out = torch.relu(out)
        # out = self.v(out)
        # out = out.repeat(1, 3)
        out = torch.squeeze(out)

        return out
