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


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        # 公式1
        resetgate = torch.sigmoid(i_r + h_r)
        # 公式2
        inputgate = torch.sigmoid(i_i + h_i)
        # 公式3
        newgate = torch.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        # self.gru = GRUCell(input_dim, hidden_dim)
        self.GRU = nn.GRU(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):

        if torch.cuda.is_available():
            h0 = Tensor(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Tensor(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim))

        outs = []
        hn = h0

        # for seq in range(x.size(1)):
        # hn= self.GRU(x[:, seq, :], hn)
        out, _ = self.GRU(x, hn)
        # outs.append(hn)

        # out = outs[-1]
        # out = self.fc(out)

        # out = torch.stack(outs, dim=1)
        out = self.fc(out[:, -3:, :])
        # out = self.fc(out)
        # out = torch.flip(torch.squeeze(out))
        #
        # for seq in range(x.size(1)):
        #     hn = self.gru(x[:, seq, :], hn)
        #     outs.append(hn)
        out = torch.squeeze(out)
        # out = torch.squeeze(out, dim=-1)

        return out
