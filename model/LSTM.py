import torch
import torch.nn as nn
from torch import Tensor
import math

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.c2c = Tensor(hidden_size * 3)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        c2c = self.c2c.unsqueeze(0)
        ci, cf, co = c2c.chunk(3, 1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate + ci * cx)
        forgetgate = torch.sigmoid(forgetgate + cf * cx)
        cellgate = forgetgate * cx + ingate * torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate + co * cellgate)

        hm = outgate * torch.tanh(cellgate)
        return (hm, cellgate)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        # 
        # self.lstm1 = LSTMCell(input_dim, hidden_dim)
        # self.lstm2 = LSTMCell(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):

        if torch.cuda.is_available():
            h0 = Tensor(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Tensor(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim))

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = Tensor(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Tensor(torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim))

        out, _ = self.lstm(x, (h0, c0))
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

        return out
