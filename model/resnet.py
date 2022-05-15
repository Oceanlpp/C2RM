from torchvision import models
import torch
import torch.nn as nn
from torch import Tenso

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)


class ResNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResNetModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

        resnet18 = models.resnet18()

        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        out = resnet18(x)

        out = self.fc(out[:, -3:, :])
        out = torch.squeeze(out)

        return out
