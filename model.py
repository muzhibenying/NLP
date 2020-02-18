import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    """
    The network contains a simple LSTM layer, a batch normalization layer,
    a full connection layer and a softmax layer.
    """

    def __init__(self, embed_size, hidden_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, batch_first = True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        bn_input, (hn, cn) = self.lstm(x)
        fc_input = bn_input[:, -1, :]
        output = self.fc3(self.fc2(self.fc1(fc_input)))
        output = F.softmax(output, dim = 1)
        return output
