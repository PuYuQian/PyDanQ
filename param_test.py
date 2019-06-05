import scipy.io
import torch
import torch.nn as nn
from torchsummary import summary
print('compling the network')


class DanQ(nn.Module):
    def __init__(self, ):
        super(DanQ, self).__init__()
        self.ConvBlock = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(p=0.2),
        )
        self.LSTMBlock = nn.LSTM(input_size=320, hidden_size=320, num_layers=2, bias=False,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.LinearBlock = nn.Sequential(
            nn.Linear(75*640, 925),
            nn.ReLU(),
            nn.Linear(925, 919),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.ConvBlock(input)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n, h_c) = self.LSTMBlock(x_x)
        #print(x.shape)
        x = x.contiguous().view(-1, 75*640)
        x = self.LinearBlock(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
danq = DanQ().to(device)
summary(danq, (4, 1000))
