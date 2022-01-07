import torch
import torch.nn as nn
import torch.nn.functional as F


class GetEmbedding(nn.Module):
    def __init__(self, len_vocab):
        super(GetEmbedding, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        return x




class CNNText(nn.Module):
    def __init__(self):
        super(CNNText, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(2, 100),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(3, 100),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(5, 100),
                padding=(2, 0),
                bias=True
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(906, 200, bias=False)
        self.fc2 = nn.Linear(200, 20, bias=False)
        self.fc3 = nn.Linear(20, 4, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc1(x)

        x = self.fc2(x)
        output = self.fc3(x)
        return output