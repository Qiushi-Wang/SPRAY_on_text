import torch
import torch.nn as nn
import torch.nn.functional as F


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
        conv_value1 = x1

        x2 = self.conv2(x)
        conv_value2 = x2

        x3 = self.conv3(x)
        conv_value3 = x3

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc1(x)
        fc_value1 = x

        x = self.fc2(x)
        fc_value2 = x
        output = self.fc3(x)
        return output, conv_value1, conv_value2, conv_value3, fc_value1, fc_value2