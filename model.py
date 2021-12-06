import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class TextCNN(nn.Module):
    def __init__(self, label_num, batch_size):
        super(TextCNN, self).__init__()
        self.label_num = label_num
        self.batch_size = batch_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(3, 100),
                padding=(1, 0),
                bias=True
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(2 * 100, self.label_num, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        #x = x.view(-1)
        conv_value = x
        x = self.fc(x)
        fc_value = x
        output = self.sig(x)
        return output, conv_value, fc_value



