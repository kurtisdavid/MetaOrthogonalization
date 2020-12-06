import torch
import torch.nn as nn
import torchvision.models

# TODO
# further adapted from: https://github.com/JerryYLi/Dataset-REPAIR/blob/master/utils/models.py
class LeNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_dims):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1   = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(inplace=True)
        )
        self.fc2   = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        self.out   = nn.Linear(84, out_dims)

    def forward(self, x):
        # conv1 + pool
        x = self.conv1(x)
        x = self.pool1(x)
        # conv2 + pool
        x = self.conv2(x)
        x = self.pool2(x)
        # reshape
        x = x.view(x.size(0), -1)
        # fc layers
        x = self.fc1(x)
        x = self.fc2(x)
        # out
        return self.out(x)
