import torch
import torch.nn as nn


class CNNRefiner(nn.Module):
    def __init__(self):
        super(CNNRefiner, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, (3, 3))
        self.conv2 = nn.Conv2d(256, 128, (3, 3))
        self.conv3 = nn.Conv2d(256, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 128, (3, 3))
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, net, inp, net_init, flow_init):
        x = torch.cat([net_init, net], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = torch.cat([x, inp], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        x = torch.cat([x, flow_init], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)

        x = self.conv4(x)
        return x


class StateRefiner(nn.Module):
    def __init__(self):
        super(StateRefiner, self).__init__()
        self.cnn_refiner = CNNRefiner()

    def forward(self, net, inp, net_init, flow_init):
        x = self.cnn_refiner(net, inp, net_init, flow_init)
        return x
