import torch
import torch.nn as nn


class CNNRefiner(nn.Module):
    def __init__(self):
        super(CNNRefiner, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(256, 128, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 256, (3, 3), padding=1)

        self.flow_conv1 = nn.Conv2d(2, 16, (3, 3), padding=1, stride=2)
        self.flow_conv2 = nn.Conv2d(16, 128, (3, 3), padding=1, stride=2)
        self.flow_conv3 = nn.Conv2d(128, 128, (3, 3), padding=1, stride=2)

        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.flow_bn1 = nn.BatchNorm2d(16)
        self.flow_bn2 = nn.BatchNorm2d(128)
        self.flow_bn3 = nn.BatchNorm2d(128)

    def forward(self, net, inp, net_init, flow_init):
        x = torch.cat([net_init, net], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = torch.cat([x, inp], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        fx = self.flow_conv1(flow_init)
        fx = self.flow_bn1(fx)
        fx = self.gelu(fx)
        fx = self.flow_conv2(fx)
        fx = self.flow_bn2(fx)
        fx = self.gelu(fx)
        fx = self.flow_conv3(fx)
        fx = self.flow_bn3(fx)
        fx = self.gelu(fx)

        x = torch.cat([x, fx], dim=1)
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
