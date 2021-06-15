import torch
from torch import nn as nn


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int, if_aux_logits=True, if_init_weights=False):
        super(GoogLeNet, self).__init__()
        self.if_training = True
        self.if_aux_logits = if_aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        """
        in_channels = 3 because the inputs, RGB images, have 3 (colour) channels; out_channels (#conv networks)
        is given by the paper; kernel_size and stride are given, and padding is calculated from them (knowing that 
        conv1 halves the height and width of the input's feature map)
        """
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        """ceil_mode=True: if the output data contain non-integer values, takes the ceiling of them"""
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # self.branch1 =
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        """adaptive: no matter what is the shape of the input, can assure the shape of the output to be ..."""
        self.flatten1 = nn.Flatten(1)
        self.dropout1 = nn.Dropout2d(0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if if_aux_logits:
            if self.if_training:
                self.aux1 = InceptionAux(512, 128, 1000)
                self.aux2 = InceptionAux(528, 128, 1000)
            else:
                self.aux1 = InceptionAux(512, 128, 1000, False)
                self.aux2 = InceptionAux(512, 128, 1000, False)

        if if_init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool3(x)
        x = self.inception4a(x)
        if self.if_training and self.if_aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.if_training and self.if_aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.max_pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool1(x)
        x = self.flatten1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        if self.if_training and self.if_aux_logits:
            return x, aux1, aux2
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, num_1x1, num_3x3_r, num_3x3, num_5x5_r, num_5x5, num_pool_r):
        """"""
        """r means reduced, i.e. conv1x1 that reduces the total number of params"""
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, num_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, num_3x3_r, kernel_size=1),
            BasicConv2d(num_3x3_r, num_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, num_5x5_r, kernel_size=1),
            BasicConv2d(num_5x5_r, num_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, num_pool_r, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        """concat the outputs of each branch against the dimension 1 (the dimension of channel)"""
        return output


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_1x1, num_classes, if_training=True):
        """"""
        super(InceptionAux, self).__init__()
        self.if_training = if_training
        self.layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),  # in: aux1: N * 512 * 4 * 4, aux2: N * 528 * 4 * 4; N: num_batch
            BasicConv2d(in_channels, num_1x1, kernel_size=1),   # in: aux1: N * 512 * 4 * 4, aux2: N * 528 * 4 * 4
            nn.Flatten(1),  # in: aux1: N * 128 * 4 * 4
            nn.Dropout2d(0.5),
            nn.Linear(num_1x1 * 4 * 4, 1024),   # in: N * 2048
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(1024, num_classes)    # in: N * 1024, out: N * num_class
        ) if self.if_training else nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),  # in: aux1: N * 512 * 4 * 4, aux2: N * 528 * 4 * 4; N: num_batch
            BasicConv2d(in_channels, num_1x1, kernel_size=1),  # in: aux1: N * 512 * 4 * 4, aux2: N * 528 * 4 * 4
            nn.Flatten(1),  # in: aux1: N * 128 * 4 * 4
            nn.Linear(num_1x1 * in_channels * in_channels, 1024),  # in: N * 2048
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)  # in: N * 1024, out: N * num_class
        )
        """
        average_pool won't modify the shape of the input
        """
        """
        after conv2 shape: (num_batch, num_1x1, in_channels, in_channels); num_1x1:= 128, in_channels = 4 in the paper
        """

    def forward(self, x):
        x = self.layers(x)
        return x

