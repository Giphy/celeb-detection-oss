# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch.nn as nn
from model_training.models.adapters.center_loss_adapter import CenterLossAdapter


class AdaptResNet(nn.Module):
    '''
    half_size=True adapts network to 112x112 inputs, retaining the same
    size of feature maps
    '''
    def __init__(self, network, num_classes, half_size=False):
        super(AdaptResNet, self).__init__()
        layers = list(network.children())
        stem, original_fc = layers[:-1], layers[-1]
        if half_size:
            stem[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_filters = original_fc.in_features
        self.stem = nn.Sequential(*stem)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AdaptResNetBottleneck(nn.Module):
    '''
    half_size=True adapts network to 112x112 inputs, retaining the same
    size of feature maps
    '''
    def __init__(self, network, embedding_size, num_classes, half_size=False):
        super(AdaptResNetBottleneck, self).__init__()
        layers = list(network.children())
        stem, original_fc = layers[:-1], layers[-1]
        if half_size:
            stem[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_filters = original_fc.in_features
        self.stem = nn.Sequential(*stem)
        self.fc1 = nn.Linear(num_filters, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc1(x)
        x = self.fc2(embedding)
        return x, embedding


class ResNetCenterLoss(CenterLossAdapter):
    pass
