# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch.nn as nn
import torch.nn.functional as F

from model_training.models.adapters.center_loss_adapter import CenterLossAdapter


class AdaptDenseNetBottleneck(nn.Module):
    def __init__(self, network, embedding_size, num_classes):
        super(AdaptDenseNetBottleneck, self).__init__()
        layers = list(network.children())
        stem, original_fc = layers[:-1], layers[-1]
        num_filters = original_fc.in_features
        self.stem = nn.Sequential(*stem)
        self.fc1 = nn.Linear(num_filters, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

    # https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py#L166
    def forward(self, x):
        features = self.stem(x)
        features_out = F.relu(features, inplace=True)
        features_out = F.avg_pool2d(features_out, kernel_size=7, stride=1).view(features.size(0), -1)
        embedding = self.fc1(features_out)
        x = self.fc2(embedding)
        return x, embedding


class DenseNetCenterLoss(CenterLossAdapter):
    pass
