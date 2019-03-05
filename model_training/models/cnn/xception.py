# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch.nn as nn
import torch.nn.functional as F

from model_training.models.adapters.center_loss_adapter import CenterLossAdapter


class AdaptXceptionBottleneck(nn.Module):
    def __init__(self, network, embedding_size, num_classes):
        super(AdaptXceptionBottleneck, self).__init__()
        layers = list(network.children())
        stem, original_fc = layers[:-1], layers[-1]
        num_filters = original_fc.in_features
        self.stem = nn.Sequential(*stem)
        self.fc1 = nn.Linear(num_filters, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        embedding = self.fc1(x)
        x = self.fc2(embedding)
        return x, embedding


class XceptionCenterLoss(CenterLossAdapter):
    pass
