# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
import torchvision.models as models
import torch.nn as nn

from torch.autograd import Variable
from torchvision.transforms import ToTensor

from model_training.models.cnn.resnet import AdaptResNetBottleneck, ResNetCenterLoss


def variable_to_numpy(var):
    return var.cpu().data.numpy()


class ResNetModel(object):
    def __init__(self, num_classes, weights_path, use_cuda=False, embedding_size=256):
        self.num_classes = num_classes
        self.weights_path = weights_path
        self.use_cuda = use_cuda
        self.embedding_size = embedding_size
        self.model_with_loss = self.build_resnet_50()

    def build_resnet_50(self):
        model = models.resnet50(num_classes=self.num_classes)
        model_with_bottleneck = AdaptResNetBottleneck(model, self.embedding_size, self.num_classes)

        model_with_loss = ResNetCenterLoss(
            model_with_bottleneck,
            self.num_classes,
            self.embedding_size,
            center_loss_weight=.0,
            use_cuda=self.use_cuda
        )

        model_with_loss = model_with_loss.cuda() if self.use_cuda else model_with_loss.cpu()
        model_with_loss.train(False)

        if self.use_cuda:
            checkpoint = torch.load(self.weights_path)
        else:
            checkpoint = torch.load(self.weights_path, map_location=lambda storage, loc: storage)

        model_with_loss.load_state_dict(checkpoint['model_state'])

        return model_with_loss

    def __call__(self, crops_numpy_array, return_tensors=False):
        crops_variable = self._numpy_crops_as_tensor(crops_numpy_array)
        fc2_output, embedding = self.model_with_loss.model(crops_variable)
        softmax = nn.Softmax()(fc2_output)

        if return_tensors:
            return softmax, embedding

        return variable_to_numpy(softmax), variable_to_numpy(embedding)

    def _cast(self, x):
        return x.cuda() if self.use_cuda else x.cpu()

    def _numpy_crops_as_tensor(self, crops):
        with torch.no_grad():
            return Variable(self._cast(torch.stack([ToTensor()(crop) for crop in crops])))
