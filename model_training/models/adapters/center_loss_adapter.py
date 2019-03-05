# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch.nn as nn

from model_training.models.losses.center_loss import CenterLoss
import model_training.utils as utils


class CenterLossAdapter(nn.Module):
    def __init__(self, model, num_classes, embedding_size, center_loss_weight=0.5, use_cuda=False, summary_writer=None):
        super(CenterLossAdapter, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_criterion = CenterLoss(num_classes, embedding_size, use_cuda)
        self.use_cuda = use_cuda
        self.center_loss_weight = center_loss_weight
        self.summary_writer = summary_writer
        self.avg_crossentropy_loss = utils.AverageMeter()
        self.avg_center_loss = utils.AverageMeter()

    def forward(self, inputs, labels):
        prediction, embedding = self.model(inputs)

        crossentropy_loss = self.criterion(prediction, labels)
        center_loss = self.center_loss_weight * self.center_loss_criterion(labels, embedding)

        self.avg_crossentropy_loss.update(crossentropy_loss.data[0])
        self.avg_center_loss.update(center_loss.data[0])

        loss = crossentropy_loss + center_loss

        return prediction, loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))

    def push_summary(self, step, prefix=''):
        self.summary_writer.add_scalar(f'{prefix}crossentropy_loss', self.avg_crossentropy_loss.avg, step)
        self.summary_writer.add_scalar(f'{prefix}center_loss', self.avg_center_loss.avg, step)

    def push_embedding_summary(self, step):
        self.summary_writer.add_embedding(self.center_loss_criterion.centers.data, global_step=step)

    def reset_summary(self):
        self.avg_crossentropy_loss.reset()
        self.avg_center_loss.reset()
