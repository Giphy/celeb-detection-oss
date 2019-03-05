# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
import torch.nn as nn
from torch.autograd import Variable


class CenterLossForward(nn.Module):
    '''
    Sample usage:

    like CenterLoss, but use
    cl.train() to allow update of centers and
    cl.eval() to disable it
    '''
    def __init__(self, num_classes, embedding_size, _lambda=0.01, alpha=0.1):
        super(CenterLossForward, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self._lambda = _lambda
        self.alpha = alpha
        indices = torch.LongTensor([range(num_classes)])
        values = torch.randn(num_classes, embedding_size)
        self.register_buffer('centers', torch.sparse.FloatTensor(indices, values).coalesce())
        self.use_cuda = False

    def forward(self, y, batch):
        batch_size = batch.size()[0]
        embeddings = batch.view(batch_size, -1)
        assert embeddings.size()[1] == self.embedding_size
        self.centers = self.centers.coalesce()
        assert self.centers.is_coalesced()
        centers_pred = self.centers._values().index_select(0, y.long().data)
        indices = y.long().view(1, -1).data
        delta = embeddings - Variable(centers_pred, requires_grad=False)

        if self.training:
            sparse_delta = torch.sparse.FloatTensor(
                indices,
                delta.data,
                torch.Size([self.num_classes, self.embedding_size])
            )
            self.centers.add_(sparse_delta * self.alpha)

        return self._lambda * (delta.pow(2).sum(1) / batch_size).sum()

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        self.centers.cuda()
        return self._apply(lambda t: t.cuda(device_id))
