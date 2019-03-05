# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
import torch.nn as nn
from torch.autograd import Variable


class CenterLoss(nn.Module):
    '''
    Sample usage:

    torch.manual_seed(123)
    cl = CenterLoss(10, 2)
    print(list(cl.parameters()))
    # print(cl.centers.grad)
    y = Variable(torch.Tensor([0,0,0,1,9]))
    embedding = Variable(torch.zeros(5, 2), requires_grad=True)
    # print(embedding.grad)
    out = cl(y, embedding)
    print(out)
    out.backward()
    print(cl.centers.grad)
    print(embedding.grad)
    '''
    def __init__(self, num_classes, embedding_size, use_cuda=False):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_size))
        self.use_cuda = use_cuda

    def forward(self, y, batch):
        if self.use_cuda:
            hist = Variable(
                torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1
            ).cuda()
        else:
            hist = Variable(
                torch.histc(y.data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1
            )

        centers_count = hist.index_select(0, y.long())  # 1 + how many examples of y[i]-th class

        batch_size = batch.size()[0]
        embeddings = batch.view(batch_size, -1)

        assert embeddings.size()[1] == self.embedding_size

        centers_pred = self.centers.index_select(0, y.long())
        diff = embeddings - centers_pred
        loss = 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))
