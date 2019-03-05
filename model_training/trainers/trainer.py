# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import copy
import logging
import os
import time

from torch.autograd import Variable
import torch

import model_training.utils as utils


class Trainer(object):
    def __init__(self, model, data_loaders, epochs, optimizer, lr_scheduler, weights_path, use_cuda=False, restore=None,
                 log_per_batches=100, save_per_batches=10000):
        self.model = model
        self.data_loaders = data_loaders
        self.epochs = range(epochs)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weights_path = weights_path
        self.use_cuda = use_cuda
        self.restore = restore
        self.log_per_batches = log_per_batches
        self.save_per_batches = save_per_batches

    def perform(self):
        best_loss = float('inf')

        if self.restore is not None:
            checkpoint = torch.load(os.path.join(self.weights_path, f'{self.restore}_model_states.pkl'))
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epochs = [i + checkpoint['epoch'] + 1 for i in self.epochs]
            best_loss = checkpoint['loss']

        total_batches = len(self.data_loaders['train'])

        self._cast(self.model)

        best_model_weights = copy.deepcopy(self.model.state_dict())

        for epoch in self.epochs:
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.lr_scheduler.step()
                    self.model.train(True)
                else:
                    self.model.train(False)

                timestamp = time.time()
                avg_loss = utils.AverageMeter()
                avg_top_1 = utils.AverageMeter()
                avg_top_5 = utils.AverageMeter()
                self.model.reset_summary()

                for i, batch in enumerate(self.data_loaders[phase]):
                    inputs = Variable(self._cast(batch['x']))
                    labels = Variable(self._cast(batch['y']))

                    prediction, loss = self.model(inputs, labels)

                    top_1, top_5 = self._accuracy(prediction, labels, topk=(1, 5))
                    avg_loss.update(loss.data[0])
                    avg_top_1.update(top_1.data[0])
                    avg_top_5.update(top_5.data[0])

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    if i % self.save_per_batches == 0 and phase == 'train':
                        self._save_state(epoch, avg_loss.avg, prefix='train')

                    if i % self.log_per_batches == 0 and phase == 'train':
                        # here we calculate time per batch in seconds
                        # and then multiply this value by remaining batches count
                        # and divide by 3600 to calculate remaining hours
                        time_per_batch_in_seconds = ((time.time() - timestamp) / self.log_per_batches)
                        remaining_time_hours = time_per_batch_in_seconds * (total_batches - i) / 3600
                        logging.info(
                            f'PHASE {phase} '
                            f'EPOCH {epoch + 1}/{self.epochs[-1] + 1} BATCH {i + 1}/{total_batches} '
                            f'TIME {remaining_time_hours:.2f}h LOSS: {avg_loss.avg:.2f} '
                            f'TOP-1: {avg_top_1.avg:.2f} TOP-5: {avg_top_5.avg:.2f}'
                        )
                        self.model.summary_writer.add_scalar('train_loss', avg_loss.avg, epoch * total_batches + i)
                        self.model.summary_writer.add_scalar('train_top_1', avg_top_1.avg, epoch * total_batches + i)
                        self.model.summary_writer.add_scalar('train_top_5', avg_top_5.avg, epoch * total_batches + i)
                        self.model.push_summary(epoch * total_batches + i, 'train_')
                        avg_loss.reset()
                        avg_top_1.reset()
                        avg_top_5.reset()
                        self.model.reset_summary()
                        timestamp = time.time()

                if phase == 'val':
                    logging.info(
                        f'FINISHED EPOCH {epoch + 1}/{self.epochs[-1] + 1} VAL LOSS: {avg_loss.avg:.2f}'
                        f'VAL TOP-1: {avg_top_1.avg:.2f} VAL TOP-5: {avg_top_5.avg:.2f}'
                    )
                    self.model.summary_writer.add_scalar('validation_loss', avg_loss.avg, epoch)
                    self.model.summary_writer.add_scalar('validation_top_1', avg_top_1.avg, epoch)
                    self.model.summary_writer.add_scalar('validation_top_5', avg_top_5.avg, epoch)
                    self.model.push_summary(epoch, 'validation_')

                    if avg_loss.avg < best_loss:
                        best_model_weights = copy.deepcopy(self.model.state_dict())
                        logging.info(f'BEST VAL LOSS ACHIEVED {avg_loss.avg} PREVIOUS {best_loss}')
                        best_loss = avg_loss.avg
                        self._save_state(epoch, best_loss, prefix='best')
                else:
                    self.model.push_embedding_summary(epoch)

        self.model.load_state_dict(best_model_weights)
        return self.model

    def _accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

    def _save_state(self, epoch, loss, prefix='best'):
        torch.save(
            {
                'epoch': epoch,
                'loss': loss,
                'optimizer_state': self.optimizer.state_dict(),
                'model_state': self.model.state_dict()
            },
            os.path.join(self.weights_path, f'{prefix}_model_states.pkl')
        )

        # TODO: investigate whether we can save the model without several attributes
        # torch.save(self.model, os.path.join(self.weights_path, f'{prefix}_model_object_snapshot.pkl'))

    def _cast(self, x):
        return x.cuda() if self.use_cuda else x
