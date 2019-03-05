# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import glob
import re

from skimage import io
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import model_training.utils as utils


class FacialDataset(Dataset):
    def __init__(self, dataset_path, labels_file_path, transform=None, return_path=False, only_labels=None):
        self.labels = utils.labels_by_name(labels_file_path)
        self.image_files = self.__load_image_files(only_labels, dataset_path)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, num):
        x = io.imread(self.image_files[num])
        x = utils.ensure_3_channels(x)

        label_name = utils.path_to_label(self.image_files[num])
        y = self.labels[label_name]
        sample = {'x': x, 'y': y}

        if self.return_path:
            sample['path'] = self.image_files[num]

        if self.transform:
            sample = self.transform(sample)

        # .copy() explanation
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        sample['x'] = transforms.ToTensor()(sample['x'].copy())

        return sample

    @staticmethod
    def __load_image_files(only_labels, dataset_path):
        if only_labels:
            image_files = []
            for label in only_labels:
                label = re.sub(r'([\[\]])', r'[\g<1>]', label)
                image_files += glob.glob(os.path.join(dataset_path, f'{label}/*.jpg'))
        else:
            image_files = glob.glob(os.path.join(dataset_path, '**/*.jpg'))

        return image_files
