# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from skimage import io
from torch.utils.data import Dataset

import model_training.utils as utils


class RawDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = sorted(image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, num):
        img = io.imread(self.image_files[num])
        img = utils.ensure_3_channels(img)
        img_path = self.image_files[num]
        return {'img': img, 'path': img_path}
