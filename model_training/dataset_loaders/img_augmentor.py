# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from imgaug import augmenters as iaa


class Augmenter(object):
    def __init__(self, augmentation_rate):
        self.augs = iaa.Sometimes(
            augmentation_rate,
            iaa.SomeOf(
                (4, 7),
                [
                    iaa.Affine(rotate=(-10, 10)),
                    iaa.Fliplr(0.2),
                    iaa.AverageBlur(k=(2, 10)),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Multiply((0.75, 1.25), per_channel=0.5),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Crop(px=(0, 20))
                ],
                random_order=True
            )
        )

    def __call__(self, sample):
        sample['x'] = self.augs.augment_image(sample['x'])
        return sample
