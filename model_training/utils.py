# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import ssl
import logging
import socket
import csv
import time
import glob
import sys
import cv2
import http.client
import numpy as np

from contextlib import contextmanager
from urllib.error import URLError
from urllib.request import HTTPError
from math import ceil
from scipy import misc

ACCEPTABLE_ERRORS = (
    ssl.CertificateError,
    ssl.SSLError,
    URLError,
    AttributeError,
    socket.timeout,
    HTTPError,
    OSError,
    cv2.error,
    http.client.error,
    UnicodeError
)


def ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            logging.warning(f'Path {directory} already exists')


def batches_gen(iterable, n=1):
    size = len(iterable)
    for ndx in range(0, size, n):
        yield iterable[ndx:min(ndx + n, size)]


def path_to_label(path):
    return path.split('/')[-2]


def labels_by_name(path):
    labels = {}
    with open(path) as f:
        file_reader = csv.reader(f)
        next(file_reader)
        rows = sorted(file_reader, key=lambda x: x[0])
        for i, row in enumerate(rows):
            labels[row[0]] = i
    return labels


def build_bboxes_mapping(bboxes_mapping_file):
    bboxes_reader = csv.reader(open(bboxes_mapping_file, 'r'))

    return {
        bbox[0].split('/')[-1]: (
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[1]) + int(bbox[3]),
            int(bbox[2]) + int(bbox[4])
        ) for bbox in bboxes_reader
    }


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def obtain_image_paths(datasets_paths):
    image_paths = []
    for dataset_path in datasets_paths:
        image_paths.extend(glob.glob(os.path.join(dataset_path, '**/*.jpg')))

    return image_paths


@contextmanager
def show_time(name='Execution'):
    t = time.time()
    yield
    logging.info(f'{name} time is {time.time() - t} seconds')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_3_channels(img):
    '''Converts a greyscale image to 3-channel image'''
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, -1)
    return img


class retry(object):
    def __init__(self, name='', times=5, pause=5, exception=Exception):
        self._name = name
        self._times = times
        self._pause = pause
        self._exception = exception

    def __call__(self, func):
        def wrapped_func(*args):
            for i in range(self._times):
                try:
                    return func(*args)
                except self._exception as ex:
                    if i < self._times - 1:
                        logging.warning(
                            f'Cannot perform {self._name} due to {type(ex)}:{ex}')
                        time.sleep(self._pause)
                    else:
                        logging.critical(
                            f'Cannot perform {self._name} due to {type(ex)}:{ex} for {self._times} times. Exiting...')
                        sys.exit(1)

        return wrapped_func


def evenly_spaced_sampling(array, n):
    """Choose `n` evenly spaced elements from `array` sequence"""
    length = len(array)

    if n == 0 or length == 0:
        return []

    if n > length:
        n = length

    return [array[ceil(i * length / n)] for i in range(n)]


def preprocess_image(img, image_size):
    return misc.imresize(img, (image_size, image_size), interp='bilinear')
