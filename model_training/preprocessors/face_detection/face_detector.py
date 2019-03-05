# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from collections import defaultdict

import tensorflow as tf
import numpy as np
from . import network


class FaceDetector(object):
    def __init__(self, data_dir, margin=0.1, use_cuda=False, gpu_memory_fraction=1):
        self.use_cuda = use_cuda

        with tf.device(self._device()):
            with tf.Graph().as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
                sess = tf.Session(config=config)
                with sess.as_default():
                    self.pnet, self.rnet, self.onet = network.create_mtcnn(sess, os.path.join(
                        data_dir, 'face_detection'
                    ))

            self.minsize = 20  # minimum size of face
            self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            self.factor = 0.709  # scale factor
            self.margin = margin

    def perform(self, img, path=None):
        img = np.array(img)
        if len(img.shape) == 3:  # w, h, rgb
            return self.perform_single(img)
        else:
            return self.perform_bulk(img, path)

    def perform_single(self, img):
        with tf.device(self._device()):
            img = img[:, :, 0:3]
            result = []

            bounding_boxes, _ = network.detect_face(
                img,
                self.minsize,
                self.pnet,
                self.rnet,
                self.onet,
                self.threshold,
                self.factor
            )
            for bounding_box in bounding_boxes:
                det = bounding_box[0:4]
                img_size = np.asarray(img.shape)[0:2]
                bb = np.zeros(4, dtype=np.int32)
                y_margin = int((det[3] - det[1]) * self.margin)
                x_margin = int((det[2] - det[0]) * self.margin)
                bb[0] = np.maximum(det[0] - x_margin / 2, 0)
                bb[1] = np.maximum(det[1] - y_margin / 2, 0)
                bb[2] = np.minimum(det[2] + x_margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + y_margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                result.append((cropped, bounding_box))

        return result

    def perform_bulk(self, images, paths):
        with tf.device(self._device()):
            face_images = defaultdict(list)
            images_bounding_boxes = network.bulk_detect_face(
                images,
                0.2,
                self.pnet,
                self.rnet,
                self.onet,
                self.threshold,
                self.factor
            )
            for i, bounding_boxes in enumerate(images_bounding_boxes):
                if not bounding_boxes:
                    continue
                bounding_boxes = bounding_boxes[0]
                for bounding_box in bounding_boxes:
                    det = bounding_box[0:4]
                    img_size = np.asarray(images[i].shape)[0:2]
                    bb = np.zeros(4, dtype=np.int32)
                    y_margin = int((det[3] - det[1]) * self.margin)
                    x_margin = int((det[2] - det[0]) * self.margin)
                    bb[0] = np.maximum(det[0] - x_margin / 2, 0)
                    bb[1] = np.maximum(det[1] - y_margin / 2, 0)
                    bb[2] = np.minimum(det[2] + x_margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + y_margin / 2, img_size[0])
                    cropped = images[i][bb[1]:bb[3], bb[0]:bb[2], :]
                    face_images[paths[i]].append((cropped, bb))

            return face_images

    def _device(self):
        if self.use_cuda:
            return '/device:GPU:0'
        else:
            return '/cpu:0'
