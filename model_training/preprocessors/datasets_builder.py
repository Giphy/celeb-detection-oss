# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import random
from multiprocessing import Pool
from itertools import repeat
import platform
import logging

from skimage import io
from skimage.transform import resize
import tensorflow  # noqa: F401
from torch.utils.data import DataLoader

import model_training.utils as utils
from model_training.preprocessors.face_detection.face_detector import FaceDetector
from model_training.dataset_loaders.raw_dataset import RawDataset


class DatasetsBuilder(object):
    def __init__(self, datasets, train_dataset_path, val_dataset_path, val_split, image_size=224, detection_margin=0.2,
                 use_cuda=False, multiple_faces=True, pool_size=os.cpu_count()):
        self.datasets = datasets
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.val_split = val_split
        self.image_size = image_size
        self.multiple_faces = multiple_faces
        self.detection_margin = detection_margin
        self.pool_size = pool_size
        self.use_cuda = use_cuda

    def perform(self):
        utils.ensure_dir(self.train_dataset_path)
        utils.ensure_dir(self.val_dataset_path)

        image_paths = utils.obtain_image_paths(self.datasets)
        aligned_images_paths = utils.obtain_image_paths([self.train_dataset_path, self.val_dataset_path])

        original_paths = [self.handle_original_path(i) for i in image_paths]
        processed_paths = [self.handle_processed_path(i) for i in aligned_images_paths]

        processed_set = set()
        for i in processed_paths:
            processed_set.update(i[1])

        image_paths_left = [i[0] for i in original_paths if i[1] not in processed_set]

        logging.info(f'GOING TO PROCESS {len(image_paths_left)} images out of {len(image_paths)}')

        random.shuffle(image_paths_left)

        val_images_count = int(len(image_paths_left) * self.val_split)
        train_images = image_paths_left[val_images_count:]
        val_images = image_paths_left[:val_images_count]
        del image_paths_left, aligned_images_paths, original_paths, processed_paths, processed_set
        if len(train_images) > 0:
            self._process_images(train_images, self.train_dataset_path)
        if len(val_images) > 0:
            self._process_images(val_images, self.val_dataset_path)

    def _process_images(self, image_paths, target_dataset_path):
        batch_size = len(image_paths) // self.pool_size
        if batch_size > 0:
            image_paths_batches = utils.batches_gen(image_paths, batch_size)
        else:
            image_paths_batches = [image_paths]
        process_params = zip(
            repeat(self),
            image_paths_batches,
            repeat(target_dataset_path)
        )

        # there is an unknown conflict with mac os, cv2 and multiprocessing pool
        # TODO: create a context for this using 'with'
        if platform.system() == 'Darwin':
            for p in process_params:
                self.process_images_batch(p)
        else:
            with Pool(self.pool_size) as p:
                p.map(self.process_images_batch, process_params)

    def handle_original_path(self, path):
        label, img = path.split('/')[-2:]
        return path, '/'.join([label, img])

    def handle_processed_path(self, path):
        label, img = path.split('/')[-2:]
        name_parts = img.split('_')

        if len(name_parts) == 1:
            name = [f'{label}/{img}']
        elif len(name_parts) == 2:
            file_format = name_parts[1].split(".")[-1]
            name_candidate = f'{label}/{name_parts[0]}.{file_format}'
            name = [f'{label}/{img}', name_candidate]
        elif len(name_parts) == 3:
            image_hash = name_parts[0]
            frame_number = name_parts[1]
            file_format = name_parts[2].split(".")[-1]
            name = ['/'.join([label, f'{image_hash}_{frame_number}.{file_format}'])]
        else:
            name = []

        return path, name

    @staticmethod
    def process_images_batch(params):
        builder, image_paths, target_dataset_path = params

        face_detector = FaceDetector(
            os.getenv('WORKDIR'),
            margin=builder.detection_margin,
            use_cuda=builder.use_cuda,
            gpu_memory_fraction=(1 / builder.pool_size)
        )

        # we set num_workers to 0 so data loader works in sync way without multiprocessing usage
        # because of "AssertionError: daemonic processes are not allowed to have children" error
        loader = DataLoader(
            RawDataset(image_paths),
            batch_size=32,
            num_workers=0,
            collate_fn=lambda b: {'img': [i['img'] for i in b], 'path': [i['path'] for i in b]}
        )

        for i, batch in enumerate(loader):
            detection_result = face_detector.perform_bulk(batch['img'], batch['path'])

            for path, face_images in detection_result.items():
                label_dir = os.path.join(target_dataset_path, path.split('/')[-2])
                utils.ensure_dir(label_dir)

                if len(face_images) > 1 and builder.multiple_faces:
                    face_images = [resize(img, (builder.image_size, builder.image_size)) for (img, bbox) in face_images]
                    filename = os.path.basename(path).split('.')[0]  # get filename and skip extension
                    for i, image in enumerate(face_images):
                        io.imsave(os.path.join(label_dir, f'{filename}_{i}.jpg'), image)
                else:
                    face_images = max(face_images, key=lambda img: img[0].shape[0] * img[0].shape[1])
                    filename = os.path.basename(path)
                    io.imsave(
                        os.path.join(label_dir, filename),
                        resize(face_images[0], (builder.image_size, builder.image_size))
                    )
