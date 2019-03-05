# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import numpy as np
import torch as to

from collections import Counter
from sklearn.metrics.pairwise import cosine_distances

from model_training.helpers.clustering import clusterize
from model_training.helpers.resnet_model import ResNetModel, variable_to_numpy


class FaceRecognizer(object):
    def __init__(self, labels, resources_path, top_n=5, use_cuda=False):
        self.labels = labels
        self.top_n = top_n

        self.model = ResNetModel(
            len(self.labels.labels_list),
            weights_path=os.path.join(resources_path, os.getenv("APP_RECOGNITION_WEIGHTS_FILE")),
            use_cuda=use_cuda
        )

    def perform(self, face_images):
        if len(face_images) == 0:
            return []

        return self._calculate_predictions(face_images)

    def _calculate_predictions(self, face_images):
        predictions_tensor, embeddings_tensor = self.model(face_images, return_tensors=True)
        predictions = variable_to_numpy(predictions_tensor)
        embeddings = variable_to_numpy(embeddings_tensor)

        confidences, classes = to.max(predictions_tensor, 1)
        centers = variable_to_numpy(self.model.model_with_loss.center_loss_criterion.centers[classes.data])
        similarities = np.array([1 - self._distance(centers[i].reshape(1, -1), embeddings[i].reshape(1, -1)).squeeze()
                                 for i in range(len(embeddings))])
        is_known = variable_to_numpy(confidences) * similarities

        # get classes centers and their weights
        classes = variable_to_numpy(classes)
        counter = Counter(classes)
        weights = np.array(list(counter.values()), dtype=float)
        weights /= weights.sum()
        centers_deduped = []
        for unique_class in counter.keys():
            for cl, center in zip(classes, centers):
                if unique_class == cl:
                    centers_deduped.append(center)
                    break
        # perform gmm clustering with initialized centers and weights of each known class
        clusters = clusterize(embeddings,
                              centers=centers_deduped,
                              weights=weights,
                              output=zip(predictions, is_known))

        results = []
        for cluster in clusters:
            cluster_prediction = np.prod([x[0] for x in cluster], axis=0)
            cluster_is_known = np.mean([x[1] for x in cluster])
            if np.any(cluster_prediction > 0.0):
                cluster_prediction /= cluster_prediction.sum()
                top_prediction_ids = np.argsort(-cluster_prediction)[:self.top_n]
                results.append(([(self.labels.by_model_id[i], cluster_prediction[i])
                                 for i in top_prediction_ids], cluster_is_known))
        return results

    @staticmethod
    def _distance(x1, x2):
        return cosine_distances(x1, x2)
