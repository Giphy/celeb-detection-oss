# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from collections import defaultdict
from sklearn.mixture import GaussianMixture


def clusterize(points, n_components=2, covariance_type='tied',
               centers=None, weights=None, output=None, random_state=1000):
    if centers is not None:
        n_components = len(centers)

    if output is None:
        output = points

    if len(points) < 2:
        return [list(output)]

    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,
                          means_init=centers,
                          weights_init=weights,
                          random_state=random_state)
    gmm.fit(points)
    labels = gmm.predict(points)

    clusters = defaultdict(list)
    for label, point in zip(labels, output):
        clusters[label].append(point)

    return sorted(clusters.values(), key=lambda x: len(x), reverse=True)
