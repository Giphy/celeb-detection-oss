# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import csv
import os


class Label(object):
    def __init__(self, tag_id, internal_id, folder_name):
        self.tag_id = tag_id
        self.internal_id = internal_id
        self.folder_name = folder_name

    def __repr__(self):
        return self.folder_name


class Labels(object):
    def __init__(self, resources_path):
        path = os.path.join(resources_path, 'face_recognition', 'labels.csv')
        self.labels_list = self.__load_labels(path)
        self.by_tag_id = self.__build_dict(self.labels_list, 'tag_id')
        self.by_model_id = self.__build_dict(self.labels_list, 'internal_id')
        self.by_folder_name = self.__build_dict(self.labels_list, 'folder_name')

    def __build_dict(self, labels, attr):
        return {getattr(label, attr): label for label in labels}

    def __load_labels(self, path):
        labels = []
        with open(path) as f:
            file_reader = csv.reader(f)
            next(file_reader)  # parse headers
            rows = sorted(file_reader, key=lambda x: x[0])
            for i, row in enumerate(rows):
                labels.append(Label(tag_id=int(row[1]), internal_id=i, folder_name=row[0]))
        return labels
