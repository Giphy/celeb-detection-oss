# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import logging
import threading

from tensorboard import main as tb


class TensorboardClient(object):
    def __init__(self, dir_path, port):
        self.dir_path = dir_path
        self.port = str(port)

    def run(self):
        # Remove http and tensorflow messages
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        tb.program.FLAGS.logdir = self.dir_path
        tb.program.FLAGS.port = self.port
        t = threading.Thread(target=tb.program.main, args=([tb.default.get_plugins(),
                                                            tb.default.get_assets_zip_provider()]))
        t.start()
