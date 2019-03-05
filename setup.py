# Copyright (c) 2018 Giphy Inc.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from setuptools import setup, find_packages
try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

requirements = parse_requirements('requirements_cpu.txt', session=False)

setup(
    name='celebrity-detection-model-train',
    version='1.0.0',

    description='Facial recognition model training.',

    url='https://github.com/Giphy/celeb-detection-oss/',
    author='Giphy',
    author_email='rd@giphy.com',

    license='Proprietary',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='deeplearning giphy',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    include_package_data=True,

    install_requires=[str(x.req) for x in requirements],

    extras_require={
        'dev': [],
        'test': [],
    },
    package_data={},
    data_files=[],
    entry_points={}
)
