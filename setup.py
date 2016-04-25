#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
 # To use a consistent encoding
from codecs import open
from os import path

import otwrapy

"""
http://python-packaging.readthedocs.org/en/latest/minimal.html
"""

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='otwrapy',
    version=otwrapy.__version__,
    packages=find_packages(),
    extras_require = {
        'joblib':  ["joblib>=0.9.3"],
        'ipyparallel': ["ipyparallel>=5.0.1"],
    },
    author="Felipe Aguirre Martinez",
    author_email="aguirre@phimeca.com",
    description="General purpose OpenTURNS python wrapper tools",
    long_description=long_description,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data = True,
    package_data = {'otwrapy': ['examples/beam/*']},
    zip_safe=False
)
