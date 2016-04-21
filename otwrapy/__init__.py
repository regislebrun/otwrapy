#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General purpose OpenTURNS python wrapper tools
"""

import os

__author__ = "Felipe Aguirre Martinez"
__copyright__ = "Copyright 2015, Phimeca Engineering"
__version__ = "0.1.1"
__email__ = "aguirre@phimeca.fr"

base_dir = os.path.dirname(__file__)

from ._otwrapy import *
from .examples import *

__all__ = (_otwrapy.__all__ +
           examples.__all__)