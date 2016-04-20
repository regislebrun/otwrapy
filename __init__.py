#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General purpose OpenTURNS python wrapper
"""

import os
import gzip
import pickle
from tempfile import mkdtemp
import shutil
from functools import wraps
import openturns as ot

__author__ = "Felipe Aguirre Martinez"
__copyright__ = "Copyright 2015, Phimeca Engineering"
__version__ = "0.1.1"
__email__ = "aguirre@phimeca.fr"

base_dir = os.path.dirname(__file__)


def load_array(filename, compress=False):
    if compress or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'rb') as fh:
            return pickle.load(fh)
    else:
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

def dump_array(array, filename, compress=False):
    if compress or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)
    else:
        with open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)


class NumericalMathFunctionDecorator(object):
    """Convert an OpenTURNSPythonFunction into a NumericalMathFunction

    This class is intended to be used as a decorator.

    Notes
    -----
    I wanted this decorator to work also with Wrapper class, but it only works
    with ParallelWrapper for the moment. Tje problem is that, apparently,
    decorated classes are not pickable, and Wrapper instances must be pickable
    so that they can be easily distributed with `multiprocessing`

    References
    ----------
    http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/
    http://www.artima.com/weblogs/viewpost.jsp?thread=240808
    http://stackoverflow.com/questions/30714485/why-does-a-decorated-class-looses-its-docstrings
    http://stackoverflow.com/questions/30711730/decorated-class-looses-acces-to-its-attributes
    """

    def __init__(self, enableCache=True, doc=None):
        """
        Parameters
        ----------
        enableCache : bool (Optional)
            If True, enable cache of the returned ot.NumericalMathFunction
        """
        self.enableCache = enableCache
        self.doc = doc

    def __call__(self, wrapper):
        @wraps(wrapper)
        def numericalmathfunction(*args, **kwargs):
            func = ot.NumericalMathFunction(wrapper(*args, **kwargs))
            # Enable cache
            if self.enableCache:
                func.enableCache()

            # Update __doc__ of the function
            if self.doc is None:
                # Inherit __doc__ from ParallelWrapper.
                func.__doc__ = wrapper.__doc__ + wrapper.__init__.__doc__
            else:
                func.__doc__ = self.doc

            # Add the kwargs as attributes of the function for reference purposes.
            func.__dict__.update(kwargs)
            return func
        # Keep the wrapper class as reference
        numericalmathfunction.__wrapper__ = wrapper
        return numericalmathfunction


class TempWorkDir(object):
    """Implement a context manager that creates a temporary working directory.
    Create a temporary working directory on `base_temp_work_dir` preceeded by
    `prefix` and clean up at the exit if neccesary.
    See: http://sametmax.com/les-context-managers-et-le-mot-cle-with-en-python/
    """
    def __init__(self, base_temp_work_dir='/tmp', prefix='run-', cleanup=False):
        """
        Parameters
        ----------
        base_temp_work_dir : str (optional)
            Root path where the temporary working directory will be created.
            Default = '/tmp'

        prefix : str (optional)
            String that preceeds the directory name. Default = 'run-'

        cleanup : bool (optional)
            If True remove the directory and its children at the exit.
            Default = False
        """
        self.dirname = mkdtemp(dir=base_temp_work_dir, prefix=prefix)
        self.cleanup = cleanup
    def __enter__(self):
        self.curdir = os.getcwd()
        os.chdir(self.dirname)
    def __exit__(self, type, value, traceback):
        os.chdir(self.curdir)
        if self.cleanup:
            shutil.rmtree(self.dirname)
