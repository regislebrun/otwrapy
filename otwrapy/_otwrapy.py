#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General purpose OpenTURNS python wrapper tools
"""

import os
import gzip
import pickle
from tempfile import mkdtemp
import shutil
from functools import wraps
import logging
import openturns as ot
import numpy as np

__author__ = "Felipe Aguirre Martinez"
__copyright__ = "Copyright 2016, Phimeca Engineering"
__version__ = "0.4"
__email__ = "aguirre@phimeca.fr"
__all__ = ['load_array', 'dump_array', '_exec_sample_joblib',
           '_exec_sample_multiprocessing', '_exec_sample_ipyparallel',
           'NumericalMathFunctionDecorator', 'TempWorkDir', 'Parallelizer',
           'create_logger', 'Debug', 'safemakedirs']


base_dir = os.path.dirname(__file__)


def load_array(filename, compressed=False):
    """Load a (possibly compressed) pickled array.

    Parameters
    ----------
    filename : str
        Path to the file to be loaded. If the extension is '.pklz', it considers
        that the file is compressed with *gzip*.
    compressed : bool
        Indicates if the file is compressed with gzip or not.
    """
    if compressed or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'rb') as fh:
            return pickle.load(fh)
    else:
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

def dump_array(array, filename, compress=False):
    """Dump an array to a (possibly compressed) file

    Parameters
    ----------
    array : array
        Array to be compressed. Typically a np.array or ot.NumericalSample.
    filename : str
        Path where the file is dumped. If the extension is '.pklz', it considers
        that the file has to be compressed with *gzip*.
    compressed : bool
        Indicates if the file has to be compressed with gzip or not.
    """
    if compress or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)
    else:
        with open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)

def safemakedirs(folder):
    """Make a directory without raising an error if it exists.

    Parameters
    ----------
    folder : str
        Path of the folder to be created.
    """
    try:
        os.makedirs(folder)
    except OSError, e:
        # Check if it was not a "directory exist" error..
        if e.errno != 17:
            raise
        pass

def create_logger(logfile, loglevel=None):
    """Create a logger with a FileHandler at the given loglevel

    Parameters
    ----------
    logfile : str
        Filename for the logger FileHandler to be created.

    loglevel : logging level
        Threshold for the logger. Logging messages which are less severe than 
        loglevel will be ignored. It defaults to logging.DEBUG.
    """
    if loglevel is None:
        loglevel = logging.DEBUG

    logger = logging.getLogger("vgp")
    logger.setLevel(loglevel)

    # ----------------------------------------------------------
    # Create file handler which logs even DEBUG messages
    fh = logging.FileHandler(filename=logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a formatter for the file handlers
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')

    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger

class Debug(object):
    """Decorator that catches exceptions inside a function and logs them.

    A decorator used to protect functions so that exceptions are logged to a
    file. It can either be instantiated with a Logger or with a filename for
    which a logger will be created with a FileHandler. It comes specialy handy
    when you launch your codes in a non interactive environement (e.g., HPC
    cluster through submission scripts), given that Exceptions are captured
    and logged to a file.

    The great benefit of this implementation is that with a simple decorator
    you can protect the methods of your Wrapper class with a try/except
    structure. However, this might not be usefull for a deeper debugging where
    you want to have acces to the locals() of the place where the Exception
    jumped. If you bump into such a case, add a try/except structure that
    catches the Exception on the specific place. It is adviced to use the
    decorator once you have develloped the wrapper and that you are ready to
    launch your uncertainty studies.

    Parameters
    ----------
    logger : logging.Logger or str
        Either a Logger instance or a filename for the logger to be created.

    loglevel : logging level
        Threshold for the logger. Logging messages which are less severe than
        loglevel will be ignored. It defaults to logging.DEBUG.
    """

    def __init__(self, logger, loglevel=None):
        if isinstance(logger, logging.Logger):
            self.logger = logger
            if loglevel is not None:
                self.logger.setLevel(loglevel)
        elif isinstance(logger, str):
            self.logger = create_logger(logger, loglevel=loglevel)

    def __call__(self, func):
        @wraps(func)
        def func_debugged(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception, e:
                self.logger.error(e, exc_info=True)
                raise e

        return func_debugged

class NumericalMathFunctionDecorator(object):
    """Convert an OpenTURNSPythonFunction into a NumericalMathFunction

    This class is intended to be used as a decorator.

    Parameters
    ----------
    enableCache : bool (Optional)
        If True, enable cache of the returned ot.NumericalMathFunction

    Notes
    -----
    I wanted this decorator to work also with Wrapper class, but it only works
    with ParallelWrapper for the moment. The problem is that, apparently,
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
                func.__doc__ = wrapper.__doc__
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

    Parameters
    ----------
    base_temp_work_dir : str (optional)
        Root path where the temporary working directory will be created.
        Default = '/tmp'

    prefix : str (optional)
        String that preceeds the directory name.
        Default = 'run-'

    cleanup : bool (optional)
        If True erase the directory and its children at the exit.
        Default = False
    """
    def __init__(self, base_temp_work_dir='/tmp', prefix='run-', cleanup=False):
        self.dirname = mkdtemp(dir=base_temp_work_dir, prefix=prefix)
        self.cleanup = cleanup
    def __enter__(self):
        self.curdir = os.getcwd()
        os.chdir(self.dirname)
    def __exit__(self, type, value, traceback):
        os.chdir(self.curdir)
        if self.cleanup:
            shutil.rmtree(self.dirname)


def _exec_sample_joblib(func, n_cpus):
    """Return a function that executes a sample in parallel using joblib

    Parameters
    ----------
    func : Function or calable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.

    n_cpus : int
        Number of CPUs on which to distribute the function calls.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized funtion.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        from sklearn.externals.joblib import Parallel, delayed
    def _exec_sample(X):
        Y = Parallel(n_jobs=n_cpus, verbose=10)(delayed(func)(x) for x in X)
        return ot.NumericalSample(Y)

    return _exec_sample


def _exec_sample_multiprocessing(func, n_cpus):
    """Return a function that executes a sample in parallel using multiprocessing

    Parameters
    ----------
    func : Function or calable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.

    n_cpus : int
        Number of CPUs on which to distribute the function calls.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized funtion.
    """

    import time
    def _exec_sample(X):
        from multiprocessing import Pool
        p = Pool(processes=n_cpus)
        rs = p.map_async(func, X)
        p.close()
        while not rs.ready():
            time.sleep(0.1)

        Y = np.vstack(rs.get())
        return ot.NumericalSample(Y)
    return _exec_sample


def _exec_sample_ipyparallel(func, n, p):
    """Return a function that executes a sample in parallel using ipyparallel

    Parameters
    ----------
    func : Function or calable
        A callable python object, usually a function. The function should take
        an input vector as argument and return an output vector.

    n_cpus : int
        Number of CPUs on which to distribute the function calls.

    Returns
    -------
    _exec_sample : Function or callable
        The parallelized funtion.
    """
    import ipyparallel as ipp

    rc = ipp.Client()

    return ot.PythonFunction(func_sample=lambda X:
                rc[:].map_sync(func, X), n=4, p=1)

@NumericalMathFunctionDecorator(enableCache=True)
class Parallelizer(ot.OpenTURNSPythonFunction):
    """Parallelize a Wrapper using 'ipyparallel', 'joblib' or 'multiprocessing'.

    Parameters
    ----------

    where : string (Optional)
        Setup configuration according to where you run it.

    backend : string (Optional)
        Whether to parallelize using 'ipyparallel', 'joblib' or
        'multiprocessing'.

    n_cpus : int (Optional)
        Number of CPUs on which the simulations will be distributed. Needed Only
        if using 'joblib' or 'multiprocessing' as backend.

    sleep : float (Optional)
        Intentional delay (in seconds) to demonstrate the effect of
        parallelizing.
    """
    def __init__(self, wrapper, backend='multiprocessing', n_cpus=10):

        # -1 cpus means all available cpus
        if n_cpus == -1:
            import multiprocessing
            n_cpus = multiprocessing.cpu_count()

        self.n_cpus = n_cpus
        self.wrapper = wrapper
        # This configures how to run single point simulations on the model :
        self._exec = self.wrapper

        ot.OpenTURNSPythonFunction.__init__(self,
                self.wrapper.getInputDimension(),
                self.wrapper.getOutputDimension())

        self.setInputDescription(self.wrapper.getInputDescription())
        self.setOutputDescription(self.wrapper.getOutputDescription())

        assert backend in ['ipython', 'ipyparallel', 'multiprocessing',
        'joblib'], "Unknown backend"

        # This configures how to run samples on the model :
        if self.n_cpus == 1:
            self._exec_sample = self.wrapper

        elif (backend == 'ipython') or (backend == 'ipyparallel'):
            # Check that ipyparallel is installed
            try:
                import ipyparallel as ipp
                # If it is, see if there is a cluster running
                try:
                    rc = ipp.Client()
                    ipy_backend = True
                except (ipp.error.TimeoutError, IOError) as e:
                    ipy_backend = False
                    import logging
                    logging.warn('Unable to connect to an ipython cluster.')
            except ImportError:
                ipy_backend = False
                import logging
                logging.warn('ipyparallel package missing.')

            if ipy_backend:
                self._exec_sample = _exec_sample_ipyparallel(self.wrapper,
                    self.getInputDimension(), self.getOutputDimension())
            else:
                logging.warn('Using multiprocessing backend instead')
                self._exec_sample = _exec_sample_multiprocessing(self.wrapper,
                    self.n_cpus)

        elif backend == 'joblib':
            # Check that joblib is installed
            try:
                import joblib
                joblib_backend = True
            except ImportError:
                try:
                    from sklearn.externals import joblib
                    joblib_backend = True
                except ImportError:
                    joblib_backend = False
                    import logging
                    logging.warn('ipyparallel package missing.')

            if joblib_backend:
                self._exec_sample = _exec_sample_joblib(self.wrapper, self.n_cpus)
            else:
                logging.warn('Using multiprocessing backend instead')
                self._exec_sample = _exec_sample_multiprocessing(self.wrapper,
                    self.n_cpus)

        elif backend == 'multiprocessing':
            self._exec_sample = _exec_sample_multiprocessing(self.wrapper, self.n_cpus)
