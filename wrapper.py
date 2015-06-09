#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
General purpose OpenTURNS python wrapper
"""

import openturns as ot
ot.ResourceMap.SetAsUnsignedInteger('cache-max-size', int(1e6))
from tempfile import mkdtemp
import pickle
import gzip
import os
import numpy as np
import shutil
from xml.dom import minidom
import time
from functools import wraps

__author__ = "Felipe Aguirre Martinez"
__copyright__ = "Copyright 2015, Phimeca Engineering"
__version__ = "0.1.1"
__email__ = "aguirre@phimeca.fr"


def NumericalMathFunction(wrapper):
    #http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/
    #http://www.artima.com/weblogs/viewpost.jsp?thread=240808
    def inner(*args, **kwargs):
        func = ot.NumericalMathFunction(wrapper(*args, **kwargs))
        func.enableCache()
        # Inherit __doc__ from ParallelWrapper.
        func.__doc__ = wrapper.__doc__ + wrapper.__init__.__doc__
        # Add the kwargs as attributes of the function for reference purposes.
        func.__dict__.update(kwargs)
        return func
   
    return inner

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
                func.__doc__ = wrapper.__doc__
            else:
                func.__doc__ = self.doc

            # Add the kwargs as attributes of the function for reference purposes.
            func.__dict__.update(kwargs)
            return func
        # Keep the wrapper class as reference
        numericalmathfunction.__wrapper__ = wrapper
        return numericalmathfunction

class Wrapper(ot.OpenTURNSPythonFunction):
    """
    This Wrapper is intended to be lightweight so that it can be easily
    distributed across several nodes in a cluster.
    """

    # Possible configurations
    places = ['phimeca', 'poincare', 'tgcc']
    configs = {
               'phimeca':
                  {
                   'base_dir': '/home/aguirre/Calculs/Formation-PRACE/beam',
                   'temp_work_dir': '/tmp'
                  },
               'poincare':
                  {
                   'base_dir': '/gpfshome/mds/staff/vdubourg/beam',
                   'temp_work_dir': '/tmp'
                  },
               'tgcc':
                  {
                   'base_dir': '/ccc/work/cont003/xxx/aguirref/Formation-PRACE/beam',
                   'temp_work_dir': '/ccc/scratch/cont003/xxx/aguirref/Formation-PRACE/'
                  }
              }

    def __init__(self, where='phimeca', sleep=0.0):
        """
        Parameters
        ----------
        where : string or dict (Optional)
            Setup configuration according to where you run it. If it is a string,
            configurations are loaded from Wrapper.configs[where]. If it is a
            dict, it must have 'base_dir' and 'temp_work_dir' as keys
            pointing, respectively, to where the executable binary is found and
            to the base path where runs will be executed (e.g., /tmp or /scratch)
        """
        # Look for setup configuration on 'Wrapper.configs[where]'.
        if isinstance(where, str):
            assert where in Wrapper.places, "Only valid places are {}".format(Wrapper.places)
            self.__dict__.update(Wrapper.configs[where])
        # Take setup configuration from 'where' dict.
        elif isinstance(where, dict):
            assert ('base_dir' in where.keys()) \
               and ('temp_work_dir' in where.keys()), "Wrong configuration dict"
            self.__dict__.update(where)

        self.input_template = os.path.join(self.base_dir, 'beam_input_template.xml')
        self.executable = os.path.join(self.base_dir, 'beam -x beam.xml')
        self.sleep = sleep

        # Number of input/output values:
        super(Wrapper, self).__init__(4, 1)

    def _exec(self, X):
        """Run the model in the shell.

        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated
        """

        # Create intentional delay
        time.sleep(self.sleep)

        # File management. Move to temp work dir. Cleanup at the end
        with TempWorkDir(self.temp_work_dir, 'ot-beam-example-', True):
        
            # Create input file
            self._create_input_file(X)

            # Execute code
            runtime = self._call(X)

            # Retrieve output (see also ot.coupling_tools.get_value)
            Y = self._parse_output()

        return Y

    def _create_input_file(self, X):
        """Create the input file required by the code.

        Replace the values of the vector X to their corresponding tokens on the
        self.input_template and create the input file `beam.xml` on the current
        working directory.

        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated
        """
        ot.coupling_tools.replace(
            self.input_template,
            'beam.xml',
            ['@F','@E','@L','@I'],
            X)

    def _call(self, X):
        """Execute code on the shell
        
        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated

        Returns
        -------
        runtime : float
            Total runtime (wall time and not cpu time)
        """

        time_start = time.time()
        ot.coupling_tools.execute(self.executable)
        time_stop = time.time()

        return time_stop - time_start


    def _parse_output(self):
        """Parse the output given by the code

        Returns
        -------
        Y : list
            Output vector of the model. Univariate in this case.
        """

        # Retrieve output (see also coupling_tools.get_value)
        xmldoc = minidom.parse('_beam_outputs_.xml')
        itemlist = xmldoc.getElementsByTagName('outputs')
        deviation = float(itemlist[0].attributes['deviation'].value)

        # Make a list out of the output(s)
        Y = [deviation]

        return Y

####################################################################
# ------------------------ Parallel Wrapper ------------------------
####################################################################

@NumericalMathFunctionDecorator(enableCache=True)
class ParallelWrapper(ot.OpenTURNSPythonFunction):
    """
    Class that distributes calls to the class Wrapper across a cluster using
    either 'ipython', 'joblib' or 'multiprocessing'.
    """
    def __init__(self, where='phimeca', backend='joblib',
        n_cpus=10, view=None, sleep=0.0):
        """
        Parameters
        ----------

        where : string (Optional)
            Setup configuration according to where you run it. If it is a string,
            configurations are loaded from Wrapper.configs[where]. If it is a
            dict, it must have 'base_dir' and 'temp_work_dir' as keys
            pointing, respectively, to where the executable binary is found and
            to the base path where runs will be executed (e.g., /tmp or /scratch)

        backend : string (Optional)
            Whether to parallelize using 'ipython', 'joblib' or 'multiprocessing'.

        n_cpus : int (Optional)
            Number of CPUs on which the simulations will be distributed. Needed Only
            if using 'joblib' or 'multiprocessing' as backend.

        view : IPython load_balanced_view (Optional)
            If backend is 'ipython', you must pass a view as an argument.

        sleep : float (Optional)
            Intentional delay (in seconds) to demonstrate the effect of
            parallelizing.
        """

        self.n_cpus = n_cpus
        self.wrapper = Wrapper(where=where, sleep=sleep)
        # This configures how to run single point simulations on the model :
        self._exec = self.wrapper

        # This configures how to run samples on the model :
        if self.n_cpus == 1:
            self._exec_sample = self.wrapper
        elif backend == 'ipython':
            assert view is not None, "You have to provide a View to comunicate with IPython engines"
            self._exec_sample = ot.PythonFunction(func_sample=lambda X:
                view.map_sync(self.wrapper, X), n=4, p=1)
        elif backend == 'joblib':
            self._exec_sample = self._exec_sample_joblib
        elif backend == 'multiprocessing':
            self._exec_sample = self._exec_sample_multiprocessing

        ot.OpenTURNSPythonFunction.__init__(self, 4, 1)
        self.setInputDescription(['Load', 'Young modulus', 'Length', 'Inertia'])
        self.setOutputDescription(['deviation'])

    def _exec_sample_joblib(self, X):
        """Run the model in parallel using joblib.

        Parameters
        ----------
        X : 2D array
            Input sample of size :math:`n x m` on which the model will be evaluated
        
        Returns
        -------
        Y : NumericalSample
            Output Sample of the model.
        """
        from sklearn.externals.joblib import Parallel, delayed
        Y = Parallel(n_jobs=self.n_cpus, verbose=10)(delayed(self.wrapper)(x) for x in X)
        return ot.NumericalSample(Y)

    def _exec_sample_multiprocessing(self, X):
        """Run the model in parallel using the multiprocessing module.

        Parameters
        ----------
        X : 2D array
            Input sample of size :math:`n x m` on which the model will be evaluated
        
        Returns
        -------
        Y : NumericalSample
            Output Sample of the model.
        """

        from multiprocessing import Pool
        p = Pool(processes=self.n_cpus)
        rs = p.map_async(self.wrapper, X)
        p.close()
        while not rs.ready():
            time.sleep(0.1)
            
        Y = np.vstack(rs.get())
        return ot.NumericalSample(Y)


def dump_array(array, filename, compress=False):
    if compress or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)
    else:
        with open(filename, 'wb') as fh:
            pickle.dump(array, fh, protocol=2)

def load_array(filename, compressed=False):
    if compressed or (filename.split('.')[-1] == 'pklz'):
        with gzip.open(filename, 'rb') as fh:
            return pickle.load(fh)
    else:
        with open(filename, 'rb') as fh:
            return pickle.load(fh)


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
        self.dirname = mkdtemp(dir=base_temp_work_dir, prefix='ot-beam-example-')
        self.cleanup = cleanup
    def __enter__(self):
        self.curdir = os.getcwd()
        os.chdir(self.dirname)
    def __exit__(self, type, value, traceback):
        os.chdir(self.curdir)
        if self.cleanup:
            shutil.rmtree(self.dirname)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description="Python wrapper example used for the PRACE training on HPC and uncertainty.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-where', default='poincare', type=str, choices=['phimeca', 'poincare', 'tgcc'],
        help='Place where simulations will run.')
    
    parser.add_argument('-seed', default=int(0), type=int,
        help='Seed number for the random number generator')
        
    parser.add_argument('-MonteCarlo', nargs=1,
        help="Launch a MonteCarlo simulation of given size")
    
    parser.add_argument('-X', nargs='*',
        help='List of floats [X1, X2.. Xp] or PATH to a pickled DOE')
    
    parser.add_argument('-n_cpus', default=-1, type=int,
        help="(Optional) number of cpus to use.")

    parser.add_argument('-backend', default='joblib', type=str,
        choices=['joblib', 'multiprocessing'],
        help="Whether to parallelize using 'joblib' or 'multiprocessing'.")

    parser.add_argument('-run', default=False, type=bool, nargs='?', const='True',
        help='If True, run the model', choices=[True, False])

    parser.add_argument('-dump', default=False, type=bool, nargs='?', const='True',
        help='If True, dump the output for later posttreatment', choices=[True, False])

    args = parser.parse_args()

    model = ParallelWrapper(where=args.where, backend=args.backend, 
        n_cpus=args.n_cpus)
    print "The wrapper has been instantiated as 'model'."

    if args.MonteCarlo is not None:
        from probability_model import X_distribution
        ot.RandomGenerator.SetSeed(args.seed)
        N = int(args.MonteCarlo[0])
        X = X_distribution.getSample(N)
        print "Generated a MonteCarlo DOE of size {}".format(N)

    elif args.X is not None:
        if isinstance(args.X[0], str) and os.path.isfile(args.X[0]):
            X = load_array(args.X[0])
            print "Loaded a DOE of size {} from file: '{}'".format(X.getSize(),
                args.X[0])
        else:
            X = ot.NumericalPoint([float(x) for x in args.X])


    if args.run:
        Y = model(X)
        # Dump the results if asked
        if args.dump:
            dump_array(Y, 'OutputSample.pkl')
            print "The output has been saved to 'OutputSample.pkl'"
        else:
            print "Finished evaluationg the model. Take a look at 'Y' variable."
    elif (args.MonteCarlo is not None) or (args.X is not None):
        print "The desired input is ready to be run using --> 'Y = model(X)'"


