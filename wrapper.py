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
__author__ = "Felipe Aguirre Martinez"
__copyright__ = "Copyright 2015, Phimeca Engineering"
__version__ = "0.1"
__email__ = "aguirre@phimeca.fr"

class Wrapper(ot.OpenTURNSPythonFunction):
    """
    This function is intended to be lightweight so that it can be easily
    distributed across several nodes in a cluster.
    """

    # Possible configurations
    places = ['phimeca', 'poincare', 'tgcc']
    base_dirs = {'phimeca': '/home/aguirre/Calculs/Formation-PRACE/beam',
                 'poincare': '/gpfshome/mds/staff/vdubourg/beam',
                 'tgcc': '/ccc/work/cont003/xxx/aguirref/Formation-PRACE/beam'}
    temp_dirs = {'phimeca': '/tmp',
                 'poincare': '/tmp',
                 'tgcc': '/ccc/scratch/cont003/xxx/aguirref/Formation-PRACE/'}

    def __init__(self, where='phimeca', sleep=0.0):
        """
        Parameters
        ----------
        where : string (Optional)
            Setup configuration according to where you run it
        """
        assert where in Wrapper.places, "Only valid places are {}".format(Wrapper.places)

        self.base_dir = Wrapper.base_dirs[where]
        self.temp_work_dir_base = Wrapper.temp_dirs[where]
        self.input_template = os.path.join(self.base_dir, 'beam_input_template.xml')
        self.executable = os.path.join(self.base_dir, 'beam -v -x beam.xml')
        self.sleep = sleep

        # Number of output values:
        ot.OpenTURNSPythonFunction.__init__(self, 4, 1)

    def _exec(self, X):
        """Run the model in the shell.

        Parameters
        ----------
        X : float (something like ot.NumericalPoint or a numpy 1D array)
            Input vector of size :math:`n` on which the model will be evaluated
        """
        try:
            # Create intentional delay
            time.sleep(self.sleep)

            # File management (move to temporary working directory)
            old_wrk_dir = os.getcwd()
            temp_work_dir = mkdtemp(dir=self.temp_work_dir_base, prefix='ot-beam-example-')
            os.chdir(temp_work_dir)
            
            # Create input file
            self._create_input_file(X)

            # Execute code
            runtime = self._call(X)

            # Retrieve output (see also coupling_tools.get_value)
            Y = self._parse_output()

            # Clear temporary working directory
            os.chdir(old_wrk_dir)
            shutil.rmtree(temp_work_dir)
        except Exception as e:
            os.chdir(old_wrk_dir)
            raise e
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

class ParallelWrapper(ot.OpenTURNSPythonFunction):
    """
    Class that distributes calls to the class Wrapper across a cluster using
    either joblib or ipython.
    """
    def __init__(self, where='phimeca', backend='joblib',
        n_cpus=10, view=None, sleep=0.0):
        """
        Parameters
        ----------

        where : string (Optional)
            Either 'phimeca', 'poincare' or 'tgcc'. The wrapper will be configured
            according to where you run it.

        backend : string (Optional)
            Wheter to parallelize using 'joblib' or 'ipython'.

        n_cpus : int (Optional)
            Number of CPUs on which the simulations will be distributed. Needed Only
            if using 'joblib' as backend.

        view : IPython load_balanced_view (Optional)
            If backend is 'ipython', you must pass a view as an argument.

        sleep : float (Optional)
            Intentional delay (in seconds) to demonstrate the effect of
            parallelizing.
        """

        assert where in Wrapper.places, "Only valid places are {}".format(Wrapper.places)
        self.n_cpus = n_cpus
        self.wrapper = Wrapper(where=where, sleep=sleep)

        if backend == 'ipython':
            assert view is not None, "You have to provide a View to comunicate with IPython engines"
            self._exec_sample = ot.PythonFunction(func_sample=lambda X:
                view.map_sync(self.wrapper, X), n=4, p=1)
        elif backend == 'joblib':
            self._exec_sample = self._exec_sample_joblib

        ot.OpenTURNSPythonFunction.__init__(self, 4, 1)
        self.setInputDescription(['Load', 'Young modulus', 'Length', 'Inertia'])
        self.setOutputDescription(['deviation'])
        

    def _exec(self, X):
        """Single call of the model
        
        Parameters
        ----------
        X : 1D array
            Input vector of size :math:`m` on which the model will be evaluated
        """
        return self.wrapper(X)

    def _exec_sample_joblib(self, X):
        """Run the model using parallel computing.

        Parameters
        ----------
        X : 2D array
            Input sample of size :math:`n x m` on which the model will be evaluated
        """
        from sklearn.externals.joblib import Parallel, delayed
        Y = Parallel(n_jobs=self.n_cpus, verbose=10)(delayed(self.wrapper)(x) for x in X)
        return ot.NumericalSample(Y)



def ParallelizedBeam(*args, **kwargs):
    __doc__ = Wrapper.__doc__
    func = ot.NumericalMathFunction(ParallelWrapper(*args, **kwargs))
    func.enableCache()
   
    return func


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

    parser.add_argument('-run', default=False, type=bool, nargs='?', const='True',
        help='If True, run the model', choices=[True, False])

    parser.add_argument('-dump', default=False, type=bool, nargs='?', const='True',
        help='If True, dump the output for later posttreatment', choices=[True, False])

    args = parser.parse_args()


    model = ParallelizedBeam(where=args.where, backend='joblib', 
        n_cpus=args.n_cpus)

    if args.MonteCarlo is not None:
        from probability_model import X_distribution
        ot.RandomGenerator.SetSeed(args.seed)
        N = int(args.MonteCarlo[0])
        X = X_distribution.getSample(N)
        print "Generated a MonteCarlo DOE of size {}".format(N)

    elif args.X is not None:
        if isinstance(args.X[0], str) and os.path.isfile(args.X[0]):
            with open(args.X[0], 'rb') as fh:
                X = pickle.load(fh)
            print "Loaded a DOE of size {} from file: '{}'".format(X.getSize(),
                args.X[0])
        else:
            X = ot.NumericalPoint([float(x) for x in args.X])


    if args.run:
        launchdir = os.getcwd()
        try:
            Y = model(X)
            print "Finished evaluationg the model. Take a look at 'Y' variable."
            if args.dump:
                dump_array(Y, 'OutputSample.pkl')
                print "The output has been saved to 'OutputSample.pkl'"
        except Exception as e:
            os.chdir(launchdir)
            raise e