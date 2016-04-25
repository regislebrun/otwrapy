otwrapy module
==============

otwrapy is a collection of tools that simplify the task of wrapping
external codes in a Python environment. It provides:

- A :code:`NumericalMathFunction` decorator
- A context manager for temporary working directories
  :code:`TempWorkDir`.
- A Parallelizer function that converts any :code:`NumericalMathFunction` into
  a parralel wrapper using either :code:`multiprocessing`,
  :code:`ipyparallel` or :code:`joblib`

The tools are built on top of `OpenTURNS
<http://www.openturns.org>`_, with its users as the target audience. 
Documentation is available `here <http://felipeam86.github.io/otwrapy/>`_.


Installation
============

In a terminal, type in :

.. code-block:: shell

    $ python setup.py install

Test are available in the 'tests' directory. They can be launched with
the following command in a terminal in the otwrapy directory:

.. code-block:: shell
    
    $ py.test

Dependencies
============
- numpy
- openturns
- ipyparallel (optional)
- joblib (optional). scikit-learn comes with joblib installed (sklearn.externals.joblib)
- pytest (optional for testing)
