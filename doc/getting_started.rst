===============
Getting started
===============

Installation
------------

In a terminal, type in :

.. code-block:: shell

    $ python setup.py install

Test are available in the 'tests' directory. They can be launched with
the following command in a terminal in the otwrapy directory:

.. code-block:: shell

    $ py.test

Dependencies
------------
- numpy
- openturns
- ipyparallel (optional)
- joblib (optional). scikit-learn comes with joblib installed (sklearn.externals.joblib)
- pytest (optional for testing)
- sphinx and numpydoc for building the doc


How to use the module
---------------------

You can either use it as a module. For example :

.. code-block:: python

    import otwrapy as otw
    model = otw.Parallelizer(Wrapper(...), backend='joblib', n_cpus=-1)

In which case, :code:`model` will be your parallelized :py:class:`openturns.NumericalMathFunction` using
`joblib <https://pythonhosted.org/joblib/>`_ as a backend and as many cpus as available in your
machine (:code:`n_cpus=-1`). For further information, refer to the :doc:`api`.

Or you can simply consider this as a cookbook and take what you consider useful.
