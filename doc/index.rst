Welcome to otwrapy's documentation!
===================================

:code:`otwrapy` is a collection of tools that simplify the task of wrapping
external codes in a Python environment. It provides:

- A :code:`NumericalMathFunction` decorator
- A context manager for temporary working directories
  :code:`TempWorkDir`.
- A Parallelizer function that converts any :code:`NumericalMathFunction` into
  a parralel wrapper using either :code:`multiprocessing`,
  :code:`ipyparallel` or :code:`joblib`.

The tools are built on top of `OpenTURNS
<http://www.openturns.org>`_, with its users as the target
audience. You can either use it as a module. For example :


.. code-block:: python

    import otwrapy as otw
    model = otw.Parallelizer(Wrapper(...), backend='joblib', n_cpus=-1)

In which case, :code:`model` will be your parallelized wrapper using
:code:`joblib` as a backend and as many cpus as available in your
machine (:code:`n_cpus=-1`). Or you can simply consider this as a
cookbook and take what you consider useful.

:code:`otwrapy` comes from the experience of wrapping a lot of
different external codes at `Phimeca engineering
<http://www.phimeca.com>`_. We are a company specialized on
uncertainty treatment and we assist our clients introducing the
probabilistic dimension in their so far deterministic studies.

.. warning::
    While fully usable, otwrapy is still pre-1.0 software and has **no**
    backwards compatibility guarantees until the 1.0 release occurs! Please
    make sure to be carefully **anytime you upgrade**!


