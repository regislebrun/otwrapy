from otwrapy.examples.beam import Wrapper, X_distribution
import otwrapy as otw
import openturns as ot

# Monte-Carlo
ot.RandomGenerator.SetSeed(0)
X_sample = X_distribution.getSample(10)


def test_joblib():
    """Test joblib backend
    """
    # Joblib backend
    model = otw.Parallelizer(Wrapper(where='phimeca', sleep=0.2),
        backend='joblib', n_cpus=-1)
    Y_sample_joblib = model(X_sample)
    Y_mean_joblib = model(X_distribution.getMean())

def test_multiprocessing():
    """Test multiprocessing backend
    """
    # Multiprocessing backend
    model = otw.Parallelizer(Wrapper(where='phimeca', sleep=0.2),
        backend='multiprocessing', n_cpus=4)
    Y_sample_multiprocessing = model(X_sample)
    Y_mean_multiprocessing = model(X_distribution.getMean())

def test_ipython():
    """Test ipython backend
    """
    model = otw.Parallelizer(Wrapper(where='phimeca', sleep=0.2),
        backend='ipython')
    X = X_distribution.getSample(4)
    Y_sample_multiprocessing = model(X)