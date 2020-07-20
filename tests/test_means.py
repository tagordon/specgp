import pytest
import exoplanet as xo
import specgp as sgp
import theano 
import theano.tensor as tt
import numpy as np

def test_structure():
    t = np.linspace(0, 10, np.random.randint(100))
    a = tt.dmatrix('a')
    mu = sgp.means.KronMean(a)(t)
    f = theano.function([a], mu)
    
    mu_lam = np.random.randint(10, size=10)
    kronmu = mu_lam[:, None] * np.ones(len(t))
    assert np.all(f(kronmu) == np.tile(mu_lam, (len(t),)))
    
def test_in_gp():
    t = np.linspace(0, 10, np.random.randint(100))
    mu_lam = np.random.randint(10, size=10)
    kronmu = mu_lam[:, None] * np.ones(len(t))
    
    term = xo.gp.terms.SHOTerm(log_S0=0, log_w0=0, log_Q=0)
    kernel = sgp.terms.KronTerm(term, alpha=[1, 2])
    gp = xo.gp.GP(kernel=kernel, x=t, diag=np.ones_like(kronmu), mean=kronmu, J=2)
    assert np.all(gp.mean_val.eval() == kronmu)