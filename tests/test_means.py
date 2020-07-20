import pytest
import exoplanet as xo
import specgp as sgp
import theano 
import theano.tensor as tt
import numpy as np

def test_structure():
    t = np.linspace(0, 10, np.random.randint(100))
    
    mu_lam = np.random.randint(10, size=10)
    kronmu = mu_lam[:, None] * np.ones(len(t))
    
    gp_mean = sgp.means.KronMean(kronmu)(t).eval()
    assert np.all(gp_mean == np.tile(mu_lam, (len(t),)))
    
def test_in_gp():
    t = np.linspace(0, 10, np.random.randint(100))
    mu_lam = np.random.randint(10, size=10)
    kronmu = mu_lam[:, None] * np.ones(len(t))
    
    term = xo.gp.terms.SHOTerm(log_S0=0, log_w0=0, log_Q=0)
    kernel = sgp.terms.KronTerm(term, alpha=np.linspace(1, 2, 10))
    gp = xo.gp.GP(kernel=kernel, x=t, diag=np.ones_like(kronmu), mean=kronmu, J=2)
    assert np.all(gp.mean_val.eval() == kronmu)