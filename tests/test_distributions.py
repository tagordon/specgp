import pytest
import theano.tensor as tt
import numpy as np
import pymc3 as pm

from specgp.distributions import MvUniform

def test_uniform():
    for n in range(5):
        mvu = MvUniform.dist(lower=[0]*n, upper=[10]*n)
        x = mvu.random(size=5)
        assert mvu.logp(x).eval() == 0
    
def test_uniform_in_model():
    with pm.Model() as model:
        u = MvUniform("u", lower=[0, 0, 0], upper=[10, 10, 10]