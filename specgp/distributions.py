import pymc3 as pm
import numpy as np
import theano.tensor as tt

__all__ = ["MvUniform"]

def MvUniform(label, lower, upper, **kwargs):
    """ A multivariate uniform distribution.
        
        Args:
            lower: an array of lower bounds
            upper: an array of upper bounds
    
    """
    
    n = len(lower)
    lower = tt.as_tensor_variable(lower)
    upper = tt.as_tensor_variable(upper)
    logp = lambda x: tt.switch(tt.all(x < upper) & tt.all(x > lower), 0, -np.inf)
    random = lambda point=None, size=None: lower + np.random.rand(n)*(upper - lower)
    return pm.DensityDist(label, logp, random=random, shape=n, **kwargs)