import numpy as np
import pymc3 as pm
import theano.tensor as tt

__all__ = ["MvUniform"]

class MvUniform(pm.distributions.Continuous):
    """ A multivariate uniform distribution.
        
        Args:
            lower: an array of lower bounds
            upper: an array of upper bounds
    
    """
    
    def __init__(self, lower, upper, *args, **kwargs):
        
        self.size = len(lower)
        self.lower = tt.as_tensor_variable(lower)
        self.upper = tt.as_tensor_variable(upper)
        if not "testval" in kwargs:
            kwargs["testval"] = tt.mean([self.upper, self.lower], axis=0)
        super(MvUniform, self).__init__(*args, shape=self.size, **kwargs)
        
    def random(self, point=None, size=None):
        if size is None:
            size = 1
        return self.lower[None, :] + np.random.rand(size, self.size) * (self.upper - self.lower)[None, :]
    
    def logp(self, x):
        return tt.switch(tt.all(x < self.upper) & tt.all(x > self.lower), 0, -np.inf)