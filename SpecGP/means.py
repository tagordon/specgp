import numpy as np
import exoplanet as xo

def validate_params(kwargs, param_names):
    if len(kwargs) > len(param_names):
        raise ValueError(
            "Too many arguments given. Expected 1 argument.")
    for p in param_names:
        if not p in kwargs:
            raise ValueError("Missing required parameter {0}.".format(p))

class Mean:
    """The abstract base class that is the superclass of 
    all other means
    """
    
    def __init__(self, nw=1, **kwargs):
        self.nw = nw
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __add__(self, m):
        return MeanSum(self, m)
    
    def __radd__(self, m):
        return MeanSum(m, self)
    
    def __mul__(self, m):
        return MeanProd(self, m)
    
    def __rmul__(self, m):
        return MeanProd(m, self)
    
    def evaluate(self, x):
        return np.zeros(len(x), nw)
    
class ConstantMean(Mean):
    """A constant mean
    
    Args:
        Float C: The constant value of the mean or an 
            array of values, one for each wavelength.
    """
    
    param_names = ("C")
    
    def __init__(self, nw=1, **kwargs):
        
        validate_params(kwargs, self.param_names)
        super(ConstantMean, self).__init__(nw, **kwargs)
        
        try:
            self.C = np.full(nw, self.C)
        except ValueError:
            print("Length of C does not match number of wavelengths, nw. "
                  "C should be either a float or an array of length nw.")
        
    def evaluate(self, x):
        return self.C[:, None] * np.ones_like(x)
    
class LinearMean(Mean):
    
    """A mean function that increases linearly from A to B
    
    Args:
        Float A: The initial value of the linear ramp or an 
            array of values, one for each wavelength
        Float B: The final value of the linear ramp or an 
            array of values, one for each wavelength
    """
    
    param_names = ("A", "B")
    
    def __init__(self, nw=1, **kwargs):
        validate_params(kwargs, self.param_names)
        super(LinearMean, self).__init__(nw, **kwargs)
        
        try:
            self.A = np.full(nw, self.A)
            self.B = np.full(nw, self.B)
        except ValueError:
            print("Length of A or B does not match number of wavelengths, nw. "
                  "A and B should be either floats or arrays of length nw.")
        
    def evaluate(self, x):
        rise = self.B - self.A
        run = x.max() - x.min()
        slope = rise / run
        return self.A[:, None] + (x - x.min())*slope[:, None]
    
class StarryPhaseCurve(Mean):
    
    """The phase curve of a starry model
    
    """
    pass