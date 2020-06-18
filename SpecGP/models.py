import exoplanet as xo
import means
import numbers
import pymc3 as pm
import numpy as np
from theano import tensor as tt

def validate_params(kwargs, param_names):
    if len(kwargs) > len(param_names):
        raise ValueError(
            "Too many arguments given. Expected 1 argument.")
    for p in param_names:
        if not p in kwargs:
            raise ValueError("Missing required parameter {0}.".format(p))

class Model:
    """The abstract base class that is the superclass of all 
    other Models
    """
    
    params = {}
    
    def __init__(self, nw, nterms, **kwargs):
        self.nw = nw
        self.nterms = nterms
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def loglikelihood(self, y):
        return -np.inf
    
class TransitModel(Model):
    
    """Model of one transit of one or more bodies 
    of a star with wavelength-dependent limb darkening 
    but without wavelength-dependent radii.
    
    Args:
        int nw: the number of wavelengths
        int nterms: the number of celerite SHO terms
        boolean ramp: if True, include a constant ramp in flux from A(\lambda) to B(\lambda)
        boolean add_const: if True add a constant value C(\labmda) to the flux at each wavelength
    """
    
    params = {}
    params["u1"] = None
    params["u2"] = None
    params["sig"] = None
    param_names = ["ramp", "nbodies"]
    
    def __init__(self, nw, nterms, **kwargs):
        validate_params(kwargs, self.param_names)
        super(TransitModel, self).__init__(nw, nterms, **kwargs)
        self.kernel = xo.gp.terms.Term()
        self.mean = means.Mean(self.nw)
        
        if self.ramp:
            self.params["A"] = None
            self.params["B"] = None
            
        for i in range(self.nbodies):
            self.params["delta[{0}]".format(i)] = None
            self.params["t0[{0}]".format(i)] = None
            self.params["b0[{0}]".format(i)] = None
            self.params["r0[{0}]".format(i)] = None
            
        for i in range(self.nterms):
            self.params["alpha[{0}]".format(i)] = None
            self.params["log_S0[{0}]".format(i)] = None
            self.params["log_w0[{0}]".format(i)] = None
            self.params["log_Q[{0}]".format(i)] = None
            
    def compute(self, t, params):
        for i in range(self.nterms):
            term = xo.gp.terms.SHOTerm(log_S0=params["log_S0[{0}]".format(i)],
                                       log_w0=params["log_w0[{0}]".format(i)],
                                       log_Q=params["log_Q[{0}]".format(i)])
            if i == 0:
                self.kernel = xo.gp.terms.KroneckerTerm(term, params["alpha[{0}]".format(i)])
            else:
                self.kernel += xo.gp.terms.KroneckerTerm(term, params["alpha[{0}]".format(i)])
        
        for i in range(self.nbodies):
            m = means.StarryLDCurve(self.nw, 
                                     u=[params["u1"], params["u2"]], 
                                     delta=params["delta[{0}]".format(i)], 
                                     b0=params["b0[{0}]".format(i)], 
                                     t0=params["t0[{0}]".format(i)], 
                                     r0=params["r0[{0}]".format(i)])
            if i == 0:
                self.mean = m
            else:
                self.mean += m
        
        if self.ramp:
            self.mean += means.LinearMean(self.nw, A=params["A"], B=params["B"])
        else: 
            self.mean += means.LinearMean(self.nw, A=1, B=1)
        
        mu = self.mean.evaluate(t)
        
        yerr = tt.ones(self.nw).fill(params["sig"])
        yerr = yerr[:, None] * tt.ones_like(t)
        gp = xo.gp.GP(self.kernel, t, yerr, J=2*self.nterms)   
        return gp, mu

    def log_likelihood(self, t, y):
        t = tt.as_tensor_variable(t)
        y = tt.as_tensor_variable(y)
        y = y.T.reshape((np.size(y), 1))
        gp, mu = self.compute(t, self.params)
        mu = mu.T.reshape((np.size(mu), 1))
        return gp.log_likelihood(y - mu)
    
    def sample(self, t):
        gp, mu = self.compute(tt.as_tensor_variable(t), self.params)
        n = np.random.randn(self.nw*len(t), 1)
        z = gp.dot_l(n).eval()
        z = z.T.reshape((len(t), self.nw))
        return z + mu.eval()