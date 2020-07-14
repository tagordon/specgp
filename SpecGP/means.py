import theano.tensor as tt

__all__ = ["KroneckerConstant"]

class KroneckerConstant:
    """
        A constant mean for use with Kronecker-structured 
        kernels. 
        
        Args:
            tensor values: A vector containing the mean for each 
                point in the second dimension.
    """
    
    def __init__(self, values):
        self.values = tt.as_tensor_variable(values)
        
    def __call__(self, x):
        mean = self.values[:, None] * tt.ones_like(x)
        return tt.reshape(mean.T, (x.size*self.values.size,))