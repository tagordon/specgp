import theano.tensor as tt

__all__ = ["KronMean"]

class KronMean:
    """
        A constant mean for use with Kronecker-structured 
        kernels. 
        
        Args:
            tensor values: A vector containing the mean for each 
                correlated process.
    """
    
    def __init__(self, values):
        self.values = tt.as_tensor_variable(values)
        
    def __call__(self, x):
        mean = self.values
        return tt.reshape(mean.T, (mean.size,))