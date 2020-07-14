from exoplanet.gp.terms import Term
import theano
import theano.tensor as tt
import numpy as np
import scipy 

__all__ = ["KroneckerTerm"]

class KroneckerTerm(Term):
    
    r"""A Kronecker-structured covariance matrix
        of the form 
        
        .. math::
        
            K = \Sigma + T \otimes R
            
        with ``T`` defined by a celerite term and ``R`` 
        either an outer product of the form 
        
        .. math::
        
            R = \alpha\ \alpha^\mathrm{T}
            
        or an arbitrary covariance matrix. 
        
        Args:
            Term term: A celerite term. 
            tensor alpha: a vector or matrix. If alpha is a vector, 
                the matrix ``R`` is defined as the outer product of 
                alpha with itself. If alpha is a matrix, then ``R`` 
                is taken to be equal to alpha.
            
    """
    
    def __init__(self, term, alpha, **kwargs):
        self.term = term
        self.alpha = tt.as_tensor_variable(alpha)
        super(KroneckerTerm, self).__init__(**kwargs)
        
    @property
    def J(self):
        return self.term.J
    
    def __add__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return KroneckerTermSum(self, b, dtype=dtype)

    def __radd__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return KroneckerTermSum(b, self, dtype=dtype)
        
    def get_celerite_matrices(self, x, diag):
        x = tt.as_tensor_variable(x)
        ar, cr, ac, bc, cc, dc = self.term.coefficients
        
        U = tt.concatenate(
            (
                ar[None, :] + tt.zeros_like(x)[:, None],
                ac[None, :] * tt.cos(dc[None, :] * x[:, None])
                + bc[None, :] * tt.sin(dc[None, :] * x[:, None]),
                ac[None, :] * tt.sin(dc[None, :] * x[:, None])
                - bc[None, :] * tt.cos(dc[None, :] * x[:, None]),
            ),
            axis=1,
        )
        V = tt.concatenate(
            (
                tt.zeros_like(ar)[None, :] + tt.ones_like(x)[:, None],
                tt.cos(dc[None, :] * x[:, None]),
                tt.sin(dc[None, :] * x[:, None]),
            ),
            axis=1,
        )
        x = tt.reshape(tt.tile(x, (self.alpha.shape[0], 1)).T, 
                       (1, x.size*self.alpha.shape[0]))[0]
        dx = x[1:] - x[:-1]
        if self.alpha.ndim == 1:
            a = diag + (self.alpha ** 2)[:, None]*(tt.sum(ar) + tt.sum(ac))
            a = tt.reshape(a.T, (1, a.size))[0]
            U = tt.slinalg.kron(U, self.alpha[:, None])
            V = tt.slinalg.kron(V, self.alpha[:, None])
            c = tt.concatenate((cr, cc, cc))
            P = tt.exp(-c[None, :] * dx[:, None])
        else: 
            a = diag + tt.diag(self.alpha)[:, None]*(tt.sum(ar) + tt.sum(ac))
            a = tt.reshape(a.T, (1, a.size))[0]
            U = tt.slinalg.kron(U, self.alpha)
            V = tt.slinalg.kron(V, tt.eye(self.alpha.shape[0]))
            c = tt.concatenate((cr, cc, cc))
            P = tt.exp(-c[None, :] * dx[:, None])
            P = tt.tile(P, (1, self.alpha.shape[0]))

        return a, U, V, P
    
    def psd(self, omega):
        """The power spectrum of the Kronecker-structured kernel. 
           
           Args:
               tensor omega: A vector of frequencies.
               
            Returns:
                psd: An array with each row representing the power spectrum 
                    at one point in the second dimension (the dimension 
                    with covariance described by ``R``). 
        """
        
        ar, cr, ac, bc, cc, dc = self.term.coefficients
        omega = tt.reshape(
            omega, tt.concatenate([omega.shape, [1]]), ndim=omega.ndim + 1
        )
        w2 = omega ** 2
        w02 = cc ** 2 + dc ** 2
        power = tt.sum(ar * cr / (cr ** 2 + w2), axis=-1)
        power += tt.sum(
            ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
            / (w2 * w2 + 2.0 * (cc ** 2 - dc ** 2) * w2 + w02 * w02),
            axis=-1,
        )
        psd = np.sqrt(2.0 / np.pi) * power
        
        if self.alpha.ndim == 1:
            return psd[:, None] * self.alpha * self.alpha
        else:
            return psd[:, None] * tt.diag(self.alpha)
        
    def posdef(self, x, diag):
        """ Check to determine postive definiteness of the Kronecker-structured 
            covariance matrix. This operation is slow, and is thus not recommended 
            to be called repeatedly as a check during optimization. Rather, the user 
            should use this function as a guide to ensuring positive definiteness 
            of the model for varying values of the kernel parameters. 
            
            Args:
                tensor x: The input coordinates.
                tensor diag: The white noise variances. This should be an NxM 
                    array where N is the length of x and M is the size of 
                    alpha.
                    
            Returns: 
                isposdef: A boolean that is True if the covariance matrix 
                    is positive definite and False otherwise. The user will 
                    need to call ``isposdef.eval()`` to compute the returned value 
                    from the theano tensor variable. 
        """
        
        diag = tt.as_tensor_variable(diag)
        diag = tt.reshape(diag.T, (1, diag.size))[0]
        x = tt.as_tensor_variable(x)
        T = self.term.value(x[:, None] - x[None, :])
        if self.alpha.ndim == 1:
            R = self.alpha[:, None] * self.alpha[None, :]
            K = tt.slinalg.kron(T, R)
        else:
            K = tt.slinalg(T, self.alpha)
        chol = tt.slinalg.Cholesky(on_error='nan')
        L = chol(K + tt.diag(diag))
        return tt.switch(tt.any(tt.isnan(L)), np.array(False), np.array(True))
    
class KroneckerTermSum(Term):
    
    def __init__(self, *terms, **kwargs):
        self.terms = terms
        super(KroneckerTermSum, self).__init__(**kwargs)
        
    @property
    def J(self):
        return sum(term.J for term in self.terms)
    
    def __add__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return KroneckerTermSum(*self.terms, b, dtype=dtype)

    def __radd__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return KroneckerTermSum(b, *self.terms, dtype=dtype)
    
    def get_celerite_matrices(self, x, diag):
        x = tt.as_tensor_variable(x)
        diag = tt.reshape(diag.T, (1, diag.size))[0]
        a, U, V, P = self.terms[0].get_celerite_matrices(x, tt.zeros((self.terms[0].alpha.shape[0], x.size)))
        for term in self.terms[1:]:
            newa, newU, newV, newP = term.get_celerite_matrices(x, tt.zeros((term.alpha.shape[0], x.size)))
            a = tt.sum((newa, a), axis=0)
            U = tt.concatenate((U, newU), axis=1)
            V = tt.concatenate((V, newV), axis=1)
            P = tt.concatenate((P, newP), axis=1)
        a = diag + a
        a = tt.reshape(a.T, (1, a.size))[0]
        return a, U, V, P
    
    def psd(self, omega):
        power = [term.psd(omega) for term in self.terms]
        return tt.sum(power, axis=0)
    
    def posdef(self, x, diag):
        diag = tt.as_tensor_variable(diag)
        diag = tt.reshape(diag.T, (1, diag.size))[0]
        x = tt.as_tensor_variable(x)
        
        T = self.terms[0].value(x[:, None] - x[None, :])
        if self.terms[0].alpha.ndim == 1:
            R = self.terms[0].alpha[:, None] * self.terms[0].alpha[None, :]
            K = tt.slinalg.kron(T, R)
        else:
            K = tt.slinalg(T, self.terms[0].alpha)
            
        for term in self.terms:
            T = term.value(x[:, None] - x[None, :])
            if term.alpha.ndim == 1:
                R = term.alpha[:, None] * term.alpha[None, :]
                K += tt.slinalg.kron(T, R)
            else:
                K += tt.slinalg(T, term.alpha)
        chol = tt.slinalg.Cholesky(on_error='nan')
        L = chol(K + tt.diag(diag))
        return tt.switch(tt.any(tt.isnan(L)), np.array(False), np.array(True))
    
                
        