import exoplanet as xo
import specgp as sgp
import numpy as np
import theano.tensor as tt
import theano 


logS0 = tt.scalar()
logw0 = tt.scalar()
logQ = tt.scalar()
logS0.tag.test_value = -5.0
logw0.tag.test_value = -2.0
logQ.tag.test_value = 1.0
term = xo.gp.terms.SHOTerm(S0=tt.exp(logS0), w0=tt.exp(logw0), Q=tt.exp(logQ))

alpha = tt.vector()
alpha.tag.test_value = np.array([1, 2, 3])
kernel = sgp.terms.KronTerm(term, alpha=alpha)

t = tt.vector()
diag = tt.dmatrix()
t.tag.test_value = np.linspace(0, 10, 10)
diag.tag.test_value = 1e-5 * tt.ones((3, 10))
mean = tt.dmatrix()
mean.tag.test_value = 0.0 * tt.ones((3, 10))
Q = alpha[:, None]*alpha[None, :]
gp = xo.gp.GP(kernel=kernel, diag=diag, mean=sgp.means.KronMean(mean), x=t, J=2)
K = tt.slinalg.kron(term.to_dense(t, tt.zeros_like(t)), Q)
d = tt.vector()
d.tag.test_value = 1e-5 * np.ones(30)
K = K + tt.diag(d)

def test_inverse():
    z = tt.dmatrix()
    z.tag.test_value = np.random.randn(30, 1)
    y = tt.dot(tt.nlinalg.matrix_inverse(K), z)
    y = tt.as_tensor_variable(y)
    assert tt.allclose(y, gp.apply_inverse(z))
    
def test_determinant():
    det = tt.nlinalg.det(K)
    assert tt.allclose(tt.log(det), gp.log_det)
    
def test_dot_l():
    z = tt.dmatrix()
    z.tag.test_value = np.random.randn(30, 1)
    y = tt.dot(tt.slinalg.cholesky(K), z)
    y = tt.as_tensor_variable(y)
    assert tt.allclose(y, gp.dot_l(z))