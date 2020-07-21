import exoplanet as xo
import specgp as sgp
import numpy as np
import theano.tensor as tt
import theano 

print("look_here: {}".format(theano.config.compute_test_value))
theano.config.compute_test_value = 'off'

term = xo.gp.terms.SHOTerm(log_S0=0.0, log_w0=0.0, log_Q=0.0)
kernel = sgp.terms.KronTerm(term, alpha=[1, 2, 3])
t = np.linspace(0, 10, 10)
diag = 1e-5 * np.ones((3, len(t)))
Q = kernel.alpha[:, None]*kernel.alpha[None, :]
gp = xo.gp.GP(kernel=kernel, diag=diag, mean=sgp.means.KronMean(np.zeros_like(diag)), x=t, J=2)
K = tt.slinalg.kron(term.to_dense(t, np.zeros(len(t))), Q).eval()
K = K + np.diag(np.concatenate(diag))

def test_inverse():
    z = np.random.randn(len(t)*3, 1)
    y = np.dot(np.linalg.inv(K), z)
    y = tt.as_tensor_variable(y)
    assert tt.allclose(y, gp.apply_inverse(z))
    
def test_determinant():
    det = np.linalg.det(K)
    assert tt.allclose(tt.log(det), gp.log_det)
    
def test_dot_l():
    z = np.random.randn(len(t)*3, 1)
    y = np.dot(np.linalg.cholesky(K), z)
    y = tt.as_tensor_variable(y)
    assert tt.allclose(y, gp.dot_l(z))