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
Q = alpha[:, None]*alpha[None, :]

t = tt.vector()
diag = tt.vector()
t.tag.test_value = np.linspace(0, 10, 10)
diag.tag.test_value = 1e-5 * np.ones(30)

cov = (tt.slinalg.kron(term.to_dense(t, tt.zeros_like(t)), Q)
     + tt.diag(diag))
args = [logS0, logw0, logQ, alpha, t, diag]
get_K = theano.function(args, cov)
K = get_K(-5, -2, 1, [1, 2, 3], np.linspace(0, 10, 10), 1e-5 * np.ones(30))

diag = tt.dmatrix()
mean = tt.dmatrix()
diag.tag.test_value = 1e-5 * np.ones((3, 10))
mean.tag.test_value = np.zeros_like(diag)
gp = xo.gp.GP(kernel=kernel, diag=diag, mean=sgp.means.KronMean(mean), x=t, J=2)

z = tt.dmatrix()
z.tag.test_value = np.zeros((30, 1))
args = [logS0, logw0, logQ, alpha, t, diag, z]
apply_inv = theano.function(args, gp.apply_inverse(z))
mult_l = theano.function(args, gp.dot_l(z))
args = [logS0, logw0, logQ, alpha, t, diag]
log_det = theano.function(args, gp.log_det)

args = [-5, -2, 1, [1, 2, 3], np.linspace(0, 10, 10), 1e-5 * np.ones((3, 10)), z]

def test_inverse():
    z = np.random.randn(30, 1)
    y = np.dot(np.linalg.inv(K), z)
    args = [-5, -2, 1, [1, 2, 3], np.linspace(0, 10, 10), 1e-5 * np.ones((3, 10)), z]
    assert tt.allclose(y, apply_inv(*args))
    
def test_determinant():
    det = np.linalg.det(K)
    args = [-5, -2, 1, [1, 2, 3], np.linspace(0, 10, 10), 1e-5 * np.ones((3, 10))]
    assert tt.allclose(np.log(det), log_det(*args))
    
def test_dot_l():
    z = np.random.randn(30, 1)
    args = [-5, -2, 1, [1, 2, 3], np.linspace(0, 10, 10), 1e-5 * np.ones((3, 10)), z]
    y = np.dot(np.linalg.cholesky(K), z)
    assert tt.allclose(y, mult_l(*args))