from scipy.sparse.linalg import eigs
import numpy as np

def scaled_laplacian(W):
    '''
    :param W: np.ndarray, shape is (N, N), N is the number of vertices
    :return: scale laplacian, np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    max_eigenvalue = eigs(L, k=1, which='LR')[0].real
    L_tilde = (2 * L) / max_eigenvalue - np.identity(W.shape[0])
    # L_tilde[np.isnan(L_tilde)] = 0
    # L_tilde[np.isinf(L_tilde)] = 0
    return L_tilde

def cheb_polynomial(L, k):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{k-1}
    :param L: scalaed laplacian, np.ndarray, shape (N, N)
    :param k: the maximum order of chebyshev polynomials
    :return: chebyshev polynomials, length K, from T_0 to T_{k-1}
    '''
    L = L.astype(np.float32)
    N = L.shape[0]
    cheb_polynomials = [np.identity(N, dtype=np.float32), L.copy()]
    for i in range(2, k):
        cheb_polynomials.append(2 * L * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials