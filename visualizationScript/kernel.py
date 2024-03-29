################################################    
# Author: SHiwen An.                           #
# Date: 2024-03-25.                            #
# Purpose: Kernels for svm                     #
################################################

from pl import *
import numpy as np
from sklearn.datasets import make_classification

x,y = make_classification(n_samples = 1000)

def linear_kernel(x_i, x_j):
    return x_i.dot(x_j.T)


def appendZeros(M):
    """
    Appends zeros to a given matrix to resize it to the next power of 2 in both dimensions.

    Parameters:
    - M: numpy array, input matrix of shape (m, m)

    Returns:
    - O: numpy array, output matrix of shape (2^n, 2^n), where n is the smallest integer such that 2^n > m.
      The elements of M are copied into the top-left corner of O, with the remaining elements filled with zeros.

    Example:
    >>> M = np.array([[1, 2],
    ...               [3, 4]])
    >>> O = append_zeros_to_matrix(M)
    >>> print(O)
    [[1. 2. 0. 0.]
     [3. 4. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    """
    m = len(M)
    n = 1
    while 2 ** n <= m:
        n += 1
    new_size = 2 ** n

    # Create a new matrix O of size (2^n) x (2^n) filled with zeros
    O = np.zeros((new_size, new_size), dtype=M.dtype)

    # Copy elements of M into the top-left corner of O
    O[:m, :m] = M

    return O,n

def resizeMatrix(M,n):
    """
    Appends zeros to a given matrix to resize it to the next power of 2 in both dimensions.

    Parameters:
    - M: numpy array, input matrix of shape (m, m)
    - n: 2**n the resize of matrix

    Returns:
    - O: numpy array, output matrix of shape (2^n, 2^n), where n is the smallest integer such that 2^n > m.
      The elements of M are copied into the top-left corner of O, with the remaining elements filled with zeros.

    Example:
    >>> M = np.array([[1, 2],
    ...               [3, 4]])
    >>> O = append_zeros_to_matrix(M)
    >>> print(O)
    [[1. 2. 0. 0.]
     [3. 4. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    """
    m = len(M)
    new_size = 2 ** n

    # Create a new matrix O of size (2^n) x (2^n) filled with zeros
    O = np.zeros((new_size, new_size), dtype=M.dtype)

    # Copy elements of M into the top-left corner of O
    O[:m, :m] = M

    return O

def von_neumann_entropy(rho):
    """
    Calculates the von Neumann entropy of a density matrix rho.

    Parameters:
    - rho: numpy array, density matrix representing the quantum state

    Returns:
    - S: float, von Neumann entropy of the density matrix rho

    The von Neumann entropy is defined as S = -Tr(R), where R = rho * log2(rho).

    Example:
    >>> rho = np.array([[0.5, 0.5],
    ...                 [0.5, 0.5]])
    >>> entropy = von_neumann_entropy(rho)
    >>> print(entropy)
    1.0
    """
    from scipy import linalg as la
    R = rho*(la.logm(rho)/la.logm(np.matrix([[2]])))
    S = -np.matrix.trace(R)
    return(S)

# self defined kenerl 
def QJSD(rho,sigma):
    phi = []
    for i in range(len(rho)):
        t = []
        for j in range(len(rho[i])):
            t.append((rho[i][j]+sigma[i][j])/2)
        phi.append(t)
    s = von_neumann_entropy(phi)
    a = von_neumann_entropy(rho)/2
    b = von_neumann_entropy(sigma)/2
    return s-a-b

def QJSK(rho,sigma):
    r = []
    for i in range(len(rho)):
        r.append(QJSD(rho[i],sigma[i]))
    return r