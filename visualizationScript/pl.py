################################################    
# Author: SHiwen An.                           #
# Date: 2024-03-20.                            #
# Purpose: Script to apply pl                  #
################################################
import pennylane as qml
from pennylane import ApproxTimeEvolution
import numpy as np

# Example

# Time Evolution based on hamiltonian
def getTimeEvolution(H,t,n, w):
    """
    Get the time evolution for the Hamiltonian System

    Args:
        H (): Hamiltonian Object for exp(-iHt)
        t (float): Time evolution for exp(-iHt)
        n (int): Division for the ApproxTimeEvolution exp(-iHt/n)*exp(-iHt/n)
        w (int): Number of wires in the Circuit.
    Returns:
        list: Tensor for the time evolution
    """
    n_wires = w
    wires = range(n_wires)
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(time,ndiv):
        ApproxTimeEvolution(H,time,ndiv )
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    return circuit(t,n)

# Density Matrix calculation
def getDensityMatrix(H,t,n, w):
    """
    Get the Density Matrix for the Hamiltonian System

    Args:
        H (): Hamiltonian Object for exp(-iHt)
        t (float): Time evolution for exp(-iHt)
        n (int): Division for the ApproxTimeEvolution exp(-iHt/n)*exp(-iHt/n)
        w (int): Number of wires in the Circuit.
    Returns:
        Matrix based on the eigenvectors
    """
    n_wires = w
    wires = range(n_wires)
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(time,ndiv):
        ApproxTimeEvolution(H,time,ndiv )
        return qml.density_matrix(range(n_wires))
    
    return circuit(t,n)

#https://discuss.pennylane.ai/t/different-outputs-of-two-methods-for-the-same-entropy/3800

# Von Neuman Entropy for the density matrix
def getEntropy(H,t,n,w):
    """
    Get the Density Matrix for the Hamiltonian System

    Args:
        H (): Hamiltonian Object for exp(-iHt)
        t (float): Time evolution for exp(-iHt)
        n (int): Division for the ApproxTimeEvolution exp(-iHt/n)*exp(-iHt/n)
        w (int): Number of wires in the Circuit.
    Returns:
        float: Von Neuman Entropy Value 
    """    
    n_wires = w
    wires = range(n_wires)
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(time,ndiv):
        ApproxTimeEvolution(H,time,ndiv )
        return qml.vn_entropy(wires = [0])

    return circuit(t,n)


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