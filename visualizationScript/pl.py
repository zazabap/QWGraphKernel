################################################    
# Author: SHiwen An.                           #
# Date: 2024-03-20.                            #
# Purpose: Script to apply pl                  #
################################################
import pennylane as qml
from pennylane import ApproxTimeEvolution

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
        return qml.density_matrix([1])
    
    return circuit(t,n)

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


