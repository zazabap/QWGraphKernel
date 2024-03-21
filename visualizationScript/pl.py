################################################    
# Author: SHiwen An.                           #
# Date: 2024-03-20.                            #
# Purpose: Script to apply pl                  #
################################################
import pennylane as qml
from pennylane import ApproxTimeEvolution

# Example
def test0():
    n_wires = 2
    wires = range(n_wires)

    dev = qml.device('default.qubit', wires=n_wires)

    coeffs = [1, 1]
    obs = [qml.PauliX(0), qml.PauliX(1)]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev)
    def circuit(time):
        ApproxTimeEvolution(hamiltonian, time, 1)
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    print(circuit(1))

test0()

# Time Evolution based on hamiltonian
def getTimeEvolution(H,t,n):
    n_wires = 4
    wires = range(n_wires)
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(time,ndiv):
        ApproxTimeEvolution(H,time,ndiv )
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    return circuit(t,n)

# Density Matrix calculation
def getDensityMatrix(H,t,n):
    n_wires = 8
    wires = range(n_wires)
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(time,ndiv):
        ApproxTimeEvolution(H,time,ndiv )
        return qml.density_matrix([1])
    
    return circuit(t,n)

# Von Neuman Entropy for the density matrix
def getEntropy(H,t,n):
    n_wires = 4
    wires = range(n_wires)
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(time,ndiv):
        ApproxTimeEvolution(H,time,ndiv )
        return qml.vn_entropy(wires = [0])

    return circuit(t,n)


