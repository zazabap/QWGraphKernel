################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-23.                            #
# Purpose: Create Simple Gates and Circuit     #
#          to practice the algorithms          #
#          reverse engineering                 # 
################################################

# Library for Qiskit
from qiskit import * 
from qiskit.result import Counts
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, Aer, execute

# Library for Random function
import random
import numpy as np
import matplotlib.pyplot as plt

# Using 4 qubits 
# Which will give 16*16 matrix
# Add number of gates with given n  
# return Adjacency Matrix after Update
def randomCircuit(n):
    circ = QuantumCircuit(4,4)

    for i in range(n):
        circ.h(random.randint(0,3))
        circ.h(random.randint(0,3))
        circ.h(random.randint(0,3))
        circ.x(random.randint(0,3))
        circ.y(random.randint(0,3))
        circ.z(random.randint(0,3))
        circ.mcrx(np.pi *i*0.1, [0,1,2],3)
    circuit_drawer(circ, output='mpl', filename='CTQW.png')  # 'mpl' for Matplotlib output
    print(circ)
    matrix = Operator(circ).data
    matrix = normalizeMatrix(matrix)
    print(matrix)
    print(np.trace(matrix))
    return np.where(matrix !=0,1,0)

def normalizeMatrix(A):
    norm_factor = np.linalg.norm(A)
    norm_matrix = A/norm_factor
    return norm_matrix

A = randomCircuit(4)
print(A)
