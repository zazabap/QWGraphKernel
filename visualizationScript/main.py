################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-14.                            #
# Purpose: Script to load MUTAG data           #
################################################    

from lib import *
from pl import *
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

# Read the adjacency matrix from the file
# file_path = '../data/PROTEINS/PROTEINS_A.txt'
mutag_A = '../data/MUTAG/MUTAG_A.txt' 
mutag_node = '../data/MUTAG/MUTAG_graph_indicator.txt'
mutag_node_label = '../data/MUTAG/MUTAG_node_labels.txt'

p_A = '../data/PROTEINS/PROTEINS_A.txt' 
p_node = '../data/PROTEINS/PROTEINS_graph_indicator.txt'
p_node_label = '../data/PROTEINS/PROTEINS_node_labels.txt'

e_A = '../data/ENZYMES/ENZYMES_A.txt' 
e_node = '../data/ENZYMES/ENZYMES_graph_indicator.txt'
e_node_label = '../data/ENZYMES/ENZYMES_node_labels.txt'

# quickView(mutag_A, mutag_node, mutag_node_label)
# quickView(e_A, e_node, e_node_label)
# quickView(p_A, p_node, p_node_label)

A_list = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label)

A = A_list[6]

# https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli_decompose.html

import pennylane as qml
from pennylane import ApproxTimeEvolution

H = qml.pauli_decompose(A) # Simple and fast solution

print(H)  # The Hamiltonian of the graph
print("Obtain circuit coefficients") # The coefficient for the circuits
print(H.coeffs)
print("Time evolution of the Hamiltonian System") # time evolution of H 
time = 0
n = 100
qml.adjoint(qml.TrotterProduct(H,time, order=1, n =n))


r1 = test1(1,10)
r2 = test1(2,5)
r2 = test1(3,5)
r3 = []
for i in range(len(r1)):
    r3.append(r1[i]*r2[i])
print(r3)

r4 = test2(2,5)
r4 = test2(2,6)
r4 = test2(2,7)

# von neuman entropy of the matrix
r5 = test3(2,5)
r6 = test3(2,6)
r7 = test3(2,7)
# def circuit(params):
#     qml.BasisState(np.array([1, 1, 1, 1,0,0, 0, 0, 1, 1, 1, 1,0,0, 0, 0]), wires=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15])
#     qml.DoubleExcitation(params, wire s=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15])
#     return qml.expval(H)

# params = np.array(0.20885146442480412, requires_grad=True)
# circuit(params)

# explanation for the process. 
# https://quantumcomputing.stackexchange.com/questions/11899/how-can-i-decompose-a-matrix-in-terms-of-pauli-matrices
