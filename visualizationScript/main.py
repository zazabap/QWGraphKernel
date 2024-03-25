################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-14.                            #
# Purpose: Script to load MUTAG data           #
################################################    

from lib import *
from pl import *
from kernel import * 
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

A, wires =  appendZeros(A_list[7])
print(A)
print(len(A))

# https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli_decompose.html

H = qml.pauli_decompose(A) # Simple and fast solution
H1 = qml.pauli_decompose(A_list[6])
print(H)  # The Hamiltonian of the graph
print("Obtain circuit coefficients") # The coefficient for the circuits
print(H.coeffs)
print("Time evolution of the Hamiltonian System") # time evolution of H 
time = 0
n = 100
qml.adjoint(qml.TrotterProduct(H,time, order=1, n =n))

r1 = getTimeEvolution(H, 1,10, wires)
r11 = getTimeEvolution(H, 1,20, wires)

r2 = getDensityMatrix(H, 1,10, wires)
r22 = getDensityMatrix(H1, 1,10, wires)
print("Density Matrix")
print(r2+r22)
print("Von Neuman Entropy")
print(von_neumann_entropy((r2+r22)/2))

r3 = getEntropy(H,1,10, wires )
r4 = []
for i in range(len(r1)):
    r4.append(r1[i]*r11[i])
print(r4)

# def circuit(params):
#     qml.BasisState(np.array([1, 1, 1, 1,0,0, 0, 0, 1, 1, 1, 1,0,0, 0, 0]), wires=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15])
#     qml.DoubleExcitation(params, wire s=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15])
#     return qml.expval(H)

# params = np.array(0.20885146442480412, requires_grad=True)
# circuit(params)

# explanation for the process. 
# https://quantumcomputing.stackexchange.com/questions/11899/how-can-i-decompose-a-matrix-in-terms-of-pauli-matrices
