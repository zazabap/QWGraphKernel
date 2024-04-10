################################################
# Author: SHiwen An.                           #
# Date: 2024-03-29.                            #
# Purpose: Test and some old functions         #
################################################


from lib import *
from pl import *
from kernel import *
from lib import adjacencyMatrices
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

# Read the adjacency matrix from the file
# file_path = '../data/PROTEINS/PROTEINS_A.txt'
mutag_A = '../data/MUTAG/MUTAG_A.txt'
mutag_node = '../data/MUTAG/MUTAG_graph_indicator.txt'
mutag_node_label = '../data/MUTAG/MUTAG_node_labels.txt'
mutag_graph_label = '../data/MUTAG/MUTAG_graph_labels.txt'

p_A = '../data/PROTEINS/PROTEINS_A.txt'
p_node = '../data/PROTEINS/PROTEINS_graph_indicator.txt'
p_node_label = '../data/PROTEINS/PROTEINS_node_labels.txt'

e_A = '../data/ENZYMES/ENZYMES_A.txt'
e_node = '../data/ENZYMES/ENZYMES_graph_indicator.txt'
e_node_label = '../data/ENZYMES/ENZYMES_node_labels.txt'

# quickView(mutag_A, mutag_node, mutag_node_label)
# quickView(e_A, e_node, e_node_label)
# quickView(p_A, p_node, p_node_label)

A_list, y = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)

# print
print("Length of the graph:")
print(len(A_list))
print(len(y))

A, wires =  appendZeros(A_list[7])


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
# print(von_neumann_entropy((r2+r22)/2))
print(QJSD(r2,r22))

# r3 = getEntropy(H,1,10, wires )
# r4 = []
# for i in range(len(r1)):
#     r4.append(r1[i]*r11[i])
# print(r4)

# https://quantumcomputing.stackexchange.com/questions/11899/how-can-i-decompose-a-matrix-in-terms-of-pauli-matrices
