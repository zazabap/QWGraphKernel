################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-14.                            #
# Purpose: Script to load MUTAG data           #
################################################    

from lib import *
from pl import *
from kernel import *
from hodgeLaplacian import * 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

# Give a list of laplacian matrices
# L_list, y = LaplacianMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)

# L_list, y = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)

L_list, y = HLMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)


# print 
print("Length of the graph:")
print(L_list[0])
print(L_list[1])


print(len(y))

# Check the largest wires is 5 
n_wires = 5
for i in range(len(L_list)):
    L_list[i] = resizeMatrix(L_list[i],n_wires)

# Matrix decoposition to Hamiltonian
H_list = []
for i in range(len(L_list)):
    H_list.append(qml.pauli_decompose(L_list[i]))

# obtain the density matrix after time evolution
rho_list = []
for i in range(len(H_list)):
    rho_list.append(resizeMatrix(getDensityMatrix(H_list[i],1,10,n_wires),n_wires))

rho_list = rho_list[:7]
y = y[:7]
# example for running linear kernel
clf1 = SVC(kernel = QJSD_kernel)
print("Fitting")
clf1.fit(rho_list,y)

print(f'Accuracy on Custom Kernel: {accuracy_score(y, clf1.predict(rho_list))}')

# Partial reference of the code: 
# https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli_decompose.html
# https://quantumcomputing.stackexchange.com/questions/11899/how-can-i-decompose-a-matrix-in-terms-of-pauli-matrices

# H = qml.pauli_decompose(A) # Simple and fast solution
# H1 = qml.pauli_decompose(L_list[6])
# print(H)  # The Hamiltonian of the graph
# print("Obtain circuit coefficients") # The coefficient for the circuits
# print(H.coeffs)
# print("Time evolution of the Hamiltonian System") # time evolution of H 
# time = 0
# n = 100
# qml.adjoint(qml.TrotterProduct(H,time, order=1, n =n))

