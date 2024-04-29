################################################    
# Author: SHiwen An.                           #
# Date: 2024-04-29.                            #
# Purpose: Time Matrices save to numpy form    #
################################################    

from lib import *
from pl import *
from kernel import *
from hodgeLaplacian import * 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

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

# Give a list of laplacian matrices# L_list, y = LaplacianMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)
# L_list, y = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)
L_list, y = HLMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)

print("Resize Matrices")
# Check the largest wires is 5 
n_wires = 5
for i in range(len(L_list)):
    L_list[i] = resizeMatrix(L_list[i],n_wires)

print("Matrix Decomposition")
# Matrix decoposition to Hamiltonian
H_list = []
for i in range(len(L_list)):
    H_list.append(qml.pauli_decompose(L_list[i]))

# obtain the density matrix after time evolution
# rho_list = []

for t in range(10):
    rho_list = []
    print(f"Time Evolution {t}")  # Corrected the string formatting
    start_time = time.time()
    for i in range(len(H_list)):
        print(f"r{i}")  # Corrected the string formatting
        rho_list.append(getDensityMatrix(H_list[i],t+1,100,n_wires))

    print("Evolution done")
    np.savez(f'matrix_list_{t}.npz', *rho_list)  # Corrected the filename formatting

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(rho_list[0])
    print(rho_list[1])

    print(f"Elapsed time: {elapsed_time} seconds")