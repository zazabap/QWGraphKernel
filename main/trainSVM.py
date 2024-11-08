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

def load_matrix_list(t):
    loaded_data = np.load(f'../timeEvolution/matrix_list_{t}.npz')
    rho_list = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]
    return rho_list

# Load the list of matrices from the single file
rho_list = load_matrix_list(6)
print(rho_list)
L_list, y = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)

print("Start Trainning SVC")
start_time = time.time()
# example for running linear kernel
clf1 = SVC(kernel = QJSD_kernel)
print("Fitting")
clf1.fit(rho_list,y)

print(f'Accuracy on Custom Kernel: {accuracy_score(y, clf1.predict(rho_list))}')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")