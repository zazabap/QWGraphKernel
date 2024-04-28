################################################
# Author: SHiwen An.                           #
# Date: 2024-04-28.                            #
# Purpose: quick data plot for the graph       #
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

quickView(mutag_A, mutag_node, mutag_node_label)
quickView(e_A, e_node, e_node_label)
quickView(p_A, p_node, p_node_label)

# Adjacency list archive
A_list, y = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label, mutag_graph_label)
