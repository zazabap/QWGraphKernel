################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-14.                            #
# Purpose: Script to load MUTAG data           #
################################################    

from lib import *
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


quickView(e_A, e_node, e_node_label)
A_list = adjacencyMatrices(mutag_A, mutag_node, mutag_node_label)