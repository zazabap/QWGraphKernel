
################################################    
# Author: SHiwen An.                           #
# Date: 2024-04-10.                            #
# Purpose: Script to quick visualize cora      #
################################################   

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from ..main.pl import * 
import sys
sys.path.append('../main')
from lib import coraSubVisual
from lib import *
from pl import *
from kernel import *
from lib import adjacencyMatrices 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
adjacency_matrix = np.load('../cora/cora_adj.npy')
features = np.load('../cora/cora_feat.npy')
labels = np.load('../cora/cora_label.npy')

# Create graph from adjacency matrix
G = nx.convert_matrix.from_numpy_array(adjacency_matrix)
for i, label in enumerate(labels):
    G.nodes[i]['label'] = label+1
    labels[i] = labels[i]+1

# Add node labels as edge weights
for edge in G.edges():
    source_label = G.nodes[edge[0]]['label']
    G.edges[edge]['weight'] = source_label

def get_nearest_neighbors(graph, start_node, num_neighbors=32):
    nearest_neighbors = list(nx.dfs_preorder_nodes(graph, start_node))[:num_neighbors]
    neighbor_edges = {node: list(graph.edges(node)) for node in nearest_neighbors}

    return nearest_neighbors,neighbor_edges

def get_nn(graph, start_node, num_neighbors=32):
    nearest_neighbors = list(nx.dfs_preorder_nodes(graph, start_node))[:num_neighbors]
    return nearest_neighbors

#############################################################
#############################################################
#############################################################
# nearest_neighbors, neighbor_edges = get_nearest_neighbors(G, start_node=1, num_neighbors=32)  # Include itself

# subG = G.subgraph(nearest_neighbors)
# subA = nx.adjacency_matrix(subG, weight='weight')
# # .todense()

# def get_G_list(graph, start_node, num_neighbors=32):
#     G_list = []
#     for i in range(9):
#         n, e = get_nearest_neighbors(graph, start_node+i, num_neighbors)
#         subG = G.subgraph(n)
#         G_list.append(subG)
#     return G_list

# gg = get_G_list(G,1,32)
# print(len(gg))
# coraSubVisual(G, G_list =gg)

# print(labels[:32])
# print(labels[nearest_neighbors])
# print("Nearest 32 neighbors (including itself):", nearest_neighbors)
# print("Nearest 32 neighbors (including itself):", subG.nodes())
#############################################################
#############################################################
#############################################################

# QML part for using SVM
print("Start of SVM analysis in Coradataset")
def get_A_list(graph, start_node, num_neighbors=32):
    A_list = []
    for i in range(2000):
        n = get_nn(graph, start_node+i, num_neighbors)
        subA = nx.adjacency_matrix(G.subgraph(n), weight='weight').todense()
        A_list.append(resizeMatrix(subA,5))
    return A_list

A_list = get_A_list(G,1,32)

print(A_list[0])
print(len(A_list))
print(len(labels))
H_list = []
for i in range(len(A_list)):
    H_list.append(qml.pauli_decompose(A_list[i]))

n_wires = 5
# obtain the density matrix after time evolution
rho_list = []
for i in range(len(H_list)):
    rho_list.append(resizeMatrix(getDensityMatrix(H_list[i],1,10,n_wires),n_wires))

rho_list_test = rho_list[1600:2000]
y_test = labels[1600:2000]

rho_list = rho_list[:1600]
y = labels[:1600]

print(len(rho_list_test))
print(len(y_test))

# example for running linear kernel
clf1 = SVC(kernel = QJSD_kernel)
print("Fitting")
clf1.fit(rho_list,y)

print(f'Accuracy on Custom Kernel: {accuracy_score(y_test, clf1.predict(rho_list_test))}')


# Calculate feature values for each node
feature_values = np.sum(features, axis=1)  # Just an example, you can use any feature aggregation method


# # Scale feature values to map to node sizes
# min_feature = min(feature_values)
# max_feature = max(feature_values)
# scaled_feature_values = [(f - min_feature) / (max_feature - min_feature) for f in feature_values]

# # Plot graph with node sizes based on features
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)  # Positions for all nodes
# nx.draw(G, pos, with_labels=False, node_size=[v * 100 for v in scaled_feature_values], cmap=plt.cm.Blues)
# # Add feature values as labels
# node_labels = {node: f'{scaled_feature_values[i]:.2f}' for i, node in enumerate(G.nodes())}
# nx.draw_networkx_labels(G, pos, labels=node_labels)
# plt.title('Cora Dataset Graph with Node Sizes based on Feature Values')
# plt.show()