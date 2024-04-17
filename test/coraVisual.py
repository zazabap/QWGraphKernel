
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
# Load data
adjacency_matrix = np.load('../cora/cora_adj.npy')
features = np.load('../cora/cora_feat.npy')
labels = np.load('../cora/cora_label.npy')

# Create graph from adjacency matrix
G = nx.convert_matrix.from_numpy_array(adjacency_matrix)
for i, label in enumerate(labels):
    G.nodes[i]['label'] = label

# Add node labels as edge weights
for edge in G.edges():
    source_label = G.nodes[edge[0]]['label']
    G.edges[edge]['weight'] = source_label

def get_nearest_neighbors(graph, start_node, num_neighbors=32):
    nearest_neighbors = list(nx.dfs_preorder_nodes(graph, start_node))[:num_neighbors]
    neighbor_edges = {node: list(graph.edges(node)) for node in nearest_neighbors}

    return nearest_neighbors,neighbor_edges

nearest_neighbors, neighbor_edges = get_nearest_neighbors(G, start_node=1, num_neighbors=32)  # Include itself

subG = G.subgraph(nearest_neighbors)
subA = nx.adjacency_matrix(subG, weight='weight')
# .todense()

def get_G_list(graph, start_node, num_neighbors=32):
    G_list = []
    for i in range(12):
        n, e = get_nearest_neighbors(graph, start_node+i, num_neighbors)
        subG = G.subgraph(n)
        G_list.append(subG)
    return G_list

gg = get_G_list(G,1,32)
print(len(gg))
coraSubVisual(G, G_list =gg)

# print(labels[:32])
print(labels[nearest_neighbors])

print("Nearest 32 neighbors (including itself):", nearest_neighbors)
print("Nearest 32 neighbors (including itself):", subG.nodes())
print(subA)
# print(len(subA))
# print("edges", neighbor_edges)

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