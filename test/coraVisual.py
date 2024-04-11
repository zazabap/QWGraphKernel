
################################################    
# Author: SHiwen An.                           #
# Date: 2024-04-10.                            #
# Purpose: Script to quick visualize cora      #
################################################   

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load data
adjacency_matrix = np.load('../cora/cora_adj.npy')
features = np.load('../cora/cora_feat.npy')
labels = np.load('../cora/cora_label.npy')

# Create graph from adjacency matrix
G = nx.convert_matrix.from_numpy_array(adjacency_matrix)

# Calculate feature values for each node
feature_values = np.sum(features, axis=1)  # Just an example, you can use any feature aggregation method

# Scale feature values to map to node sizes
min_feature = min(feature_values)
max_feature = max(feature_values)
scaled_feature_values = [(f - min_feature) / (max_feature - min_feature) for f in feature_values]

# Plot graph with node sizes based on features
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=False, node_size=[v * 100 for v in scaled_feature_values], cmap=plt.cm.Blues)
# Add feature values as labels
node_labels = {node: f'{scaled_feature_values[i]:.2f}' for i, node in enumerate(G.nodes())}
nx.draw_networkx_labels(G, pos, labels=node_labels)
plt.title('Cora Dataset Graph with Node Sizes based on Feature Values')
plt.show()