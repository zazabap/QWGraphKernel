################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-14.                            #
# Purpose: Script to load MUTAG data           #
################################################    

from datasets import load_dataset
from networkx import planar_layout
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

# Read the adjacency matrix from the file
file_path = '../data/PROTEINS/PROTEINS_A.txt'
data = pd.read_csv(file_path, header=None, names=['nodes', 'edges'])

# Create a graph
G = nx.Graph()

# Add edges to the graph based on the adjacency matrix
for _, row in data.iterrows():
    node1, node2 = row['nodes'], row['edges']
    G.add_edge(node1, node2)

# Visualize the graph
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_weight='bold', font_size=10)
plt.title('Graph Visualization')
plt.show()
# # For the train set (replace by valid or test as needed)
# dataset_pg_list = [Data(graph) for graph in dataset_hf["train"]]
# dataset_pg = DataLoader(dataset_pg_list)
