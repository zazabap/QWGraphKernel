################################################
# Author: SHiwen An.                           #
# Date: 2023-12-23.                            #
# Purpose: Quick Visualization for graphs      #
#          different type of graphs            #
################################################

import networkx as nx
import pandas as pd
import numpy as np
import linecache
import matplotlib.pyplot as plt
from collections import defaultdict

# A Function to visualize 
# ENZYMES MUTAG PROTEINS
def quickView(file_A, file_node, file_node_label):
    data = pd.read_csv(file_A, header=None, names=['nodes', 'edges'])

    # Create a graph
    G = nx.Graph()
    G_list = []
    node_pre = 1
    # Add edges to the graph based on the adjacency matrix
    for _, row in data.iterrows():
        node1, node2 = row['nodes'], row['edges']
        if int(linecache.getline(file_node, node1)) == node_pre :
            G.add_edge(node1, node2)
        else: 
            node_pre = int(linecache.getline(file_node, node1))
            G_list.append(G)
            G = nx.Graph()
            G.add_edge(node1, node2)
    
        node_attr = {node1: {'label': linecache.getline(file_node_label, node1),  
                    node2: {'label': linecache.getline(file_node_label, node2)}}}
        nx.set_node_attributes(G, node_attr)

    plt.figure(figsize=(15,10))
    colors = ['red', 'orange', 'yellow', 
          'green', 'blue','skyblue',
          'violet', 'brown', 'gray']

    for i in range(9):
        plt.subplot(3,3,1+i)
        pos = nx.spring_layout(G_list[i])
        node_labels = nx.get_node_attributes(G_list[i],'label')
        nx.draw(G_list[i], pos, with_labels=True, node_color=colors[i], node_size=200, font_weight='bold', font_size=10)
        nx.draw_networkx_labels(G,pos,labels=node_labels)
        plt.title(f'Graph {i+1}')    

    plt.show()

def coraSubVisual(G, G_list):

    # Create a graph
    plt.figure(figsize=(15,10))
    colors = ['red', 'orange', 'yellow', 
          'green', 'blue','skyblue',
          'violet', 'brown', 'gray']

    for i in range(9):
        plt.subplot(3,3,1+i)
        pos = nx.spring_layout(G_list[i])
        node_labels = nx.get_node_attributes(G_list[i],'label')
        nx.draw(G_list[i], pos, with_labels=True, node_color=colors[i], node_size=200, font_weight='bold', font_size=10)
        nx.draw_networkx_labels(G,pos,labels=node_labels)
        plt.title(f'Graph {i+1}')    

    plt.show()

# Self defined random walk function 
# on graph define the steps on adj_matrix
def random_walk_steps(adj_matrix, steps, num_walks):
    n = len(adj_matrix)
    steps_taken = np.zeros(n, dtype=int)

    for _ in range(num_walks):
        current_position = np.random.randint(n) # Starting node

        for _ in range(steps):
            # Get neighbors of the current node
            neighbors = [i for i in range(n) if adj_matrix[current_position][i] == 1]
            
            if not neighbors:
                break  # If no neighbors, break the walk

            # Move to a random neighbor
            current_position = np.random.choice(neighbors)

            # Increment steps taken for the current node
            steps_taken[current_position] += 1

    return steps_taken


# Function to condense the matrix
def condense_matrix(original_matrix):
    m = len(original_matrix)
    n = 1
    while 2**n < m:
        n += 1
    
    condensed_matrix = np.zeros((2**n, 2**n), dtype=original_matrix.dtype)
    condensed_matrix[:m, :m] = original_matrix
    return condensed_matrix

