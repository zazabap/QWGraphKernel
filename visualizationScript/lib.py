################################################
# Author: SHiwen An.                           #
# Date: 2023-12-23.                            #
# Purpose: Quick Visualization for graphs      #
#          different type of graphs            #
################################################

import networkx as nx
import pandas as pd
import linecache
import matplotlib.pyplot as plt

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

# A function converting 
# graph structures to numpy matrix
def adjacencyMatrices(file_A, file_node, file_node_label):
    data = pd.read_csv(file_A, header=None, names=['nodes', 'edges'])

    # Create a graph
    G = nx.Graph()
    A_list = []
    node_pre = 1
    # Add edges to the graph based on the adjacency matrix
    for _, row in data.iterrows():
        node1, node2 = row['nodes'], row['edges']
        if int(linecache.getline(file_node, node1)) == node_pre :
            G.add_edge(node1, node2)
        else: 
            node_pre = int(linecache.getline(file_node, node1))
            A_list.append(nx.adjacency_matrix(G).todense())
            G = nx.Graph()
            G.add_edge(node1, node2)
    
        node_attr = {node1: {'label': linecache.getline(file_node_label, node1),  
                    node2: {'label': linecache.getline(file_node_label, node2)}}}
        nx.set_node_attributes(G, node_attr)

    for i in range(3):
        print(A_list[i])
    return A_list