################################################
# Author: SHiwen An.                           #
# Date: 2024-04-28.                            #
# Purpose: Hodge Laplacian for the model       #
################################################

import networkx as nx
import pandas as pd
import numpy as np
import linecache
import matplotlib.pyplot as plt
from collections import defaultdict


# A function converting 
# graph structures to numpy matrix
def adjacencyMatrices(file_A, file_node, file_node_label, file_graph_labels):
    """
    Print the direction of each edge in a directed graph.

    Parameters:
    - G: NetworkX directed graph

    Returns:
    - A_list 
    """
    data = pd.read_csv(file_A, header=None, names=['nodes', 'edges'])
    lines = []
    with open(file_graph_labels, 'r') as file:
        # Read lines and strip whitespace
        lines = [line.strip() for line in file]
    y = []
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
            y.append(lines[node_pre-1])
            node_pre = int(linecache.getline(file_node, node1))
            A_list.append(nx.adjacency_matrix(G).todense())
            G = nx.Graph() # use Digraph / graph for the laplacian options. 
            G.add_edge(node1, node2)
    
        node_attr = {node1: {'label': linecache.getline(file_node_label, node1),  
                    node2: {'label': linecache.getline(file_node_label, node2)}}}
        nx.set_node_attributes(G, node_attr)


    return A_list, y


# Define the laplacian matrix 
# first get the degree matrix-adjacency matrix
def LaplacianMatrices(file_A, file_node, file_node_label, file_graph_labels):
    data = pd.read_csv(file_A, header=None, names=['nodes', 'edges'])
    lines = []
    with open(file_graph_labels, 'r') as file:
        # Read lines and strip whitespace
        lines = [line.strip() for line in file]
    y = []
    # Create a graph
    G = nx.Graph()
    L_list = []
    node_pre = 1
    # Add edges to the graph based on the adjacency matrix
    for _, row in data.iterrows():
        node1, node2 = row['nodes'], row['edges']
        if int(linecache.getline(file_node, node1)) == node_pre :
            G.add_edge(node1, node2)
        else: 
            y.append(lines[node_pre-1])
            node_pre = int(linecache.getline(file_node, node1))
            L_list.append(nx.laplacian_matrix(G).toarray())
            G = nx.Graph()
            G.add_edge(node1, node2)
    
        node_attr = {node1: {'label': linecache.getline(file_node_label, node1),  
                    node2: {'label': linecache.getline(file_node_label, node2)}}}
        nx.set_node_attributes(G, node_attr)

    return L_list, y


# Define the laplacian matrix 
# first get the degree matrix-adjacency matrix
def HLMatrices(file_A, file_node, file_node_label, file_graph_labels):
    data = pd.read_csv(file_A, header=None, names=['nodes', 'edges'])
    lines = []
    with open(file_graph_labels, 'r') as file:
        # Read lines and strip whitespace
        lines = [line.strip() for line in file]
    y = []
    # Create a graph
    G = nx.DiGraph()
    HL_list = []
    node_pre = 1
    # Add edges to the graph based on the adjacency matrix
    for _, row in data.iterrows():
        node1, node2 = row['nodes'], row['edges']
        if int(linecache.getline(file_node, node1)) == node_pre :
            G.add_edge(node1, node2)
        else: 
            y.append(lines[node_pre-1])
            node_pre = int(linecache.getline(file_node, node1))
            HL = graphHelmholtzian(G)
            HL_list.append(HL)
            G = nx.DiGraph()
            G.add_edge(node1, node2)
    
        node_attr = {node1: {'label': linecache.getline(file_node_label, node1),  
                    node2: {'label': linecache.getline(file_node_label, node2)}}}
        nx.set_node_attributes(G, node_attr)

    # Calculate the eigenvalue of the matrices
    return HL_list, y


def graphHelmholtzian(G):
    # I is the incidence matrix 
    I = nx.incidence_matrix(G, oriented=True).toarray()
    # edge same as number of rows
    A = I.T
    num_edges = G.number_of_edges()

    edge_values = np.zeros(num_edges)

    # Find all triangles in the directed graph
    triangles = []
    for u in G.nodes():
        for v in G.successors(u):
            for w in G.successors(v):
                if G.has_edge(w, u):
                    triangles.append((u, v, w))

    ee = G.edges()
    # Iterate over each triangle
    for triangle in triangles:
        u, v, w = triangle
        # Determine the clockwise orientation of the triangle
        clockwise = (u, v) in ee and (v, w) in ee and (w, u) in ee
        # Assign edge values based on their direction relative to the clockwise orientation
        for edge_idx, edge in enumerate(ee):
            if edge in [(u, v), (v, w), (w, u)]:
                edge_values[edge_idx] = -1 if clockwise else 1
    

    H = A @ A.transpose() + edge_values @ edge_values.T

    # Debug purpose
    # print(A @ A.transpose())
    # print(edge_values @ edge_values.T)
    # print(H)

    return H 
