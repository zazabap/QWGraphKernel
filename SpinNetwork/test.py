import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Function to create SU(2) matrix given theta, phi, psi
def su2_matrix(theta, phi, psi):
    return np.array([
        [np.cos(theta) * np.exp(1j * phi), np.sin(theta) * np.exp(1j * psi)],
        [-np.sin(theta) * np.exp(-1j * psi), np.cos(theta) * np.exp(-1j * phi)]
    ])

# Function to assign parameters based on Strategy 1
def assign_parameters(G):
    n = G.number_of_nodes()
    max_degree = max(dict(G.degree()).values())
    
    # Initialize parameters
    theta = {}
    phi = {}
    psi = {}
    
    # Compute sum of neighbor degrees for psi
    neighbor_degree_sums = {}
    for node in G.nodes():
        neighbor_degree_sums[node] = sum(G.degree(neighbor) for neighbor in G.neighbors(node))
    max_neighbor_degree_sum = max(neighbor_degree_sums.values()) if neighbor_degree_sums else 1
    
    # Assign parameters
    for i, node in enumerate(G.nodes(), 1):
        degree = G.degree(node)
        theta[node] = np.pi * degree / max_degree if max_degree > 0 else np.pi / 2
        phi[node] = 2 * np.pi * (i - 1) / n
        psi[node] = 2 * np.pi * neighbor_degree_sums[node] / max_neighbor_degree_sum if max_neighbor_degree_sum > 0 else 0
    
    return theta, phi, psi

# Function to visualize the graph
def visualize_graph(G, theta, phi, psi, title):
    # Set up layout
    pos = nx.spring_layout(G)  # or nx.circular_layout(G) for cycle graph
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Normalize phi for colormap (0 to 1)
    phi_values = [phi[node] / (2 * np.pi) for node in G.nodes()]
    colors = cm.hsv(phi_values)
    
    # Node sizes based on theta
    node_sizes = [500 + 1000 * theta[node] / np.pi for node in G.nodes()]
    
    # Draw graph
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=node_sizes, 
            font_color='white', font_weight='bold', edge_color='gray')
    
    # Add title
    plt.title(title)
    plt.show()

# Example 1: Cycle Graph (4 nodes)
G_cycle = nx.cycle_graph(10)
theta_cycle, phi_cycle, psi_cycle = assign_parameters(G_cycle)

# Compute matrices for cycle graph
matrices_cycle = {node: su2_matrix(theta_cycle[node], phi_cycle[node], psi_cycle[node]) 
                 for node in G_cycle.nodes()}

# Print parameters and matrices for cycle graph
print("Cycle Graph Parameters:")
for node in G_cycle.nodes():
    print(f"Node {node}: theta={theta_cycle[node]:.2f}, phi={phi_cycle[node]:.2f}, psi={psi_cycle[node]:.2f}")
    print(f"Matrix U_{node}:\n{matrices_cycle[node]}\n")

# Visualize cycle graph
visualize_graph(G_cycle, theta_cycle, phi_cycle, psi_cycle, "Cycle Graph (10 Nodes)")

# Example 2: Star Graph (4 nodes)
G_star = nx.star_graph(9)  # Central node 0, peripheral nodes 1, 2, 3
theta_star, phi_star, psi_star = assign_parameters(G_star)

# Compute matrices for star graph
matrices_star = {node: su2_matrix(theta_star[node], phi_star[node], psi_star[node]) 
                 for node in G_star.nodes()}

# Print parameters and matrices for star graph
print("Star Graph Parameters:")
for node in G_star.nodes():
    print(f"Node {node}: theta={theta_star[node]:.2f}, phi={phi_star[node]:.2f}, psi={psi_star[node]:.2f}")
    print(f"Matrix U_{node}:\n{matrices_star[node]}\n")

# Visualize star graph
visualize_graph(G_star, theta_star, phi_star, psi_star, "Star Graph (10 Nodes)")