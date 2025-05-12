import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
def visualize_graph(G, theta, phi, psi, title, layout='spring'):
    # Set up layout
    if layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize phi for colormap (0 to 1)
    phi_values = [phi[node] / (2 * np.pi) for node in G.nodes()]
    colors = cm.hsv(phi_values)
    
    # Node sizes based on theta
    node_sizes = [300 + 1500 * theta[node] / np.pi for node in G.nodes()]
    
    # Draw graph
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors, node_size=node_sizes, 
            font_color='white', font_weight='bold', edge_color='gray')
    
    # Add colorbar for phi
    sm = plt.cm.ScalarMappable(cmap=cm.hsv, norm=plt.Normalize(vmin=0, vmax=2*np.pi))
    plt.colorbar(sm, ax=ax, label='phi (radians)')
    
    # Add title
    plt.title(title)
    plt.show()

# Function to compute Bloch vector from SU(2) matrix
def get_bloch_vector(matrix):
    # Apply matrix to |0> state
    state = matrix @ np.array([1, 0])
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Compute expectation values
    x = np.real(np.conj(state) @ sigma_x @ state)
    y = np.real(np.conj(state) @ sigma_y @ state)
    z = np.real(np.conj(state) @ sigma_z @ state)
    
    return x, y, z

# Function to visualize Bloch vectors for both graphs
def visualize_bloch_vectors(matrices_cycle, matrices_star):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Bloch sphere surface
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
    
    # Plot cycle graph nodes
    for node in matrices_cycle:
        x, y, z = get_bloch_vector(matrices_cycle[node])
        ax.scatter([x], [y], [z], c='blue', s=100, label='Cycle Graph' if node == 0 else "")
    
    # Plot star graph nodes
    for node in matrices_star:
        x, y, z = get_bloch_vector(matrices_star[node])
        ax.scatter([x], [y], [z], c='red', s=100, label='Star Graph' if node == 0 else "")
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere: SU(2) Matrices from Cycle and Star Graphs')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.show()

# Example 1: Cycle Graph (10 nodes)
G_cycle = nx.cycle_graph(10)
theta_cycle, phi_cycle, psi_cycle = assign_parameters(G_cycle)

# Compute matrices for cycle graph
matrices_cycle = {node: su2_matrix(theta_cycle[node], phi_cycle[node], psi_cycle[node]) 
                 for node in G_cycle.nodes()}

# Print parameters for cycle graph
print("Cycle Graph (10 Nodes) Parameters:")
for node in G_cycle.nodes():
    print(f"Node {node}: theta={theta_cycle[node]:.2f}, phi={phi_cycle[node]:.2f}, psi={psi_cycle[node]:.2f}")
    print(f"Matrix U_{node}:\n{matrices_cycle[node]}\n")

# Visualize cycle graph
visualize_graph(G_cycle, theta_cycle, phi_cycle, psi_cycle, "Cycle Graph (10 Nodes)", layout='circular')

# Example 2: Star Graph (10 nodes)
G_star = nx.star_graph(9)  # Central node 0, peripheral nodes 1 to 9
theta_star, phi_star, psi_star = assign_parameters(G_star)

# Compute matrices for star graph
matrices_star = {node: su2_matrix(theta_star[node], phi_star[node], psi_star[node]) 
                 for node in G_star.nodes()}

# Print parameters for star graph
print("Star Graph (10 Nodes) Parameters:")
for node in G_star.nodes():
    print(f"Node {node}: theta={theta_star[node]:.2f}, phi={phi_star[node]:.2f}, psi={psi_star[node]:.2f}")
    print(f"Matrix U_{node}:\n{matrices_star[node]}\n")

# Visualize star graph
visualize_graph(G_star, theta_star, phi_star, psi_star, "Star Graph (10 Nodes)", layout='spring')

# New Experiment: Visualize SU(2) matrix elements on Bloch sphere
visualize_bloch_vectors(matrices_cycle, matrices_star)