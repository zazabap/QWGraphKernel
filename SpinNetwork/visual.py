import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

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

# Function to visualize Bloch vectors for all graphs
def visualize_bloch_vectors(matrices_cycle, matrices_star, matrices_random):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Bloch sphere surface
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
    
    # Plot cycle graph nodes
    print("Cycle Graph Bloch Vectors:")
    for node in matrices_cycle:
        x, y, z = get_bloch_vector(matrices_cycle[node])
        print(f"Node {node}: ({x:.2f}, {y:.2f}, {z:.2f})")
        ax.scatter([x], [y], [z], c='blue', s=100, marker='o', alpha=0.6, 
                   label='Cycle Graph' if node == 0 else "")
    
    # Plot star graph nodes
    print("\nStar Graph Bloch Vectors:")
    for node in matrices_star:
        x, y, z = get_bloch_vector(matrices_star[node])
        print(f"Node {node}: ({x:.2f}, {y:.2f}, {z:.2f})")
        ax.scatter([x], [y], [z], c='red', s=150, marker='^', alpha=0.6, 
                   label='Star Graph' if node == 0 else "")
    
    # Plot random graph nodes
    print("\nRandom Graph Bloch Vectors:")
    for node in matrices_random:
        x, y, z = get_bloch_vector(matrices_random[node])
        print(f"Node {node}: ({x:.2f}, {y:.2f}, {z:.2f})")
        ax.scatter([x], [y], [z], c='green', s=120, marker='s', alpha=0.6, 
                   label='Random Graph' if node == 0 else "")
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere: SU(2) Matrices from Cycle, Star, and Random Graphs')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.show()

# Function to visualize parameters (theta, phi, psi) in 3D
def visualize_parameters(theta_cycle, phi_cycle, psi_cycle, 
                        theta_star, phi_star, psi_star, 
                        theta_random, phi_random, psi_random):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cycle graph nodes
    for node in theta_cycle:
        ax.scatter([theta_cycle[node]], [phi_cycle[node]], [psi_cycle[node]], 
                   c='blue', s=100, marker='o', alpha=0.6, 
                   label='Cycle Graph' if node == 0 else "")
    
    # Plot star graph nodes
    for node in theta_star:
        ax.scatter([theta_star[node]], [phi_star[node]], [psi_star[node]], 
                   c='red', s=150, marker='^', alpha=0.6, 
                   label='Star Graph' if node == 0 else "")
    
    # Plot random graph nodes
    for node in theta_random:
        ax.scatter([theta_random[node]], [phi_random[node]], [psi_random[node]], 
                   c='green', s=120, marker='s', alpha=0.6, 
                   label='Random Graph' if node == 0 else "")
    
    # Set labels and title
    ax.set_xlabel('theta (radians)')
    ax.set_ylabel('phi (radians)')
    ax.set_zlabel('psi (radians)')
    ax.set_title('Parameter Space: (theta, phi, psi) for Cycle, Star, and Random Graphs')
    ax.legend()
    
    plt.show()

def plot_all_matrices_with_pca(matrices_cycle, matrices_star, matrices_random):
    """
    Plots matrices from cycle, star, and random graphs on the same figure using PCA.

    Parameters:
        matrices_cycle (dict): SU(2) matrices for the cycle graph.
        matrices_star (dict): SU(2) matrices for the star graph.
        matrices_random (dict): SU(2) matrices for the random graph.
    """
    # Combine all matrices into a single dataset
    all_matrices = []
    labels = []
    
    def normalize_matrix(matrix):
        """Normalize a complex matrix by dividing by its Frobenius norm."""
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm != 0 else matrix
    
    def flatten_normalized_matrix(matrix):
        """Flatten a normalized complex matrix into a real-valued vector (real and imaginary parts)."""
        normalized = normalize_matrix(matrix)
        return np.concatenate([normalized.real.flatten(), normalized.imag.flatten()])
    
    for matrix in matrices_cycle.values():
        all_matrices.append(flatten_normalized_matrix(matrix))
        labels.append('Cycle')
    
    for matrix in matrices_star.values():
        all_matrices.append(flatten_normalized_matrix(matrix))
        labels.append('Star')
    
    for matrix in matrices_random.values():
        all_matrices.append(flatten_normalized_matrix(matrix))
        labels.append('Random')
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(all_matrices)
    
    # Extract x and y coordinates
    x_coords = reduced_data[:, 0]
    y_coords = reduced_data[:, 1]
    
    # Plot the reduced data
    plt.figure(figsize=(10, 8))
    
    # Plot each group with a different color
    for label, color in zip(['Cycle', 'Star', 'Random'], ['blue', 'red', 'green']):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(x_coords[indices], y_coords[indices], label=label, alpha=0.7, edgecolor='k')
    
    # Add labels and title
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Normalized SU(2) Matrices: Cycle, Star, and Random Graphs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_star_and_random_with_pca(matrices_star, matrices_random):
    """
    Plots matrices from star and random graphs on the same figure using PCA.

    Parameters:
        matrices_star (dict): SU(2) matrices for the star graph.
        matrices_random (dict): SU(2) matrices for the random graph.
    """
    # Combine matrices into a single dataset
    all_matrices = []
    labels = []
    
    def normalize_matrix(matrix):
        """Normalize a complex matrix by dividing by its Frobenius norm."""
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm != 0 else matrix
    
    def flatten_normalized_matrix(matrix):
        """Flatten a normalized complex matrix into a real-valued vector (real and imaginary parts)."""
        normalized = normalize_matrix(matrix)
        return np.concatenate([normalized.real.flatten(), normalized.imag.flatten()])
    
    for matrix in matrices_star.values():
        all_matrices.append(flatten_normalized_matrix(matrix))
        labels.append('Star')
    
    for matrix in matrices_random.values():
        all_matrices.append(flatten_normalized_matrix(matrix))
        labels.append('Random')
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(all_matrices)
    
    # Extract x and y coordinates
    x_coords = reduced_data[:, 0]
    y_coords = reduced_data[:, 1]
    
    # Plot the reduced data
    plt.figure(figsize=(10, 8))
    
    # Plot each group with a different color
    for label, color in zip(['Star', 'Random'], ['red', 'green']):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(x_coords[indices], y_coords[indices], label=label, alpha=0.7, edgecolor='k')
    
    # Add labels and title
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Normalized SU(2) Matrices: Star and Random Graphs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_random_with_pca_3d(matrices_random):
    """
    Visualizes matrices from the random graph in 3D using PCA.

    Parameters:
        matrices_random (dict): SU(2) matrices for the random graph.
    """
    # Combine matrices into a single dataset
    all_matrices = []
    
    def normalize_matrix(matrix):
        """Normalize a complex matrix by dividing by its Frobenius norm."""
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm != 0 else matrix
    
    def flatten_normalized_matrix(matrix):
        """Flatten a normalized complex matrix into a real-valued vector (real and imaginary parts)."""
        normalized = normalize_matrix(matrix)
        return np.concatenate([normalized.real.flatten(), normalized.imag.flatten()])
    
    for matrix in matrices_random:
        all_matrices.append(flatten_normalized_matrix(matrix))
    
    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(all_matrices)
    
    # Extract x, y, and z coordinates
    x_coords = reduced_data[:, 0]
    y_coords = reduced_data[:, 1]
    z_coords = reduced_data[:, 2]
    
    # Plot the reduced data in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c='green', alpha=0.7, edgecolor='k', s=50)
    
    # Add labels and title
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D PCA of Normalized SU(2) Matrices: Random Graph')
    plt.show()

def plot_three_matrix_lists_with_pca_3d(matrices_list1, matrices_list2, matrices_list3, labels=('List1', 'List2', 'List3'), colors=('blue', 'red', 'green')):
    """
    Visualizes three lists of 2x2 complex matrices in 3D using PCA, each with a different color.

    Parameters:
        matrices_list1 (list): First list of 2x2 complex matrices.
        matrices_list2 (list): Second list of 2x2 complex matrices.
        matrices_list3 (list): Third list of 2x2 complex matrices.
        labels (tuple): Labels for the three lists.
        colors (tuple): Colors for the three lists.
    """
    all_matrices = []
    all_labels = []

    def normalize_matrix(matrix):
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm != 0 else matrix

    def flatten_normalized_matrix(matrix):
        normalized = normalize_matrix(matrix)
        return np.concatenate([normalized.real.flatten(), normalized.imag.flatten()])

    for matrix in matrices_list1:
        all_matrices.append(flatten_normalized_matrix(matrix))
        all_labels.append(labels[0])
    for matrix in matrices_list2:
        all_matrices.append(flatten_normalized_matrix(matrix))
        all_labels.append(labels[1])
    for matrix in matrices_list3:
        all_matrices.append(flatten_normalized_matrix(matrix))
        all_labels.append(labels[2])

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(all_matrices)
    x_coords = reduced_data[:, 0]
    y_coords = reduced_data[:, 1]
    z_coords = reduced_data[:, 2]

    # Plot the reduced data in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, label in enumerate(labels):
        indices = [i for i, l in enumerate(all_labels) if l == label]
        ax.scatter(x_coords[indices], y_coords[indices], z_coords[indices],
                   c=colors[idx], alpha=0.7, edgecolor='k', s=50, label=label)

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D PCA of Three Matrix Lists')
    ax.legend()
    plt.show()