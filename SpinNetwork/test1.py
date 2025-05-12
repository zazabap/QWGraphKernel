from qutip import Qobj, bloch

# Function to plot Bloch sphere for SU(2) matrices
def plot_bloch_sphere(matrices, title):
    b = bloch.Bloch()
    for node, matrix in matrices.items():
        # Convert SU(2) matrix to a state vector (apply to |0>)
        state = Qobj(matrix) * Qobj([[1], [0]])
        b.add_states(state)
    b.show()
    plt.title(title)
    plt.show()

# Plot Bloch sphere for cycle graph
plot_bloch_sphere(matrices_cycle, "Bloch Sphere for Cycle Graph")

# Plot Bloch sphere for star graph
plot_bloch_sphere(matrices_star, "Bloch Sphere for Star Graph")