import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def transition_matrix(A):
    degrees = np.sum(A, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.nan_to_num(A / degrees[:, None])
    return P

def random_walk_probabilities(A, start_node, steps):
    P = transition_matrix(A)
    num_nodes = A.shape[0]
    
    # Initialize probability vector
    prob = np.zeros(num_nodes)
    prob[start_node] = 1
    
    # Perform random walk
    for _ in range(steps):
        prob = np.dot(prob, P)
    
    return prob

def plot_histogram(probabilities):
    num_nodes = len(probabilities)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(num_nodes), probabilities, color='skyblue')
    plt.xlabel('Node')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Random Walk (Final Step)')
    plt.ylim(0, 1)
    
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{prob:.2f}', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.show()

def main():
    # Define the adjacency matrix
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0]
    ])
    
    # Define the starting node and number of steps
    start_node = 0
    steps = 10
    
    # Calculate the probabilities
    probabilities = random_walk_probabilities(A, start_node, steps)
    
    # Plot the histogram for the final step
    plot_histogram(probabilities)

if __name__ == "__main__":
    main()
