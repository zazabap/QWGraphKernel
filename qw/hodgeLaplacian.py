import numpy as np

# Define the matrices \bbB_1^T and \bbB_2
B1_T = np.array([
    [-1,  1,  0,  0,  0],
    [ 1,  0, -1,  0,  0],
    [ 0, -1,  1,  0,  0],
    [ 0, -1,  0,  1,  0],
    [ 0,  0,  1, -1,  0],
    [ 0, -1,  0,  0,  1],
    [ 0,  0,  1,  0, -1]
])

B2 = np.array([
    [ 1,  0,  0],
    [ 1,  0,  0],
    [ 1, -1, -1],
    [ 0,  1,  0],
    [ 0,  1,  0],
    [ 0,  0,  1],
    [ 0,  0,  1]
])

# Compute \bbB_1 from \bbB_1^T
B1 = B1_T.T

# Compute \bbL_1 = \bbB_1^T \bbB_1 + \bbB_2 \bbB_2^T
L1 = B1_T @ B1 + B2 @ B2.T

print("Matrix L1:")
print(L1)

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define the matrix A
A = np.array([[0,1,1,0,0],
              [1,0,1,1,1], 
              [1,1,0,1,1], 
              [0,1,1,0,0], 
              [0,1,1,0,0]])
A = np.array([[-2,1,1,0,0],
              [1,-4,1,1,1], 
              [1,1,-4,1,1], 
              [0,1,1,-2,0], 
              [0,1,1,0,-2]])

A = L1
# Define the initial vector v
v = np.array([1,0,0,0,0,0,0])
# v = np.array([0,0,1,0,0])

# compute psi(7)
exp_At = expm( 1j * A * 7)
v_t = np.dot(exp_At, v)
print(v_t)
v = np.array([0,0,0,0,0,1,0])
# n_e = np.dot(v_t, np.conjugate(v_t)) 
n_e = np.dot( v, v_t)
r = np.dot( np.conjugate(n_e), n_e)
print(r)

# Define a range of time parameters
times = np.linspace(0, 10, 2000)

# Lists to store norm-square values
norms_initial = []
norms_evolved = []

for t in times:
    # Compute the matrix exponential e^(At)
    exp_At = expm( 1j * A * t)
    
    # Evolve the vector v in time
    v = np.array([1,0,0,0,0,0,0])
    v_t = np.dot(exp_At, v)

    # Compute norm-square
    norm_initial = np.dot(v, v)
    v = np.array([0,0,0,0,0,1,0])
    prob = np.dot(v, v_t)
    norm_evolved = np.dot( np.conjugate(prob), prob)
    
    norms_initial.append(norm_initial)
    norms_evolved.append(norm_evolved)

# Print the result for a specific time (e.g., t = 1.0)
# t = 1.0
# exp_At = expm(1j* A * t)
# v_t = np.dot(exp_At, v)
# norm_initial = np.dot(v, v)
# norm_evolved = np.dot(v_t, v_t)
# print(f"At time t = {t}:")
# print(f"Initial vector norm-square: {norm_initial}")
# print(f"Evolved vector norm-square: {norm_evolved}")

# Visualization
fig, ax = plt.subplots()
ax.plot(times, norms_initial, label='Initial vector norm-square', linestyle='--', color='blue')
ax.plot(times, norms_evolved, label='Evolved vector norm-square', linestyle='-', color='red')

ax.set_xlabel('Time')
ax.set_ylabel('Probability(<psi_0|psi_t>)')
ax.set_title('Norm-Square of Vectors Over Time')
ax.legend()
ax.grid()

plt.show()
