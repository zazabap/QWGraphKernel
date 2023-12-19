
################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-19.                            #
# Purpose: Create Simple Gates and Circuit     #
#          to practice the algorithms          #
################################################

from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# Define the Hamiltonian matrix (for example, a Pauli-X on qubit 0)
hamiltonian = np.array([[0, 1], [1, 0]])

# Define an initial state (for example, |0⟩)
initial_state = np.array([1, 0])  # Represents |0⟩ in computational basis

# Define the time duration for evolution
time = 1.0  # The time for evolution (arbitrary value)

# Create a quantum circuit representing the time evolution
qc = QuantumCircuit(1)
qc.unitary(np.exp(-1j * time * hamiltonian), [0])  # Applying the time evolution

# Simulate the circuit on a local simulator
backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
result = job.result()

# Get the final statevector after time evolution
final_state = result.get_statevector()
print("Initial State (|0⟩):", initial_state)
print("State after time evolution:", final_state)
