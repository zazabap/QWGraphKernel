
################################################
# Author: SHiwen An.                           #
# Date: 2023-12-23.                            #
# Purpose: Create Simple Gates and Circuit     #
#          to practice the algorithms          #
#          reverse engineering                 #
################################################

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import circuit_drawer
import matplotlib

# Create a quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cnot(0, 1)

# Transpile the circuit for the unitary simulator
qc = transpile(qc, basis_gates=['u', 'cx'])

# Create a simulator backend
simulator = Aer.get_backend('unitary_simulator')

# Convert the circuit to a matrix
job = assemble(qc)
result = simulator.run(job).result()
unitary_matrix = result.get_unitary()

print("Unitary matrix:")
print(unitary_matrix)


# Create a quantum circuit with two qubits
bell_circuit = QuantumCircuit(2, 2)


# Apply a Hadamard gate to the first qubit
bell_circuit.h(0)

# Apply a CNOT gate where qubit 0 is the control and qubit 1 is the target
bell_circuit.cx(0, 1)

# Measure both qubits
bell_circuit.measure([0, 1], [0, 1])

# Visualize the circuit
print("Bell State Circuit:")
print(bell_circuit)

circuit_drawer(qc, output='mpl', filename='quantum_circuit.png')  # 'mpl' for Matplotlib output


