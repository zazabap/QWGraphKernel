
################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-19.                            #
# Purpose: Measurement Example                 #
################################################


from qiskit import QuantumCircuit, Aer, execute

# Create a quantum circuit with 2 qubits and 2 classical bits for measurement
qc = QuantumCircuit(2, 2)

# Apply some quantum gates
qc.h(0)  # Hadamard gate on qubit 0
qc.cx(0, 1)  # CNOT gate with qubit 0 as control and qubit 1 as target

# Measure qubits 0 and 1 and store results in classical bits 0 and 1
qc.measure([0, 1], [0, 1])

# View the circuit
print("Quantum Circuit:")
print(qc)

# Simulate the circuit on a local simulator
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1024)  # 'shots' define the number of times the circuit is executed
result = job.result()

# Get the measurement outcomes
counts = result.get_counts(qc)
print("\nMeasurement Outcomes:")
print(counts)


