################################################    
# Author: SHiwen An.                           #
# Date: 2023-12-19.                            #
# Purpose: Create Simple Gates and Circuit     #
#          to practice the algorithms          #
################################################

import numpy as np
from qiskit import * 
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, Aer, execute
import random
import matplotlib.pyplot as plt
from qiskit.result import Counts




circ = QuantumCircuit(3)
# Add a H gate on qubit $q_{0}$, putting this qubit in superposition.
circ.h(0)
# Add a CX (CNOT) gate on control qubit $q_{0}$ and target qubit $q_{1}$, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit $q_{0}$ and target qubit $q_{2}$, putting
# the qubits in a GHZ state.
circ.cx(0, 2)

circ.draw('mpl')

circuit_drawer(circ, output='mpl', filename='HCXCX.png')  # 'mpl' for Matplotlib output
print(circ)

# Define a simple quantum circuit for implementation
# where its time evolution to be solidify
def CTQW(t):
    circCTQW = QuantumCircuit(4,4)
    circCTQW.h(0)
    circCTQW.h(1)
    circCTQW.h(2)
    circCTQW.h(3)
    circCTQW.x(0)
    circCTQW.x(1)
    circCTQW.x(2)
    circCTQW.x(3)
    
    theta = np.pi /2 *t
    circCTQW.mcrx(theta,[0,1,2],3)

    circCTQW.x(0)
    circCTQW.x(1)
    circCTQW.x(2)
    circCTQW.x(3)
    circCTQW.h(0)
    circCTQW.h(1)
    circCTQW.h(2)
    circCTQW.h(3)

    circCTQW.measure([0, 1, 2, 3], [0, 1, 2, 3])

    # circuit_drawer(circCTQW, output='mpl', filename='CTQW.png')  # 'mpl' for Matplotlib output
    # print(circCTQW)

    # Simulate the circuit on a local simulator
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circCTQW, backend, shots=128)  # 'shots' define the number of times the circuit is executed
    result = job.result()

    counts = result.get_counts(circCTQW)
    # print("\nMeasurement Outcomes:")
    # print(counts)
    return counts

double_list = [random.uniform(-1, 1) for _ in range(50)]
plt.figure(figsize=(10, 6))
for i in double_list:
    counts = CTQW(i)
    # print(counts)
    # print(type(counts))
    # print(counts.keys())
    # print(counts.values())

    outcomes = list(counts.keys())
    counts = list(counts.values())
    print(outcomes)
    print(counts)
    plt.bar(outcomes, counts, alpha=0.5)
    # plt.bar(outcomes, counts, alpha=0.5, label=f'Experiment {1+i}')

plt.xlabel('Outcomes')
plt.ylabel('Counts')
plt.title('Histograms of Measurement Outcomes for Multiple Experiments')
plt.legend()
plt.tight_layout()
plt.show()


# theta = np.pi/2

# # Create a quantum circuit with 4 qubits
# qc = QuantumCircuit(4)

# # Apply controlled Rx gate with 4th qubit as the control
# control_qubits = [0, 1, 2]  # Control qubits (first three qubits)
# target_qubit = 3  # Target qubit (fourth qubit)
# qc.mcrx(theta, control_qubits, target_qubit)

# matrix = Operator(qc).data
# formatted_matrix = np.vectorize(lambda x: np.around(x.real) + np.around(x.imag) * 1j)(matrix)
# trace = np.trace(formatted_matrix)

# # View the circuit
# print("Quantum Circuit:")
# circuit_drawer(qc, output='mpl', filename='Rxpi3.png')  # 'mpl' for Matplotlib output
# print(qc)
# print(formatted_matrix)
# print(trace)

