################################################
# Author: SHiwen An.                           #
# Date: 2023-12-23.                            #
# Purpose: Create Simple Gates and Circuit     #
#          to practice the algorithms          #
#          reverse engineering                 #
#                                              #
################################################

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import numpy as np
from qiskit.visualization import array_to_latex
import matplotlib.pyplot as plt


psi0 =  qi.Statevector([0,1,0,0])
qc = QuantumCircuit(2)
qc.h(1)
qc.cx(1,0)
print(qc)
psi1 = psi0.evolve(qc)
rho1 = qi.DensityMatrix(psi1)
print(rho1)
