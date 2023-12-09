#include "Circuit.h"

// Constructor
Circuit::Circuit() {
    // Initialize
}

// Destructor
Circuit::~Circuit() {
    // Cleanup
}

// Function to add gates to the circuit
void Circuit::addGate(const cmat& gate, std::vector<idx> qubits) {
    gates_.push_back(gate);
    qubit_indices_.push_back(qubits);
}

// Function to apply the circuit to a quantum state
ket Circuit::apply(const ket& state) {
    ket result = state;
    for (idx i = 0; i < gates_.size(); ++i) {
        cmat gate = gates_[i];
        std::vector<idx> qubits = qubit_indices_[i];
        // Apply gate to specified qubits
        result = qpp::apply(result, {gate}, {qubits});
    }
    return result;
}

// Experimental function
cmat Circuit::controlledRz4q(double angle) {
    // Create the rotational matrix around the Z-axis
    cmat Rz = gt.RZ(angle); // Rotational matrix around Z-axis

    // Define the controlled-Rz gate for 4 qubits with the 4th qubit as control
    cmat controlled_Rz(16, 16); // Controlled-Rz gate for 4 qubits
    controlled_Rz.setIdentity(); // Initialize to identity

    // Set the controlled-Rz gate based on the control state of the 4th qubit
    controlled_Rz(14,14) = Rz(0, 0);
    controlled_Rz(14,15) = Rz(0, 1);
    controlled_Rz(15,14) = Rz(1, 0);
    controlled_Rz(15,15) = Rz(1, 1);

    return controlled_Rz;
}

cmat Circuit::kron4(const cmat& A, const cmat& B, 
           const cmat& C, const cmat& D){
     return kron(kron(kron(A,B),C),D);
}

double QJSK(const ket& rho, const ket& sigma){
    
}

