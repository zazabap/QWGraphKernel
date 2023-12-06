///////////////////////////////////////
// Author: Shiwen An 
// Date: 2023-12-5
// Purpose: Create unitary evolution 
//          of the quantum state
////////////////////////////////////////

#include <iostream>
#include <Eigen/Dense>
#include <qpp/qpp.h>
#include "Encoding.h"
#include "Circuit.h"


using namespace qpp;

//
// Function for applying certain gate 
//
ket applyRotation(const cmat& gate, const ket& state) {
    ket evolved_state = gate*state;
    return evolved_state;
}

// 
// Implement the unitary CTQW on 4 qubits  
//
// ket CTQW_4qubits(const ket& state) {
//     cmat gate =     
//     ket evolved_state = gate*state;
//     return evolved_state;
// }

// Save some example for matrix eigen3
void eigenExample(){
    std::cout << "Hello, World!" << std::endl;
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::VectorXd vec(2);
    vec << 5, 6;
    Eigen::VectorXd result = mat * vec;
    std::cout << "Matrix:\n" << mat << "\nVector:\n" << vec << "\nResult:\n" << result << std::endl;
}

int main() {
    std::cout<<"Quantum State Evolution over matrix"<<std::endl;
    ket psi = st.z0;
    cmat X = gt.X;
    std::cout << disp(psi) << std::endl;

    // unitary evolution by applying the gate
    psi = X * psi;

    std::cout << "Evolved state after applying the Pauli-X gate: \n";
    std::cout << disp(psi) << std::endl;

    // Define a quantum state (for example, |0⟩ state)
    psi = st.z0;

    std::cout << "Before \n";
    std::cout << disp(psi) << std::endl;
    auto a = new Circuit();

    // Perform the unitary evolution by applying the rotational gate using the function
    // ket evolved_state = applyRotation(Rz, psi);

    // Display the evolved state
    // std::cout << "Evolved state after applying rotational gate around Z-axis: \n";
    // std::cout << disp(evolved_state) << std::endl;

    // Create a Unitary Hadmard gate
    cmat H_4 = a->kron4(gt.H, gt.H, gt.H, gt.H);
    cmat X_4 = a->kron4(gt.X, gt.X, gt.X, gt.X);
    double angle = M_PI / 4; // Example: rotate by 45 degrees
    cmat Rz = gt.RZ(angle); // Rotational gate around Z-axis
    auto U_2 = gt.Id(2);
    cmat C_R = a->controlledRz4q(angle);
    std::cout<<disp(C_R)<<"\n";
    auto CC = H_4 * X_4 * C_R*X_4*H_4;

    // Measurement based on the C_Zz C_R above
    auto measured = measure(CC, gt.Z, {0});
    std::cout << ">> Measurement result: " << std::get<RES>(measured) << '\n';
    std::cout << ">> Probabilities: ";
    std::cout << disp(std::get<PROB>(measured), ", ") << '\n';
    std::cout << ">> Resulting states:\n";
    for (auto&& it : std::get<ST>(measured))
        std::cout << disp(it) << "\n\n";

    // Initialize the quantum circuit
    QCircuit circuit;

    // Add qubits to the circuit
    circuit.add_qudit(); // Add qubit 0
    circuit.add_qudit(); // Add qubit 1

    // Apply Hadamard gates to qubits
    circuit.gate(gt.H, 0); // Apply Hadamard gate to qubit 0
    circuit.gate(gt.H, 1); // Apply Hadamard gate to qubit 1

    circuit.gate(Rz, 0); // Apply Hadamard gate to qubit 0
    circuit.gate(Rz, 1); // Apply Hadamard gate to qubit 1
    
    circuit.gate(gt.CNOT, 0,1);

    auto cc = circuit.get_gate_depth();

    QEngine qe(circuit);
    qe.execute();

    // Display the resulting state
    std::cout << "Resulting state after applying Hadamard gates to both qubits:\n";
    std::cout << disp(cc) << std::endl;
    std::cout << circuit.get_resources() <<std::endl;
    qpp::idx nq_c = 2 ;
    std::vector<idx> counting_qubits(nq_c);
    delete a;
    return 0;
}
