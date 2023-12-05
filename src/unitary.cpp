#include <iostream>
#include <Eigen/Dense>
#include <qpp/qpp.h>
#include "Encoding.h"

////////
// Author: Shiwen An 
// Date: 2023-12-5
// Purpose: Create unitary evolution of the quantum state
///////
using namespace qpp;

//
// Function for applying certain gate 
//
ket applyRotation(const cmat& gate, const ket& state) {
    ket evolved_state = gate*state;
    return evolved_state;
}

//
// Kronecker Delta 
//
cmat tensorProduct(const cmat& X, const cmat& Y){
    cmat tensor = kron(X,Y);
    return tensor;
}

int main() {

    std::cout << "Hello, World!" << std::endl;

    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::VectorXd vec(2);
    vec << 5, 6;
    Eigen::VectorXd result = mat * vec;
    std::cout << "Matrix:\n" << mat << "\nVector:\n" << vec << "\nResult:\n" << result << std::endl;

    std::cout<<"Quantum State Evolution over matrix"<<std::endl;
    ket psi = st.z0;
    cmat X = gt.X;
    std::cout << disp(psi) << std::endl;

    // unitary evolution by applying the gate
    psi = X * psi;

    std::cout << "Evolved state after applying the Pauli-X gate: \n";
    std::cout << disp(psi) << std::endl;

    // Define the angle for rotation (in radians)
    double angle = M_PI / 4; // Example: rotate by 45 degrees

    // Create the rotational gate around the Z-axis
    cmat Rz = gt.RZ(angle); // Rotational gate around Z-axis

    std::cout << "Rotation with PI/4 \n";
    std::cout << disp(Rz) << std::endl;

    // Define a quantum state (for example, |0⟩ state)
    psi = st.z0;

    std::cout << "Before \n";
    std::cout << disp(psi) << std::endl;

    // Perform the unitary evolution by applying the rotational gate using the function
    ket evolved_state = applyRotation(Rz, psi);

    // Display the evolved state
    std::cout << "Evolved state after applying rotational gate around Z-axis: \n";
    std::cout << disp(evolved_state) << std::endl;

    // Define two quantum gates (for example, Pauli-X and Pauli-Y)
    cmat Y = gt.Y; // Pauli-Y gate
    cmat Z = gt.Z; // Pauli-Y gate


    auto tensor = tensorProduct(X, Y);
    std::cout<<"X gate: \n";
    std::cout << disp(X) << std::endl;
    std::cout<<"Y gate: \n";
    std::cout << disp(Y) << std::endl;
    std::cout<<"Z gate: \n";
    std::cout << disp(Z) << std::endl;
    std::cout<<"X Y tensor: \n";
    std::cout << disp(tensor) << std::endl;

    return 0;
}
