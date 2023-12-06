#include <iostream>
#include <Eigen/Dense>
#include <qpp/qpp.h>
#include "Encoding.h"
#include "Circuit.h"


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

cmat kron4(const cmat& A, const cmat& B, 
           const cmat& C, const cmat& D){
     return kron(kron(kron(A,B),C),D);
}

// 
// Implement the unitary CTQW on 4 qubits  
//
// ket CTQW_4qubits(const ket& state) {
//     cmat gate =     
//     ket evolved_state = gate*state;
//     return evolved_state;
// }


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



    // Define a quantum state (for example, |0⟩ state)
    psi = st.z0;

    std::cout << "Before \n";
    std::cout << disp(psi) << std::endl;

    // Perform the unitary evolution by applying the rotational gate using the function
    // ket evolved_state = applyRotation(Rz, psi);

    // Display the evolved state
    // std::cout << "Evolved state after applying rotational gate around Z-axis: \n";
    // std::cout << disp(evolved_state) << std::endl;

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

    // display the 4 qubits state
    auto ket0 = st.z0;
    auto psi_4 = kron(kron(kron(ket0, ket0), ket0), ket0);
    std::cout<<"Tensor Tensor Tensor Product\n";
    std::cout<<"\n"<<disp(psi_4)<<std::endl;

    // Create a Unitary Hadmard gate
    cmat H_4 = kron(kron(kron(gt.H, gt.H), gt.H), gt.H);
    // Create a Unitary X gate
    cmat X_4 = kron(kron(kron(gt.X, gt.X), gt.X ), gt.X);
    // Create a Control Gate 
    double angle = M_PI / 4; // Example: rotate by 45 degrees
    cmat Rz = gt.RZ(angle); // Rotational gate around Z-axis
    auto U_2 = gt.Id(2);
    cmat C_R =  kron4( U_2, U_2, U_2, Rz);
    std::cout<<disp(C_R)<<"\n";

    auto a = new Circuit();

    cmat C_Zz = a->controlledRz4q(angle);
    std::cout<<disp(C_Zz)<<"\n";


    // auto U_4 = X_4 * H_4*C_R;
    // std::cout<<disp(U_4)<<"\n";
    // std::cout<<disp(psi_4*U_4)<<"\n";
    // std::cout<<"trace"<<trace(psi_4*U_4)<<"\n";
    return 0;
}
