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

    // Angle for rotation
    double angle = M_PI / 4; // Example: rotate by 45 degrees
    

    return 0;
}
