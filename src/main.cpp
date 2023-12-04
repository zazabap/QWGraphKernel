#include <iostream>
#include <Eigen/Dense>
#include <matplot/matplot.h>
#include <qpp/qpp.h>

using namespace matplot;

int main() {
    using namespace qpp;

    std::cout << "Hello, World!" << std::endl;

    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::VectorXd vec(2);
    vec << 5, 6;
    Eigen::VectorXd result = mat * vec;
    std::cout << "Matrix:\n" << mat << "\nVector:\n" << vec << "\nResult:\n" << result << std::endl;

    // std::vector<double> x = {1, 2, 3, 4, 5};
    // std::vector<double> y = {1, 4, 9, 16, 25};

    // matplot::plot(x, y);
    // matplot::show();

    std::vector<double> x = iota(0, 10, 100);
    std::vector<double> y = {20, 30, 45, 40, 60, 65, 80, 75, 95, 90};
    std::vector<double> err(y.size(), 10.);
    errorbar(x, y, err);
    axis({0, 100, 0, 110});
    show();

    // Create a quantum circuit with 3 qubits
    ket psi = mket({0, 0, 0});

    QCircuit qc{1, 1, 2, "coin flip"};
    qc.gate(gt.H, 0);
    qc.measure_all();

    std::cout << qc << "\n\n" << qc.get_resources() << "\n\n";
    std::cout << QEngine{qc}.execute(100) << "\n";

    // Apply a Hadamard gate on the first qubit
    // psi = hadamard(psi, {0});

    // Display the state of the quantum system
    std::cout << disp(psi) << std::endl;
    // return 0;
    return 0;
}
