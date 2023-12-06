// Quantum phase estimation circuit simulator
// Source: ./examples/circuits/qpe_circuit.cpp
// See also ./examples/qpe.cpp for a low-level API example
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include <qpp/qpp.h>

int main() {
    using namespace qpp;

    idx nq_c = 4;         // number of counting qubits
    idx nq_a = 2;         // number of ancilla qubits
    idx nq = nq_c + nq_a; // total number of qubits

    std::cout << ">> Quantum phase estimation quantum circuit simulation\n";
    std::cout << ">> nq_c = " << nq_c << " counting qubits, nq_a = " << nq_a
              << " ancilla qubits\n\n";

    realT theta = 0.25; // change if you want, increase nq_c for more precision
    cmat U(2, 2);       // initialize a unitary operator
    U << 1, 0, 0, std::exp(pi * 1_i * theta); // T gate
    // we use the T\otimes T gate as an example; we want to estimate its last
    // (4-th) eigenvalue; we expect estimated theta = 1/4 (0.25).
    U = kron(U, U); // OK, size will re-adjust since U is a dynamic Eigen matrix

    QCircuit qc{nq, nq_c};
    std::vector<idx> counting_qubits(nq_c);
    std::iota(counting_qubits.begin(), counting_qubits.end(), 0);
    std::vector<idx> ancilla(nq_a);
    std::iota(ancilla.begin(), ancilla.end(), nq_c);

    qc.gate_fan(gt.H, counting_qubits);
    qc.gate_fan(gt.X, ancilla); // prepare |11>, the fourth eigenvector of U
    for (idx i = nq_c; i-- > 0;) {
        qc.CTRL(U, i, ancilla);
        U = powm(U, 2);
    }
    qc.TFQ(counting_qubits); // inverse Fourier transform
    // measure many qubits at once, store starting with the 0 classical dit
    qc.measure(counting_qubits);

    // display the quantum circuit and its corresponding resources
    std::cout << qc << "\n\n" << qc.get_resources() << "\n\n";

    std::cout << ">> Running...\n";
    QEngine engine{qc};
    engine.execute();
    // decimal representation of the measurement result
    idx decimal = multiidx2n(engine.get_dits(), std::vector<idx>(nq_c, 2));
    auto theta_e =
        static_cast<realT>(decimal) / static_cast<realT>(std::pow(2, nq_c));

    std::cout << ">> Input theta = " << theta << '\n';
    std::cout << ">> Estimated theta = " << theta_e << '\n';
    std::cout << ">> Norm difference: " << std::abs(theta_e - theta) << '\n';

    // Another example for measurement
    ket psi = 00_ket;
    U = gt.CNOT * kron(gt.H, gt.Id2);
    ket result = U * psi; // we have the Bell state (|00> + |11>) / sqrt(2)

    std::cout << ">> We just produced the Bell state:\n";
    std::cout << disp(result) << '\n';

    // apply a bit flip on the second qubit
    result = apply(result, gt.X, {1}); // we produced (|01> + |10>) / sqrt(2)
    std::cout << ">> We produced the Bell state:\n";
    std::cout << disp(result) << '\n';

    // measure the first qubit in the X basis
    auto measured = measure(result, gt.H, {0});
    std::cout << ">> Measurement result: " << std::get<RES>(measured) << '\n';
    std::cout << ">> Probabilities: ";
    std::cout << disp(std::get<PROB>(measured), ", ") << '\n';
    std::cout << ">> Resulting states:\n";
    for (auto&& it : std::get<ST>(measured))
        std::cout << disp(it) << "\n\n";
}
