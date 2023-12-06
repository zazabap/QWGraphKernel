#ifndef CIRCUIT_H
#define CIRCUIT_H

#include <qpp/qpp.h>

using namespace qpp;

class Circuit {
public:
    // Constructor
    Circuit();

    // Destructor
    ~Circuit();

    // Function to add gates to the circuit
    void addGate(const cmat& gate, std::vector<idx> qubits);

    // Function to apply the circuit to a quantum state
    ket apply(const ket& state);

    // Function to use 4 qubits in Calculation
    cmat controlledRz4q(double angle);

    cmat kron4(const cmat& A, const cmat& B, 
           const cmat& C, const cmat& D);

private:
    std::vector<cmat> gates_;
    std::vector<std::vector<idx>> qubit_indices_;
};

#endif // CIRCUIT_H

