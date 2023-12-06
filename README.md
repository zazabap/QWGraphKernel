# Quantum Walk Graph Kernel Implementations

This repository contains implementations of Quantum Walk Graph Kernels in various programming languages/libraries.

## Overview

Quantum Walk Graph Kernels are computational tools used in graph analysis and machine learning. They leverage quantum walk principles to compute the similarity between graphs by capturing structural information.

This repository provides implementations of Quantum Walk Graph Kernels using different programming languages and libraries to showcase their usage and performance.

## Implementations

### Quantum++ (C++)

- `main.cpp`: C++ implementation using Quantum++ library.
- `unitary.cpp` : Circuit Representation for fixed time CTQW

### Qiskit (Python)

- `main.py`: Python implementation using Qiskit library.

## Usage

Each implementation comes with its own instructions and dependencies documented within the respective directory.

### Example (Quantum++ - C++)

1. Navigate to the `qpp_quantum_walk_kernel` directory.
2. Compile the code using your C++ compiler.
    ```bash
    g++ -std=c++11 -o qpp_quantum_walk_kernel qpp_quantum_walk_kernel.cpp -lqpp
    ```
3. Execute the compiled binary.
    ```bash
    ./qpp_quantum_walk_kernel
    ```

### Contribution

Contributions are welcome! If you'd like to add implementations in other languages/libraries or improve the existing ones, feel free to fork this repository and submit a pull request.

Please refer to the CONTRIBUTING.md for guidelines.

### License

This repository is licensed under the [MIT License](LICENSE).


