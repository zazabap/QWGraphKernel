//////////////////////////////////////////////////////////////////
// Author: Shiwen An                                            //
// Date: 2023-12-9                                              //
// Purpose: Test Different Encoding Regime                      //
//          And Verify with simple circuis                      //
//        Reference on Pennylane:                               //
//   https://pennylane.ai/qml/glossary/quantum_embedding/       //
//////////////////////////////////////////////////////////////////


#include "Encoding.h"

Encoding::Encoding(int num, string str) {
    myNumber = num;
    myString = str;
}

void Encoding::setNumber(int num) {
    myNumber = num;
}

void Encoding::setString(string str) {
    myString = str;
}

void Encoding::display() {
    cout << "Number: " << myNumber << std::endl;
    cout << "String: " << myString << std::endl;
}

void Encoding::BasisEncoding(){
    cout << "Start Testing Basis Encoding" << std::endl;
    
    
}

void Encoding::QuantumEncoding(){
    cout << "Start Testing Quantum Encoding" << std::endl;

}



void Encoding::BlockEncoding(){
    cout << "Start Testing Quantum Encoding" << std::endl;
}

void Encoding::AmplitudeEncoding(){
    cout << "Start Testing Quantum Encoding" << std::endl;
}
