//////////////////////////////////////////////////////////////////
// Author: Shiwen An                                            //
// Date: 2023-12-18                                             //
// Purpose: Test Different CppTest Regime                       //
//          And Verify with simple circuis                      //
//        Reference on Pennylane:                               //
//   https://pennylane.ai/qml/glossary/quantum_embedding/       //
//////////////////////////////////////////////////////////////////


#include "CppTest.h"

CppTest::CppTest(int num, string str) {
    myNumber = num;
    myString = str;
}

void CppTest::setNumber(int num) {
    myNumber = num;
}

void CppTest::setString(string str) {
    myString = str;
}

void CppTest::display() {
    cout << "Number: " << myNumber << std::endl;
    cout << "String: " << myString << std::endl;
}

void CppTest::BasisCppTest(){
    cout << "Start Testing Basis CppTest" << std::endl;
    
    
}

void CppTest::QuantumCppTest(){
    cout << "Start Testing Quantum CppTest" << std::endl;

}


void CppTest::BlockCppTest(){
    cout << "Start Testing Quantum CppTest" << std::endl;
}

void CppTest::AmplitudeCppTest(){
    cout << "Start Testing Quantum CppTest" << std::endl;
}
