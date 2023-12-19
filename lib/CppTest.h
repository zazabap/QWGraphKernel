#ifndef CPPTEST_H
#define CPPTEST_H

#include <iostream>
#include <string>
#include <qpp/qpp.h>

using namespace std;

class CppTest {
private:
    int myNumber;
    string myString;

public:
    CppTest(int num, string str);
    void setNumber(int num);
    void setString(string str);
    void display();
    void QuantumCppTest();
    void AmplitudeCppTest();
    void BasisCppTest();
    void BlockCppTest();
};

#endif

