#ifndef ENCODING_H
#define ENCODING_H

#include <iostream>
#include <string>
#include <qpp/qpp.h>

using namespace std;

class Encoding {
private:
    int myNumber;
    string myString;

public:
    Encoding(int num, string str);
    void setNumber(int num);
    void setString(string str);
    void display();
    void QuantumEncoding();
    void AmplitudeEncoding();
    void BasisEncoding();
    void BlockEncoding();
};

#endif

