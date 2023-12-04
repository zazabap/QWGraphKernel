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
    cout << "Number: " << myNumber << endl;
    cout << "String: " << myString << endl;
}

