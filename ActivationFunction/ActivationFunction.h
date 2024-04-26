#pragma once
#include <iostream>

enum activationFunc {
    sigmoid = 1,
    ReLU,
    thx
};

class ActivationFunction {
public:
    void set();
    void use (double* value, int n);
    void useDer(double* value, int n);
    double useDer(double value);
private:
    activationFunc actFunc;
};