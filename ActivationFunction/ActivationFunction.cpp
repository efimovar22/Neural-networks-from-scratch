#include "ActivationFunction.h"

void ActivationFunction::set() {
    std::cout << "Choose actFunc pls\n1 - sigmoid\n2 - ReLU\n3 - th(x)\n";
    int tmp;
    std::cin >> tmp;
    switch (tmp) {
    case sigmoid:
        actFunc = sigmoid;
        break;
    case ReLU:
        actFunc = ReLU;
        break;
    case thx:
        actFunc = thx;
        break;
    default:
        throw std::runtime_error("Error read actFunc");
        break;
    }
}

void ActivationFunction::use(double* value, int n) {
    switch (actFunc) {
    case activationFunc::sigmoid:
        for (int i = 0; i < n; ++i) {
            value[i] = 1 / (1 + exp(-value[i]));
        }
        break;
    case activationFunc::ReLU:
        for (int i = 0; i < n; ++i) {
            if (value[i] < 0) {
                value[i] *= 0.01;
            } else if (value[i] > 1) {
                value[i] = 1. + 0.01 * (value[i] - 1.);
            }
        }
        break;
    case activationFunc::thx:
        for (int i = 0; i < n; ++i) {
            if (value[i] < 0) {
                value[i] = 0.01 * (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
            }
        }
    }
}

void ActivationFunction::useDer(double* value, int n) {
    switch (actFunc) {
    case activationFunc::sigmoid:
        for (int i = 0; i < n; ++i) {
            value[i] = value[i] * (1 - value[i]);
        }
        break;
    case activationFunc::ReLU:
        for (int i = 0; i < n; ++i) {
            if (value[i] < 0 || value[i] > 1) {
                value[i] = 0.01;
            } else {
                value[i] = 1;
            }
        }
        break;
    case activationFunc::thx:
        for (int i = 0; i < n; ++i) {
            if (value[i] < 0) {
                value[i] = 0.01 * (1 - value[i] * value[i]);
            } else {
                value[i] = 1 - value[i] * value[i];
            }
        }
        break;
    default:
        throw std::runtime_error("Error actFuncDer\n");
        break;
    }
}

double ActivationFunction::useDer(double value) {
	switch (actFunc)
	{
	case activationFunc::sigmoid:
		value = 1 / (1 + exp(-value));
		break;
	case activationFunc::ReLU:
		if (value < 0 || value > 1)
			value = 0.01;
		break;
	case activationFunc::thx:
		if (value < 0) 
			value = 0.01 * (exp(value) - exp(-value)) / (exp(value) + exp(-value));
		else 
			value = (exp(value) - exp(-value)) / (exp(value) + exp(-value));
		break;

	default:
		throw std::runtime_error("Error actFunc \n");
		break;
	}
	return value;
}