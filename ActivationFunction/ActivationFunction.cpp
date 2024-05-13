#include "ActivationFunction.h"

namespace NeuralNetwork {
	[[nodiscard]] std::string Sigmoid::functionName() const {
        return "Sigmoid";
    }
    [[nodiscard]] Vector Sigmoid::applyActivation(const Vector &x) const {
        return 1 / ((-x.array()).exp() + 1);
    }
    [[nodiscard]] Matrix Sigmoid::computeJacobian(const Vector &x) const {
        return ((-x.array()).exp() / pow(1.0 + (-x.array()).exp(), 2)).matrix().asDiagonal();
    }

	[[nodiscard]] std::string ReLu::functionName() const {
        return "ReLu";
    }
    [[nodiscard]] Vector ReLu::applyActivation(const Vector &x) const {
        return x.cwiseMax(0.0);
    }
    [[nodiscard]] Matrix ReLu::computeJacobian(const Vector &x) const {
        return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
    }

	[[nodiscard]] std::string Softmax::functionName() const {
        return "Softmax";
    }
    [[nodiscard]] Vector Softmax::applyActivation(const Vector &x) const {
        return x.array().exp() / result.sum();
    }
    [[nodiscard]] Matrix Softmax::computeJacobian(const Vector &x) const {
        return Compute(x).asDiagonal() - Compute(x) * Compute(x).transpose();
    }
} 