#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class ActivationFunction {
public:
    friend class NetModel;
    friend class LayeredModel;
private:
	[[nodiscard]] virtual std::string GefunctionNametType() const = 0;
    [[nodiscard]] virtual Vector applyActivation(const Vector &x) const = 0;
    [[nodiscard]] virtual Matrix computeJacobian(const Vector &x) const = 0;
};

class Sigmoid final : public ActivationFunction {
private:
	[[nodiscard]] std::string functionName() const final;
    [[nodiscard]] Vector applyActivation(const Vector &x) const final;
    [[nodiscard]] Matrix computeJacobian(const Vector &x) const final;
};

class ReLu final : public ActivationFunction {
private:
	[[nodiscard]] std::string functionName() const final;
    [[nodiscard]] Vector applyActivation(const Vector &x) const final;
    [[nodiscard]] Matrix computeJacobian(const Vector &x) const final;
};

class Softmax : public ActivationFunction {
private:
	[[nodiscard]] std::string functionName() const final;
    [[nodiscard]] Vector applyActivation(const Vector &x) const final;
    [[nodiscard]] Matrix computeJacobian(const Vector &x) const final;
};
}
