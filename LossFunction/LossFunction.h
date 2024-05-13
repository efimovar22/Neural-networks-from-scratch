#pragma once

#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class LossFunction {
public:
    friend class Model;
private:
    [[nodiscard]] virtual std::string functionName() const = 0;
    [[nodiscard]] virtual double applyActivation(const Vector &target, const Vector &output) const = 0;
    [[nodiscard]] virtual Vector computeJacobian(const Vector &target, const Vector &output) const = 0;
};

class MSE final : public LossFunction {
private:
    [[nodiscard]] std::string functionName() const final;
    [[nodiscard]] double applyActivation(const Vector &target, const Vector &output) const final;
    [[nodiscard]] Vector computeJacobian(const Vector &target, const Vector &output) const final;
};
}
