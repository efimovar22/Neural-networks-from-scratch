#include "LossFunction.h"

namespace NeuralNetwork {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

[[nodiscard]] std::string MSE::functionName() const { return "MSE"; }
[[nodiscard]] double MSE::applyActivation(const Vector &target,
                                          const Vector &output) const {
  return (target - output).squaredNorm() / target.size();
}
[[nodiscard]] Vector MSE::computeJacobian(const Vector &target,
                                          const Vector &output) const {
  return (output - target) * 2;
}
} // namespace NeuralNetwork
