#pragma once

#include "../ActivationFunction/ActivationFunction.h"
#include "../MNIST/MNIST.h"
#include "../eigen/Eigen/Dense"
#include <vector>

namespace NeuralNetwork {
struct Layer {
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

public:
  Layer() = default;
  Layer(long width, long height);
  Layer(Matrix &weight_matrix, Vector &bias_vector);
  friend std::ostream &operator<<(std::ostream &output, const Layer &layer);
  friend std::istream &operator>>(std::istream &input, Layer &layer);
  [[nodiscard]] Vector Activate(const Vector &input) const;
  friend class Seq;

private:
  void Adjust(double rate, const Matrix &weight_updates,
              const Vector &bias_updates);
  Matrix weights;
  Vector biases;
};

class Seq {
public:
  Seq(std::initializer_list<size_t> dimensions,
      std::vector<std::unique_ptr<ActivationFunction>> &&activations);
  Seq(std::vector<Layer> layers,
      std::vector<std::unique_ptr<ActivationFunction>> &&activations);

private:
  friend class Model;
  void Forward(Vector &input);
  void Backward(Vector &input);
  void UpdateWeights(double learning_rate);
  void Reinitialize();
  friend std::ostream &operator<<(std::ostream &output, const Seq &sequence);
  friend std::istream &operator>>(std::istream &input, Seq &sequence);
  size_t layer_count = 0;
  std::vector<Matrix> gradient_w;
  std::vector<Vector> gradient_b;
  std::vector<Vector> inputs_saved;
  std::vector<Vector> outputs_saved;
  std::vector<Layer> layers;
  std::vector<std::unique_ptr<ActivationFunction>> activations;
};

} // namespace NeuralNetwork
