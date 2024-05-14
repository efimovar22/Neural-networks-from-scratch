#pragma once

#include "../eigen/Eigen/Dense"
#include "MNIST.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace NeuralNetwork {
class DataLoader {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Batch = std::vector<std::pair<Vector, Vector>>;
public:
  DataLoader(const std::string &image_path, const std::string &label_path,
             size_t size_of_batch);

private:
  class Model;
  void LoadNextBatch(Batch &batch);
  void Restart();
  Eigen::Vector<double, MNIST::IMAGE_SIZE> ExtractImage();
  uint8_t ExtractLabel();
  std::ifstream image_stream;
  std::ifstream label_stream;
  size_t batch_size;
  size_t image_size;
  size_t current_index = 0;
  uint32_t total_images;
};
} // namespace NeuralNetwork
