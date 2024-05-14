#include "ActivationFunction/ActivationFunction.h"
#include "NeuralNetwrok/NeuralNetwrok.h"
#include <memory>
#include <vector>

constexpr int num_epochs = 10;
constexpr int pix_h = 784;
constexpr int pix_w = 128;
constexpr int EPOCHS = 1; 
constexpr int size_batch = 1;
constexpr int size_batch_2 = 30;

int main() {
  using Vector = Eigen::VectorXd;
  using Matrix = Eigen::MatrixXd;
  using namespace NeuralNetwork;
  try {
    nn::run_all_tests();
  } catch(...) {
    except::react();
  }
  return 0;
  }
  std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
  activation_functions.push_back(std::make_unique<ReLu>());
  activation_functions.push_back(std::make_unique<Softmax>());
  NetworkModel model({{pix_h, pix_w, num_epochs}, std::move(activation_functions)},
                     std::make_unique<MSE>(), 0.05);
  for (int i = 0; i < NUM_EPOCHS; ++i) {
    model.Train("train/train-images.idx3-ubyte", "train/train-labels.idx1-ubyte", BATCH_SIZE, EPOCHS);
    auto a = model.Predict("test/t10k-images.idx3-ubyte",
                           "test/t10k-labels.idx1-ubyte", BATCH_SIZE_2);
    std::cout << a.first * 1.0 / a.second << std::endl;
  }
  std::ofstream file("out.txt");
  file << model;
  return 0;
}
