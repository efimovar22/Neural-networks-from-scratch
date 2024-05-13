#pragma once

#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include "../LossFunction/LossFunction.h"
#include "../DataLoader/DataLoader.h"
#include "../Layer/Layer.h"

namespace NeuralNetwork {
class NetworkModel {
public:
    NetworkModel(Seq seq, std::unique_ptr<LossFunction> loss_function, double learning_rate);
    void Train(const std::string& path_to_images, const std::string& path_to_labels, size_t batch_size, size_t epoch);
    std::pair<size_t, size_t> Predict(const std::string& path_to_images, const std::string& path_to_labels, size_t batch_size);
    friend std::ostream &operator<<(std::ostream &os, const NetworkModel &model) {
        os << "NetworkModel: Layers - " << model.sequential.number_of_layers 
           << ", Loss Function - " << model.loss_function->GetType() 
           << ", Learning Rate - " << model.learning_rate << "\n";
        os << model.sequential << "\n";
        return os;
    }
    friend std::istream &operator>>(std::istream &is, NetworkModel &model) {
        double lr;
        Seq seq;
        is >> seq >> lr;
        model = NetworkModel(std::move(seq), std::make_unique<MSE>(), lr);
        return is;
    }
private:
    void UpdateWeights(const Batch &batch, double learning_rate_epoch);
    void ComputeGradients(const Batch &batch);
    double learning_rate;
    Seq sequential;
    std::unique_ptr<LossFunction> loss_function;
    std::vector<Vector> predictions;
};
} 
