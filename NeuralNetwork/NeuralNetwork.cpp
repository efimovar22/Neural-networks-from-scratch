#include "NetworkModel.h"

namespace NeuralNetwork {
    NetworkModel::NetworkModel(Sequential seq, std::unique_ptr<LossFunction> loss_function, double learning_rate)
    : learning_rate(learning_rate), sequential(std::move(seq)), loss_function(std::move(loss_function)) {
        predictions.resize(MNIST::DIGIT_COUNT);
        for (int32_t i = 0; i < MNIST::DIGIT_COUNT; ++i) {
            predictions[i] = MNIST::EncodeDigit(i);
        }
    }
    void NetworkModel::Train(const std::string& path_to_images, const std::string& path_to_labels, size_t batch_size, size_t epochs) {
        DataLoader data_loader(path_to_images, path_to_labels, batch_size);
        Batch batch;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            data_loader.LoadNextBatch(batch);
            while (!batch.empty()) {
                UpdateWeights(batch, learning_rate / (batch.size() * (epoch / 3 + 1)));
                data_loader.LoadNextBatch(batch);
            }
            data_loader.Restart();
        }
    }
    std::pair<size_t, size_t> NetworkModel::Predict(const std::string& path_to_images, const std::string& path_to_labels, size_t batch_size) {
        DataLoader data_loader(path_to_images, path_to_labels, batch_size);
        data_loader.Restart();
        size_t correct_predictions = 0;
        size_t total_images = 0;
        Batch batch;
        data_loader.LoadNextBatch(batch);
        while (!batch.empty()) {
            ComputeGradients(batch);
            for (auto &x_y : batch) {
                if (MNIST::DecodeVector(x_y.second) == MNIST::DecodeVector(x_y.first)) {
                    ++correct_predictions;
                }
            }
            total_images += batch.size();
            data_loader.LoadNextBatch(batch);
        }
        data_loader.Restart();
        return std::make_pair(correct_predictions, total_images);
    }
    void NetworkModel::UpdateWeights(Batch& batch, double adjusted_learning_rate) {
        sequential.Reset();
        ComputeGradients(batch);
        sequential.UpdateWeights(adjusted_learning_rate);
    }
    void NetworkModel::ComputeGradients(Batch &batch) {
        for (auto &x_y : batch) {
            sequential.ForwardPropagate(x_y.first);
        }
    }
    void NetworkModel::BackPropogate(const Batch &batch) {
        Vector derivative(batch[0].first.size());
        derivative.setZero();
        for (auto &x_y : batch) {
            derivative += loss_function->Derivative(x_y.second, x_y.first);
        }
        sequential.BackwardPropagate(derivative);
    }
} 
