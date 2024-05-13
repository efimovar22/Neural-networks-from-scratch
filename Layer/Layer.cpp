#include "Layer.h"

namespace NeuralNetwork {
    Layer::Layer(long width, long height) {
        assert(width > 0 && height > 0);
        weights = Matrix::Random(width, height);
        biases = Vector::Random(width);
    }
    Layer::Layer(Matrix &weight_matrix, Vector &bias_vector) {
        weights = weight_matrix;
        biases = bias_vector;
    }
    friend std::ostream &operator<<(std::ostream &output, const Layer &layer) {
        output << "  Weights " << layer.weights.rows() << " " << layer.weights.cols() << "\n";
        output << layer.weights << "\n";
        output << "  Biases " << layer.biases.size() << "\n";
        output << layer.biases;
        return output;
    }
    friend std::istream &operator>>(std::istream &input, Layer &layer) {
        std::string label;
        long rows, cols;
        input >> label >> rows >> cols;
        assert(rows > 0 && cols > 0);
        layer.weights.resize(rows, cols);
        for (long row = 0; row < rows; row++) {
            for (long col = 0; col < cols; col++) {
                input >> layer.weights(row, col);
            }
        }
        input >> label >> rows;
        layer.biases.resize(rows);
        for (long idx = 0; idx < rows; idx++) {
            input >> layer.biases(idx);
        }
        return input;
    }
    [[nodiscard]] Vector Layer::Activate(const Vector &input) const {
        return weights * input + biases;
    }
    void Layer::Adjust(double rate, const Matrix &weight_updates, const Vector &bias_updates) {
        weights -= weight_updates * rate;
        biases -= bias_updates * rate;
    }
    Seq::Seq(std::initializer_list<size_t> dimensions, std::vector<std::unique_ptr<ActivationFunction>> &&activations)
        : activation_functions(std::move(activations)) {
        assert((dimensions.size() == activation_functions.size() + 1));
        assert(!activation_functions.empty());
        assert(*dimensions.begin() == MNIST::IMAGE_SIZE);
        layers.reserve(activation_functions.size());
        for (auto it = dimensions.begin(); it + 1 != dimensions.end(); ++it) {
            layers.emplace_back(*(it + 1), *(it));
        }
        assert(layers.back().biases.size() == MNIST::COUNT_OF_DIGITS);

        layer_count = activation_functions.size();
        gradient_w.resize(layer_count);
        gradient_b.resize(layer_count);
        inputs_saved.resize(layer_count);
        outputs_saved.resize(layer_count);

        for (size_t i = 0; i < layer_count; ++i) {
            gradient_w[i] = layers[i].weights;
            gradient_b[i] = layers[i].biases;
        }
        Reinitialize();
    }
    Seq::Seq(std::vector<Layer> layers_t, std::vector<std::unique_ptr<ActivationFunction>> &&activations)
        : layers(std::move(layers_t)), activation_functions(std::move(activations)) {
        layer_count = activation_functions.size();
        gradient_w.resize(layer_count);
        gradient_b.resize(layer_count);
        for (size_t i = 0; i < layer_count; ++i) {
            gradient_w[i] = layers[i].weights;
            gradient_b[i] = layers[i].biases;
        }
        Reinitialize();
        inputs_saved.resize(layer_count);
        outputs_saved.resize(layer_count);
    }
    void Seq::Forward(Vector &input) {
        for (size_t i = 0; i < layer_count; ++i) {
            inputs_saved[i] = input;
            outputs_saved[i] = activation_functions[i]->Activate(layers[i].Activate(input));
            input = outputs_saved[i];
        }
    }
    void Seq::Backward(Vector &input) {
        for (int32_t i = static_cast<int>(layer_count) - 1; i >= 0; --i) {
            Vector activated_deriv = activation_functions[i]->Derivative(outputs_saved[i]) * input;
            gradient_w[i] += activated_deriv * inputs_saved[i].transpose();
            gradient_b[i] += activated_deriv;
            input = layers[i].weights.transpose() * activated_deriv;
        }
    }
    void Seq::UpdateWeights(double learning_rate) {
        for (size_t i = 0; i < layer_count; ++i) {
            layers[i].Adjust(learning_rate, gradient_w[i], gradient_b[i]);
        }
    }
    void Seq::Reinitialize() {
        for (size_t i = 0; i < layer_count; ++i) {
            gradient_w[i].setZero();
            gradient_b[i].setZero();
        }
    }
    friend std::ostream &operator<<(std::ostream &output, const Seq &sequence) {
        for (size_t i = 0; i < sequence.layers.size(); i++) {
            output << i * 2 + 1 << " Layer\n";
            output << sequence.layers[i] << "\n";
            output << "\n" << i * 2 + 2 << " " << sequence.activation_functions[i]->Type() << "\n";
        }
        return output;
    }
    friend std::istream &operator>>(std::istream &input, Seq &sequence) {
        size_t layer_count;
        input >> layer_count;
        layer_count /= 2;
        std::vector<Layer> layers_t(layer_count);
        std::vector<std::unique_ptr<ActivationFunction>> activation_functions_t;
        activation_functions_t.reserve(layer_count);
        for (size_t i = 0; i < layer_count * 2; i++) {
            size_t index;
            std::string layer_name;
            input >> index >> layer_name;
            if (layer_name == "Layer") {
                input >> layers_t[i / 2];
            } else {
                activation_functions_t.push_back(ActivationFunctionFactory::Create(layer_name));
            }
        }
        sequence = Seq(layers_t, std::move(activation_functions_t));
        return input;
    }
}
