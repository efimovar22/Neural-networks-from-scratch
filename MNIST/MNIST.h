#pragma once

#include "../eigen/Eigen/Dense"

using Vector = Eigen::VectorXd;

namespace MNIST {
constexpr int32_t IMAGE_SIZE = 784; 
constexpr int32_t DIGIT_COUNT = 10; 
constexpr std::streamsize POINTER_SIZE = 4; 
constexpr double MAX_PIXEL_VALUE = 255.0; 
constexpr int32_t IMAGE_MAGIC_NUMBER = 0x00000803; 
constexpr int32_t LABEL_MAGIC_NUMBER = 2049;

static Vector EncodeDigit(int32_t digit);
static int32_t DecodeVector(const Vector &encoded_vector);
}
