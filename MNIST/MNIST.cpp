#include "MNIST.h"

namespace MNIST {

Vector EncodeDigit(int32_t digit) {
    Vector encoded = Vector::Zero(DIGIT_COUNT);
    encoded[digit] = 1;
    return encoded;
}

int32_t DecodeVector(const Vector &encoded_vector) {
    int32_t max_index = 0;
    for (int32_t i = 0; i < DIGIT_COUNT; ++i) {
        if (encoded_vector[i] > encoded_vector[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}
}
