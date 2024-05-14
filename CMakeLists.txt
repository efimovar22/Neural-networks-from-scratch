cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} main.cpp ActivationFunction/ActivationFunction.cpp DataLoader/DataLoader.cpp Layer/Layer.cpp LossFunction/LossFunction.cpp MNIST/MNIST.cpp NeuralNetwork/NeuralNetwork.cpp)

include_directories(${PROJECT_SOURCE_DIR}/ActivationFunction DataLoader Layer LossFunction MNIST NeuralNetwork)
