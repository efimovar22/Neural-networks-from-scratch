project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(SOURCES
    main.cpp
    ActivationFunction/ActivationFunction.cpp
    Matrix/Matrix.cpp
    NetWork/NetWork.cpp
)
