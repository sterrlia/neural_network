#include "Eigen/Dense"
#include "activation.hpp"
#include "util/random.hpp"
#include <vector>

struct DenseNeuralNetwork
{
    const ActivationFnInterface& activationFn;
    std::vector<std::vector<double>> biases;     // neuron biases
    std::vector<Eigen::MatrixXd> weightMatrixes; // weights between layers
};

DenseNeuralNetwork
    buildNetwork(const ActivationFnInterface& activationFn,
        const std::vector<size_t>& layerWidthList);

std::vector<double>
    forwardPass(const DenseNeuralNetwork& network,
        const std::vector<double>& input);

void
    backPropagation(DenseNeuralNetwork& network,
        const std::vector<double>& input,
        const std::vector<double>& predictedOutput,
        double learningRate);
