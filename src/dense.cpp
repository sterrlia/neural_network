#include "Eigen/Dense"
#include "dense.hpp"
#include "util/random.hpp"
#include <vector>

DenseNeuralNetwork
    buildNetwork(const ActivationFnInterface& activationFn,
        const std::vector<size_t>& layerWidthList)
{
    size_t layerCount = layerWidthList.size();
    std::vector<Eigen::MatrixXd> weightMatrixes(layerCount - 1);
    std::vector<std::vector<double>> biases(layerCount - 1);
    auto random = Random();

    size_t previousLayerNeuronCount = layerWidthList[0];

    for (size_t j = 0; j < layerCount - 1; ++j)
    {
        size_t layerNeuronCount = layerWidthList[j + 1];
        biases[j] = std::vector<double>(layerNeuronCount);

        Eigen::MatrixXd weightMatrix(layerNeuronCount, previousLayerNeuronCount);
        // Xavier (Glorot) init for weights
        double weightLimit = std::sqrt(6.0 / (previousLayerNeuronCount + layerNeuronCount));
        for (size_t n = 0; n < layerNeuronCount; ++n)
        {
            for (size_t i = 0; i < previousLayerNeuronCount; ++i)
            {
                weightMatrix(n, i) = random.getDouble(-weightLimit, weightLimit);
            }

            biases[j][n] = random.getDouble(-0.01, 0.01);
        }
        previousLayerNeuronCount = layerNeuronCount;

        weightMatrixes[j] = weightMatrix;
    }

    return DenseNeuralNetwork{ activationFn, biases, weightMatrixes };
}

std::vector<double>
    forwardPass(const DenseNeuralNetwork& network,
        const std::vector<double>& input)
{
    std::vector<double> previousLayerValues = input;
    std::vector<double> currentLayerValues;

    for (size_t j = 0; j < network.biases.size(); ++j)
    {
        auto layerBiases = network.biases[j];
        size_t layerNeuronCount = layerBiases.size();
        currentLayerValues = std::vector<double>(layerNeuronCount);
        for (size_t n = 0; n < layerNeuronCount; ++n)
        {
            double bias = layerBiases[n];
            double value = 0;
            for (size_t i = 0; i < previousLayerValues.size(); ++i)
            {
                auto weight = network.weightMatrixes[j](n, i);
                value += previousLayerValues[i] * weight;
            }
            value += bias;

            auto activatedValue = network.activationFn.invoke(value);
            currentLayerValues[n] = activatedValue;
        }

        previousLayerValues = currentLayerValues;
    }

    return currentLayerValues;
}

void
    backPropagation(DenseNeuralNetwork& network,
        const std::vector<double>& input,
        const std::vector<double>& predictedOutput,
        double learningRate)
{
    size_t layerCount = network.biases.size();

    std::vector<std::vector<double>> activatedValues(layerCount + 1);

    // Computing values
    // I push input so I would able to get input layer neuron count
    activatedValues[0] = input;
    for (size_t j = 0; j < layerCount; ++j)
    {
        auto layerBiases = network.biases[j];
        size_t layerNeuronCount = layerBiases.size();

        activatedValues[j + 1] = std::vector<double>(layerNeuronCount);

        size_t previousLayerNeuronCount = activatedValues[j].size();
        auto layerWeightMatrix =
            network.weightMatrixes[j]; // Matrix of weights from j+1 to j

        for (size_t n = 0; n < layerNeuronCount; ++n)
        {
            double bias = layerBiases[n];
            double value = 0;
            for (size_t i = 0; i < previousLayerNeuronCount; ++i)
            {
                double weight = layerWeightMatrix(n, i);
                value += activatedValues[j][i] * weight;
            }

            value += bias;
            double activatedValue = network.activationFn.invoke(value);
            activatedValues[j + 1][n] = activatedValue;
        }
    }

    std::vector<std::vector<double>> gradients(layerCount);

    // Computing output layer gradient (gradient array direction is reverse)
    std::vector<double> actualOutput =
        activatedValues[layerCount]; // Cause activatedValues have one row more
                                     // (from input)

    size_t outputLayerNeuronCount = network.biases[layerCount - 1].size();
    gradients[layerCount - 1] = std::vector<double>(outputLayerNeuronCount);
    for (size_t i = 0; i < outputLayerNeuronCount; ++i)
    {
        double currentNeuronDerivativeValue =
            network.activationFn.getDerivativeResultByInvokeOutput(activatedValues[layerCount][i]);

        double diff = predictedOutput[i] - actualOutput[i];
        double outputLayerGradient = -2 * diff * currentNeuronDerivativeValue;

        gradients[layerCount - 1][i] = outputLayerGradient;
    }

    // Computing gradients for hidden layers
    for (size_t j = layerCount - 1; j-- > 0;)
    {
        auto layerBiases = network.biases[j];
        double layerNeuronCount = layerBiases.size();
        auto previousLayerGradients = gradients[j + 1];
        size_t previousLayerNeuronCount = network.biases[j + 1].size();
        // Cause weights are connected from next layer to previous (backwards),
        // we need to get matrix from previous layer
        auto layerWeightMatrix = network.weightMatrixes[j + 1];
        gradients[j] = std::vector<double>(layerNeuronCount);

        for (size_t n = 0; n < layerNeuronCount; ++n)
        {
            double gradient = 0;

            for (size_t i = 0; i < previousLayerNeuronCount; ++i)
            {
                double weight = layerWeightMatrix(
                    i, n); // now we swap i and n arguments to
                           // because we go backwards (it would fail if not)

                double previousLayerGradient = previousLayerGradients[i];

                gradient += weight * previousLayerGradient;
            }

            double activatedValue =
                network.activationFn.getDerivativeResultByInvokeOutput(activatedValues[j + 1][n]);

            gradient *= activatedValue;

            gradients[j][n] = gradient;
        }
    }

    // Applying gradients to weights and biases
    for (size_t j = 0; j < layerCount; ++j)
    {
        auto& layerBiases = network.biases[j];
        double layerNeuronCount = layerBiases.size();
        size_t previousLayerNeuronCount = activatedValues[j].size();
        auto& layerWeightMatrix = network.weightMatrixes[j];

        for (size_t n = 0; n < layerNeuronCount; ++n)
        {
            double gradient = gradients[j][n];

            for (size_t i = 0; i < previousLayerNeuronCount; ++i)
            {
                double previousLayerActivatedValue =
                    activatedValues[j][i]; // Cause activatedValues have one row
                                           // more

                layerWeightMatrix(n, i) -=
                    gradient * previousLayerActivatedValue * learningRate;
            }

            layerBiases[n] -= gradient * learningRate;
        }
    }
}
