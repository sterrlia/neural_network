#include "dataset.hpp"
#include "lib.hpp"
#include <iostream>
#include <vector>

void
    testDataSet(const DenseNeuralNetwork& network,
        const std::vector<DataSample> dataset)
{
    size_t sampleCount = dataset.size();
    for (size_t i = 0; i < sampleCount; ++i)
    {
        auto sample = dataset[i];

        std::cout << std::endl;
        std::cout << "Input: " << std::endl;
        printVector(sample.inputs);

        std::cout << std::endl;
        std::cout << "Expected outputs: " << std::endl;
        printVector(sample.outputs);

        auto actual = forwardPass(network, sample.inputs);

        std::cout << std::endl;
        std::cout << "Actual outputs: " << std::endl;
        printVector(actual);
    }
}

void
    testDataSet(const DenseNeuralNetwork& network,
        const std::vector<DataSample> dataset,
        size_t caseCount)
{
    auto random = Random();
    size_t sampleCount = dataset.size();
    for (size_t i = 0; i < caseCount; ++i)
    {
        auto sample = dataset[random.getInt(0, sampleCount)];

        std::cout << std::endl;
        std::cout << "Input: " << std::endl;
        printVector(sample.inputs);

        std::cout << std::endl;
        std::cout << "Expected outputs: " << std::endl;
        printVector(sample.outputs);

        auto actual = forwardPass(network, sample.inputs);

        std::cout << std::endl;
        std::cout << "Actual outputs: " << std::endl;
        printVector(actual);
    }
}

void
    constantTest()
{
    std::cout << std::endl;
    std::cout << "=== Constant test ===" << std::endl;

    auto activationFn = SigmoidActivationFn();
    std::vector<size_t> layerWidths = { 5, 40, 30, 20, 10 };
    size_t layerCount = layerWidths.size();
    auto network = buildNetwork(activationFn, layerWidths);
    auto random = Random();

    std::vector<double> input = random.getDoubleRange(-5.0, 5.0, layerCount);
    std::cout << std::endl;
    std::cout << "Input: " << std::endl;

    printVector(input);

    auto result = forwardPass(network, input);

    std::cout << std::endl;
    std::cout << "Result: " << std::endl;

    printVector(result);

    std::cout << std::endl;
    std::cout << "Expected result: " << std::endl;
    auto output = std::vector<double>(layerWidths[layerCount - 1], 1);
    printVector(output);

    std::cout << std::endl;
    std::cout << "Training..." << std::endl;

    size_t epochsCount = 100;
    for (size_t i = 0; i < epochsCount; i++)
    {
        if (i % 10 == 0)
        {
            std::cout << epochsCount - i << " iterations left" << std::endl;
        }

        input = random.getDoubleRange(-5.0, 5.0, layerCount);

        double learningRate = 0.2;
        backPropagation(network, input, output, learningRate);
    }

    for (size_t i = 0; i < 5; i++)
    {
        input = random.getDoubleRange(-5.0, 5.0, layerCount);

        std::cout << std::endl;
        std::cout << "Input: " << std::endl;
        printVector(input);

        result = forwardPass(network, input);

        std::cout << std::endl;
        std::cout << "Result: " << std::endl;
        printVector(result);
    }
}

void
    functionDataSetTest()
{
    std::cout << std::endl;
    std::cout << "=== Function test ===" << std::endl;

    auto activationFn = SigmoidActivationFn();
    std::vector<size_t> layerWidths = { 5, 20, 10, 10 };
    auto network = buildNetwork(activationFn, layerWidths);

    size_t sampleCount = 1000;
    std::vector<DataSample> dataset = generateDataset(sampleCount, 5, 10);

    size_t epochsCount = 7000;
    for (size_t j = 0; j < epochsCount; ++j)
    {
        if (j % 1000 == 0)
        {
            std::cout << epochsCount - j << " epochs left" << std::endl;
        }

        for (size_t i = 0; i < sampleCount; ++i)
        {

            auto sample = dataset[i];

            double learningRate = 0.07;
            backPropagation(
                network, sample.inputs, sample.outputs, learningRate);
        }
    }

    testDataSet(network, dataset, 10);
}

void
    xorDataSetTest()
{
    std::cout << std::endl;
    std::cout << "=== XOR test ===" << std::endl;

    std::vector<DataSample> dataset = { { { 0.0, 0.0 }, { 0.0 } },
        { { 0.0, 1.0 }, { 1.0 } },
        { { 1.0, 0.0 }, { 1.0 } },
        { { 1.0, 1.0 }, { 0.0 } } };

    auto activationFn = SigmoidActivationFn();
    std::vector<size_t> layerWidths = { 2, 3, 1 };
    auto network = buildNetwork(activationFn, layerWidths);

    for (size_t j = 0; j < 3000; ++j)
    {
        for (size_t i = 0; i < dataset.size(); ++i)
        {
            auto sample = dataset[i];

            double learningRate = 0.3;
            backPropagation(
                network, sample.inputs, sample.outputs, learningRate);
        }
    }

    testDataSet(network, dataset);
}

int
    main()
{
    constantTest();
    xorDataSetTest();
    functionDataSetTest();

    return (0);
}
