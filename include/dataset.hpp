#include "util/random.hpp"
#include <vector>

struct DataSample
{
    std::vector<double> inputs;
    std::vector<double> outputs;
};

// Refuses to link if I put it in .cpp file
// Partially AI generated
inline std::vector<DataSample>
    generateDataset(size_t samplesCount, size_t inputWidth, size_t outputWidth)
{
    std::vector<DataSample> dataset;
    auto random = Random();

    for (size_t s = 0; s < samplesCount; ++s)
    {
        std::vector<double> inputs =
            random.getDoubleRange(-0.1, 0.1, inputWidth);

        std::vector<double> outputs(outputWidth, 0.0);

        // Simple example relationships between inputs and outputs:
        for (size_t o = 0; o < outputWidth; ++o)
        {
            double val = 0.0;

            // Example: output[o] = sum of some inputs with nonlinear transforms
            for (size_t i = 0; i < inputWidth; ++i)
            {
                double weight =
                    (o + 1) * 0.1 + i * 0.01; // some weights pattern
                double inputVal = inputs[i];

                // Nonlinear transform example
                if (i % 3 == 0)
                    val += weight * std::sin(inputVal);
                else if (i % 3 == 1)
                    val += weight * std::log(inputVal + 1.1);
                else
                    val += weight * inputVal * inputVal;
            }

            // Optional nonlinearity on output
            outputs[o] = std::abs(std::tanh(val));
        }

        dataset.push_back({ inputs, outputs });
    }

    return dataset;
}
