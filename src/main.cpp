#include "util/random.hpp"
#include "util/vector.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

class ActivationFnInterface
{
public:
    virtual double invoke(double x);
    virtual double invokeDerivative(double x);
};

struct Neuron;

struct NeuronLink
{
    const Neuron* neuron;
    double weight;
};

struct Neuron
{
    int id;
    double bias;
    const ActivationFnInterface& activationFn;
    std::vector<NeuronLink> links;

//private:
//    std::optional<double> value = std::nullopt;
//
//    friend std::optional<double> getNeuronValue(Neuron neuron);
};



struct LayerInterface
{
    virtual std::vector<Neuron> getNeurons();
};

template<std::size_t SIZE>
struct Layer : LayerInterface
{
    Neuron neurons[SIZE];

    std::vector<Neuron> getNeurons()
    {
        return (vec(neurons));
    }
};

template<std::size_t LAYER_COUNT,
    std::size_t INPUT_WIDTH,
    std::size_t OUTPUT_WIDTH>
struct DenseNeuralNetwork
{
    Layer<INPUT_WIDTH> inputLayer;
    LayerInterface hiddenLayers[LAYER_COUNT - 2];
    Layer<OUTPUT_WIDTH> outputLayer;
};

template<typename T>
concept ACTIVATION_FN = std::is_base_of_v<ActivationFnInterface, T>;

// rectangle network builder
template<std::size_t HEIGHT, std::size_t WIDTH>
DenseNeuralNetwork<HEIGHT, WIDTH, WIDTH>
    build_network(const ActivationFnInterface& activationFn)
{
    auto network = DenseNeuralNetwork<HEIGHT, WIDTH, WIDTH>({});
    auto random = Random();

    int maxId = 0;

    for (int j = 0; j < HEIGHT; ++j)
    {
        Layer layer = Layer<WIDTH>();
        for (int i = 0; i < WIDTH; ++i)
        {
            auto bias = random.getDouble(-0.1, 0.1);
            std::vector<NeuronLink> links = {};

            auto id = ++maxId;
            auto neuron = Neuron{ id, bias, activationFn, links };
            layer.neurons.push_back(neuron);
        }

        if (j == 0)
        {
            network.inputLayer = layer;
        }
        else if (j == WIDTH)
        {
            network.outputLayer = layer;
        }
        else
        {
            network.hiddenLayers.push_back(layer);
        }
    }

    Layer<WIDTH>* previousLayer = nullptr;
    for (Layer<WIDTH>& layer : network.hiddenLayers)
    {
        if (previousLayer == nullptr)
        {
            previousLayer = &layer;
            continue;
        }

        std::vector<NeuronLink> links = array_map(
            [&random](const Neuron& neuron)
            {
                auto weight = random.getDouble(-0.1, 0.1);

                return (NeuronLink{ &neuron, weight });
            },
            previousLayer->neurons);

        for (Neuron& previousLayerNeuron : layer.neurons)
        {
            previousLayerNeuron.links = links;
        }

        previousLayer = &layer;
    }

    return (network);
}

template<std::size_t HEIGHT, std::size_t INPUT_WIDTH, std::size_t OUTPUT_WIDTH>
std::array<double, OUTPUT_WIDTH>
    forward_pass(
        const DenseNeuralNetwork<HEIGHT, INPUT_WIDTH, OUTPUT_WIDTH>& network,
        const double (&input)[INPUT_WIDTH])
{

    auto layers = array_merge(
        { network.inputLayer }, network.hiddenLayers, { network.outputLayer });

    struct NeuronValue
    {
        int id;
        double value;
    };

    std::vector<NeuronValue> previousLayerValues;
    std::vector<NeuronValue> currentLayerValues;

    for (LayerInterface layer : layers)
    {
        currentLayerValues = {};
        for (const Neuron& neuron : layer.getNeurons())
        {
            double value = 0;
            for (const NeuronLink& link : neuron.links)
            {
                auto link_value = array_binary_find(previousLayerValues,
                    link.neuron->id,
                    [](const NeuronValue& value, int id)
                    { return value.id <=> id; });

                value += link_value * link.weight + link.neuron->bias;
            }
        }
        previousLayerValues = currentLayerValues;
    }
}

// void back_propagation(NeuralNetwork network, double learningRate) {}

int
    main()
{
    std::cout << "Hello world" << std::endl;

    return (0);
}
