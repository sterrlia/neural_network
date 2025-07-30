#include "util/random.hpp"
#include <random>

Random::Random()
    : gen(std::random_device{}()) {}

double Random::getDouble(double min, double max) const {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

int Random::getInt(int min, int max) const {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

std::vector<double> Random::getDoubleRange(double min, double max, int count) const {
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<double> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i) {
        result.push_back(dist(gen));
    }
    return result;
}
