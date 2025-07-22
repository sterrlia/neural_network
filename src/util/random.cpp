#include <random>

class Random {
    public:
        Random() : gen(std::random_device{}()) {}

        double getDouble(double min, double max) {
            std::uniform_real_distribution<double> dist(min, max);
            return dist(gen);
        }

        int getInt(int min, int max) {
            std::uniform_int_distribution<int> dist(min, max);
            return dist(gen);
        }

    private:
        std::mt19937 gen;
};
