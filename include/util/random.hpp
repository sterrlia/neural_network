#pragma once

#include <random>
#include <vector>

class Random {
    public:
        Random();

        double getDouble(double min, double max) const;

        std::vector<double> getDoubleRange(double min, double max, int count) const;

        int getInt(int min, int max) const;

    private:
        mutable std::mt19937 gen;
};
