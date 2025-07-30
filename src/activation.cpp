#include "activation.hpp"
#include <cmath>
#include <numbers>

double
    SigmoidActivationFn::invoke(double x) const
{
    return 1 / (1 + std::pow(std::numbers::e, -x));
}

double
    SigmoidActivationFn::getDerivativeResultByInvokeOutput(double y) const
{
    return y * (1 - y);
}

double
    SigmoidActivationFn::invokeDerivative(double x) const
{
    double e = std::numbers::e;
    return std::pow(e, -x) / std::pow(1 + std::pow(e, -x), 2);
}
