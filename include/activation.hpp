class ActivationFnInterface
{
    public:
        virtual double invoke(double x) const = 0;
        virtual double invokeDerivative(double x) const = 0;
        virtual double getDerivativeResultByInvokeOutput(double x) const = 0;
};

class SigmoidActivationFn : public ActivationFnInterface
{
    public:
        double invoke(double x) const override;
        double getDerivativeResultByInvokeOutput(double y) const override;
        double invokeDerivative(double x) const override;
};

