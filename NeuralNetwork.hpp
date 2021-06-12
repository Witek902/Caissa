#include <vector>

namespace nn {

struct TrainingVector
{
    std::vector<float> input;
    std::vector<float> output;
};

inline float InvTan(float x)
{
    return atanf(x);
}
inline float InvTanDerivative(float x)
{
    return 1.0f / (1.0f + x * x);
}

inline float Sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
inline float SigmoidDerivative(float x)
{
    float s = Sigmoid(x);
    return s * (1.0f - s);
}

inline float ReLu(float x)
{
    return x < 0.0f ? 0.0f : x;
}
inline float ReLuDerivative(float x)
{
    return x < 0.0f ? 0.0f : 1.0f;
}

enum class ActivationFunction
{
    Linear,
    ReLu,
    Sigmoid,
    ATan,
};

class Layer
{
    friend class NeuralNetwork;

public:
    using Values = std::vector<float>;

    Layer(size_t inputSize, size_t outputSize);

    void InitWeights();
    void Run(const Values& in);
    void Backpropagate(const Values& error);
    void QuantizeWeights(float strength);

    // Get weight of a specific neuron input
    float GetWeight(size_t neuronIdx, size_t neuronInputIdx) const;

    // Set a new weight for a specific neuron input
    void SetWeight(size_t neuronIdx, size_t neuronInputIdx, float newWeigth);

    const Layer::Values& GetOutput() const { return output; }
    const Layer::Values& GetNextError() const { return nextError; }

private:

    Values linearValue;
    Values output;
    Values input;
    Values weights;

    // used for learning
    Values gradient;
    Values nextError;
    Values adam_m;
    Values adam_v;

    ActivationFunction activationFunction;
};


class NeuralNetwork
{
public:

    // Create multi-layer neural network
    void Init(size_t inputSize, const std::vector<size_t>& layersSizes);

    // save to file
    bool Save(const char* filePath) const;

    // load from file
    bool Load(const char* filePath);

    // Calculate neural network output based on input
    const Layer::Values& Run(const Layer::Values& input);

    // Train the neural network
    void Train(const std::vector<TrainingVector>& trainingSet, Layer::Values& tempValues, size_t batchSize);

    void NextEpoch();

    Layer& GetLayer(size_t idx)
    {
        return layers[idx];
    }

    size_t GetLayersNumber() const
    {
        return layers.size();
    }

    size_t GetInputSize() const
    {
        return layers.front().input.size();
    }

    size_t GetOutputSize() const
    {
        return layers.back().output.size();
    }

    const Layer::Values& GetOutput() const
    {
        return layers.back().GetOutput();
    }

private:

    void UpdateLayerWeights(Layer& layer, float scale) const;

    std::vector<Layer> layers;

    // used for learning
    Layer::Values tempError;
    float adamBeta1;
    float adamBeta2;
};

} // namespace nn
