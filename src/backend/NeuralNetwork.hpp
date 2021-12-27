#include <vector>
#include <cmath>

namespace nn {

class PackedNeuralNetwork;

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

inline float ClippedReLu(float x)
{
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 1.0f;
    return x;
}
inline float ClippedReLuDerivative(float x)
{
    if (x <= 0.0f) return 0.0f;
    if (x >= 1.0f) return 0.0f;
    return 1.0f;
}

enum class ActivationFunction
{
    Linear,
    ClippedReLu,
    Sigmoid,
    ATan,
};

struct Layer
{
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

    Values linearValue;
    Values output;
    Values input;
    Values weights;

    // used for learning
    Values gradient;
    Values nextError;
    Values m;
    Values v;

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

    // convert to packed (quantized) network
    bool ToPackedNetwork(PackedNeuralNetwork& outNetwork) const;

    // Calculate neural network output based on input
    const Layer::Values& Run(const Layer::Values& input);

    // Train the neural network
    void Train(const std::vector<TrainingVector>& trainingSet, Layer::Values& tempValues, size_t batchSize);

    void PrintStats() const;

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
        return layers.back().output;
    }

private:

    void UpdateLayerWeights(Layer& layer) const;
    void QuantizeLayerWeights(Layer& layer, float weightQuantizationScale, float biasQuantizationScale) const;

    std::vector<Layer> layers;

    // used for learning
    Layer::Values tempError;
};

} // namespace nn
