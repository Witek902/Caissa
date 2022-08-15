#include "Common.hpp"

#include <vector>
#include <cmath>

namespace nn {

class PackedNeuralNetwork;

using Values = std::vector<float, AlignmentAllocator<float, 32>>;

struct TrainingVector
{
    // intput as float values or active features list
    Values inputs;
    std::vector<uint16_t> features;

    Values output;
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
inline __m256 ClippedReLu(const __m256 x)
{
    return _mm256_min_ps(_mm256_set1_ps(1.0f), _mm256_max_ps(_mm256_setzero_ps(), x));
}
inline __m256 ClippedReLuDerivative(const __m256 x, const __m256 coeff)
{
    return _mm256_and_ps(coeff,
                         _mm256_and_ps(_mm256_cmp_ps(x, _mm256_setzero_ps(),  _CMP_GT_OQ),
                                       _mm256_cmp_ps(x, _mm256_set1_ps(1.0f), _CMP_LT_OQ)));
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
    Layer(size_t inputSize, size_t outputSize);

    void InitWeights();
    void Run(const Values& in);
    void Run(const uint16_t* featureIndices, uint32_t numFeatures);
    void Backpropagate(uint32_t layerIndex, const Values& error);
    void QuantizeWeights(float strength);
    void ComputeOutput();

    std::vector<uint16_t> activeFeatures;
    Values input;

    Values linearValue;
    Values output;
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
    void Init(size_t inputSize, const std::vector<size_t>& layersSizes, ActivationFunction outputLayerActivationFunc = ActivationFunction::Sigmoid);

    // save to file
    bool Save(const char* filePath) const;

    // load from file
    bool Load(const char* filePath);

    // convert to packed (quantized) network
    bool ToPackedNetwork(PackedNeuralNetwork& outNetwork) const;

    // Calculate neural network output based on arbitrary input
    const Values& Run(const Values& input);

    // Calculate neural network output based on ssparse input (list of active features)
    const Values& Run(const uint16_t* featureIndices, uint32_t numFeatures);

    // Train the neural network
    void Train(const std::vector<TrainingVector>& trainingSet, Values& tempValues, size_t batchSize, float learningRate = 0.5f);

    void PrintStats() const;

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

    const Values& GetOutput() const
    {
        return layers.back().output;
    }

    void UpdateLayerWeights(Layer& layer, float learningRate) const;
    void QuantizeLayerWeights(size_t layerIndex, float weightRange, float biasRange, float weightQuantizationScale, float biasQuantizationScale);

    std::vector<Layer> layers;

    // used for learning
    Values tempError;
};

} // namespace nn
