#include "Common.hpp"

#include <vector>
#include <deque>
#include <cmath>
#include <mutex>


namespace threadpool {

class TaskBuilder;

} // namespace threadpool


namespace nn {

class Layer;
class NeuralNetwork;
class PackedNeuralNetwork;

using Values = std::vector<float, AlignmentAllocator<float, 32>>;

enum class InputMode : uint8_t
{
    Unknown,
    Full,           // full list of inputs as floats
    Sparse,         // list of sparse inputs (as floats)
    SparseBinary,   // list of sparse binary inputs (active feature is always 1)
};

enum class OutputMode : uint8_t
{
	Single,
    Array,
};

struct ActiveFeature
{
    uint32_t index;
    float value;
};

struct TrainingVector
{
    InputMode inputMode = InputMode::Unknown;
    OutputMode outputMode = OutputMode::Single;

    // depends on 'inputMode'
    Values inputs;
    std::vector<uint16_t> sparseBinaryInputs;
    std::vector<ActiveFeature> sparseInputs;

    // depends on 'outputMode'
    Values outputs;
    float singleOutput;

    void CombineSparseInputs();
    void Validate() const;
};

using TrainingSet = std::vector<TrainingVector>;

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

class LayerRunContext
{
public:
    InputMode inputMode;

    Values inputs;
    std::vector<uint16_t> sparseBinaryInputs;
    std::vector<ActiveFeature> sparseInputs;

    Values linearValue;
    Values output;

    // used for learning
    Values inputGradient;

    void Init(const Layer& layer);
    void ComputeOutput(ActivationFunction activationFunction);
};

struct Gradients
{
    uint32_t            m_numInputs;
    uint32_t            m_numOutputs;
    Values              m_values;
    std::vector<bool>   m_dirty;

    void Init(uint32_t numInputs, uint32_t numOutputs);
    void Clear();
    void Accumulate(Gradients& rhs);
};

class Layer
{
public:
    Layer(uint32_t inputSize, uint32_t outputSize);

    void InitWeights();
    void Run(const Values& in, LayerRunContext& ctx) const;
    void Run(uint32_t numFeatures, const uint16_t* binaryFeatures, LayerRunContext& ctx) const;
    void Run(uint32_t numFeatures, const ActiveFeature* features, LayerRunContext& ctx) const;
    void Backpropagate(const Values& error, LayerRunContext& ctx, Gradients& gradients) const;
    void UpdateWeights(float learningRate, const Gradients& gradients, const float gradientScale, const float weightsRange, const float biasRange, const float weightDecay);
    void QuantizeWeights(float strength);

    uint32_t numInputs;
    uint32_t numOutputs;

    ActivationFunction activationFunction;

    Values weights;

    // used for learning
    Values gradientMean;
    Values gradientMoment;
};

class NeuralNetworkRunContext
{
public:
    std::vector<LayerRunContext> layers;
    
    // used for learning
    Values tempValues;

    void Init(const NeuralNetwork& network);
};

class NeuralNetwork
{
    friend class NeuralNetworkTrainer;

public:

    // Create multi-layer neural network
    void Init(uint32_t inputSize, const std::vector<uint32_t>& layersSizes, ActivationFunction outputLayerActivationFunc = ActivationFunction::Sigmoid);

    // save to file
    bool Save(const char* filePath) const;

    // load from file
    bool Load(const char* filePath);

    // convert to packed (quantized) network
    bool ToPackedNetwork(PackedNeuralNetwork& outNetwork) const;

    // Calculate neural network output based on arbitrary input
    const Values& Run(const Values& input, NeuralNetworkRunContext& ctx) const;

    // Calculate neural network output based on ssparse input (list of active features)
    const Values& Run(uint32_t numFeatures, const ActiveFeature* features, NeuralNetworkRunContext& ctx) const;

    // Calculate neural network output based on ssparse input (list of active binary features)
    const Values& Run(uint32_t numFeatures, const uint16_t* features, NeuralNetworkRunContext& ctx) const;

    void PrintStats() const;

    uint32_t GetLayersNumber() const
    {
        return (uint32_t)layers.size();
    }

    uint32_t GetInputSize() const
    {
        return layers.front().numInputs;
    }

    uint32_t GetOutputSize() const
    {
        return layers.back().numOutputs;
    }

    std::vector<Layer> layers;
};

struct TrainParams
{
    size_t batchSize = 32;
    float learningRate = 0.5f;
    bool clampWeights = true;
};

class NeuralNetworkTrainer
{
public:

    NeuralNetworkTrainer();

    void Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder = nullptr);

private:

    struct PerThreadData
    {
        std::deque<Gradients>       gradients;      // per-layer gradients
        NeuralNetworkRunContext     runContext;
    };

    std::vector<PerThreadData> m_perThreadData;
};

} // namespace nn
