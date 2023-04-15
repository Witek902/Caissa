#pragma once

#include "Common.hpp"
#include "NeuralNetworkLayer.hpp"

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

    uint32_t networkVariant = 0;
    float lastLayerBias = 0.0f;

    void CombineSparseInputs();
    void Validate() const;
};

using TrainingSet = std::vector<TrainingVector>;

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

    struct InputDesc
    {
        InputMode mode = InputMode::Unknown;
        uint32_t numFeatures = 0;

        union
        {
            const float* floatValues;
            const ActiveFeature* floatFeatures;
            const uint16_t* binaryFeatures;
        };

        // used to select weights variant in deeper layers
        uint32_t variant = 0;

        // additional bias for last layer
        float lastLayerBias = 0.0f;

        InputDesc() = default;

        InputDesc(const std::vector<float>& features)
            : mode(InputMode::Full)
            , numFeatures(static_cast<uint32_t>(features.size()))
            , floatValues(features.data())
        {}

        InputDesc(const std::vector<ActiveFeature>& features)
            : mode(InputMode::Sparse)
            , numFeatures(static_cast<uint32_t>(features.size()))
            , floatFeatures(features.data())
        {}

        InputDesc(const std::vector<uint16_t>& features)
            : mode(InputMode::SparseBinary)
            , numFeatures(static_cast<uint32_t>(features.size()))
            , binaryFeatures(features.data())
        {}
    };

    // Create multi-layer neural network
    void Init(uint32_t inputSize,
              const std::vector<uint32_t>& layersSizes,
              ActivationFunction outputLayerActivationFunc = ActivationFunction::Sigmoid,
              const std::vector<uint32_t>& layerVariants = std::vector<uint32_t>());

    // save to file
    bool Save(const char* filePath) const;

    // load from file
    bool Load(const char* filePath);

    // convert to packed (quantized) network
    bool ToPackedNetwork(PackedNeuralNetwork& outNetwork) const;

    // Calculate neural network output based on input
    const Values& Run(const InputDesc& input, NeuralNetworkRunContext& ctx) const;

    void PrintStats() const;

    uint32_t GetLayersNumber() const { return (uint32_t)layers.size(); }
    uint32_t GetInputSize() const { return layers.front().numInputs; }
    uint32_t GetOutputSize() const { return layers.back().numOutputs; }

    std::vector<Layer> layers;
};

enum class Optimizer : uint8_t
{
    Adadelta,
    Adam,
};

struct TrainParams
{
    size_t iteration = 0;
    size_t batchSize = 32;
    float learningRate = 0.5f;
    float weightDecay = 1.0e-5f;
    Optimizer optimizer = Optimizer::Adadelta;
    bool clampWeights = true;
};

class NeuralNetworkTrainer
{
public:

    NeuralNetworkTrainer();

    size_t Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder = nullptr);

private:

    using PerVariantGradients = std::vector<Gradients>;

    struct PerThreadData
    {
        std::vector<PerVariantGradients>    gradients;      // per-layer, per-variant gradients
        NeuralNetworkRunContext             runContext;
    };

    std::vector<PerThreadData> m_perThreadData;
};

} // namespace nn
