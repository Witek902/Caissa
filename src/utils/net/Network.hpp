#pragma once

#include "Common.hpp"
#include "Node.hpp"
#include "Gradient.hpp"

#include <vector>
#include <deque>
#include <cmath>
#include <mutex>


namespace threadpool {

class TaskBuilder;

} // namespace threadpool


namespace nn {

class NeuralNetwork;
struct WeightsStorage;

struct NodeInput
{
    InputMode mode = InputMode::Unknown;
    uint32_t numFeatures = 0;

    union
    {
        const float* floatValues;
        const ActiveFeature* floatFeatures;
        const uint16_t* binaryFeatures;
    };

    INLINE NodeInput() = default;

    INLINE NodeInput(const std::vector<float>& features)
        : mode(InputMode::Full)
        , numFeatures(static_cast<uint32_t>(features.size()))
        , floatValues(features.data())
    {}

    INLINE NodeInput(const std::vector<ActiveFeature>& features)
        : mode(InputMode::Sparse)
        , numFeatures(static_cast<uint32_t>(features.size()))
        , floatFeatures(features.data())
    {}

    INLINE NodeInput(const std::vector<uint16_t>& features)
        : mode(InputMode::SparseBinary)
        , numFeatures(static_cast<uint32_t>(features.size()))
        , binaryFeatures(features.data())
    {}

    void Validate() const;
};

struct NodeOutput
{
    OutputMode mode = OutputMode::Unknown;
    uint32_t numValues = 0;

    union
    {
        float singleValue;
        const float* floatValues;
    };

    INLINE NodeOutput() = default;

    INLINE NodeOutput(const float value)
        : mode(OutputMode::Single)
        , singleValue(value)
    {}

    INLINE NodeOutput(const std::vector<float>& values)
        : mode(OutputMode::Full)
        , numValues(static_cast<uint32_t>(values.size()))
        , floatValues(values.data())
    {}
};

struct InputDesc
{
    NodeInput inputs[MaxInputNodes];
    uint32_t variant; // used to select weights variant in deeper layers
};

struct TrainingVector
{
    InputDesc input;
    NodeOutput output;
};

using TrainingSet = std::vector<TrainingVector>;

class NeuralNetworkRunContext
{
public:
    std::vector<NodeContextPtr> nodeContexts;
    std::vector<const Values*> inputErrors;     // input error for each node
    Values tempValues;                          // used for last node output

    void Init(const NeuralNetwork& network);
};

class NeuralNetwork
{
    friend class NeuralNetworkTrainer;
    friend class NeuralNetworkRunContext;

public:

    // create network from nodes
    // last node must be output node
    void Init(const std::vector<NodePtr>& nodes);

    // save to file
    bool Save(const char* filePath) const;

    // load from file
    bool Load(const char* filePath);

    // Calculate neural network output based on input
    const Values& Run(const InputDesc& inputDesc, NeuralNetworkRunContext& ctx) const;

    void PrintStats() const;

private:

    std::vector<NodePtr> m_nodes;
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

    void Init(NeuralNetwork& network);
    size_t Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder = nullptr);

private:

    struct PerThreadData
    {
        std::vector<Gradients*> perNodeGradients;
        std::vector<Gradients>  perWeightsStorageGradients;
        NeuralNetworkRunContext runContext;
    };

    std::vector<WeightsStorage*> m_weightsStorages;
    std::vector<PerThreadData> m_perThreadData;
};

} // namespace nn
