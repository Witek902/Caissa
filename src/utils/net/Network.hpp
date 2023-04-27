#pragma once

#include "Common.hpp"
#include "Node.hpp"

#include <vector>
#include <deque>
#include <cmath>
#include <mutex>


namespace threadpool {

class TaskBuilder;

} // namespace threadpool


namespace nn {

class NeuralNetwork;

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

    void CombineSparseInputs();
    void Validate() const;
};

using TrainingSet = std::vector<TrainingVector>;

class NeuralNetworkRunContext
{
public:
    std::vector<NodeContextPtr> nodeContexts;
    
    // used for learning
    Values tempValues;

    void Init(const NeuralNetwork& network);
};

class NeuralNetwork
{
    friend class NeuralNetworkTrainer;
    friend class NeuralNetworkRunContext;

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

    // create network from nodes
    // last node must be output node
    void Init(const std::vector<NodePtr>& nodes);

    // save to file
    bool Save(const char* filePath) const;

    // load from file
    bool Load(const char* filePath);

    // Calculate neural network output based on input
    const Values& Run(const InputDesc& input, NeuralNetworkRunContext& ctx) const;

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

    size_t Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder = nullptr);

private:

    struct PerThreadData
    {
        std::vector<Gradients>  gradients;      // per-node gradients
        NeuralNetworkRunContext runContext;
    };

    std::vector<PerThreadData> m_perThreadData;
};

} // namespace nn
