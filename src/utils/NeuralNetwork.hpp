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

    // Calculate neural network output based on sparse input (list of active features)
    const Values& Run(uint32_t numFeatures, const ActiveFeature* features, NeuralNetworkRunContext& ctx) const;

    // Calculate neural network output based on sparse input (list of active binary features)
    const Values& Run(uint32_t numFeatures, const uint16_t* features, NeuralNetworkRunContext& ctx) const;

    void PrintStats() const;

    uint32_t GetLayersNumber() const { return (uint32_t)layers.size(); }
    uint32_t GetInputSize() const { return layers.front().numInputs; }
    uint32_t GetOutputSize() const { return layers.back().numOutputs; }

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
