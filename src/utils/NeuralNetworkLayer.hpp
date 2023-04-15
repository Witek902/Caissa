#pragma once

#include "NeuralNetworkCommon.hpp"

namespace nn {

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

enum class ActivationFunction : uint8_t
{
    Linear,
    ReLU,
    CReLU,
    Sigmoid,
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
    // maximum number of layer outputs
    static constexpr uint32_t MaxLayerOutputs = 2048;

    Layer(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants = 1);

    struct WeightsUpdateOptions
    {
        float learningRate = 1.0f;
        float gradientScale = 1.0f;
        float weightsRange = 10.0f;
        float biasRange = 10.0f;
        float weightDecay = 0.0f;
        size_t iteration = 0;
    };

    void InitWeights();
    void Run(uint32_t variantIndex, const float* values, LayerRunContext& ctx, float additionalBias = 0.0f) const;
    void Run(uint32_t variantIndex, uint32_t numFeatures, const uint16_t* binaryFeatures, LayerRunContext& ctx) const;
    void Run(uint32_t variantIndex, uint32_t numFeatures, const ActiveFeature* features, LayerRunContext& ctx) const;
    void Backpropagate(uint32_t variantIndex, const Values& error, LayerRunContext& ctx, Gradients& gradients) const;
    void UpdateWeights_Adadelta(uint32_t variantIndex, const Gradients& gradients, const WeightsUpdateOptions& options);
    void UpdateWeights_Adam(uint32_t variantIndex, const Gradients& gradients, const WeightsUpdateOptions& options);

    uint32_t numInputs;
    uint32_t numOutputs;
    ActivationFunction activationFunc;

    // weights variant is selected globally for the whole network (it's kind of network input)
    struct Variant
    {
        Values weights;
        Values weightsMask;

        // used for learning
        Values gradientMoment1;
        Values gradientMoment2;
    };

    std::vector<Variant> variants;

private:

    Variant& GetVariant(uint32_t index) { return index < variants.size() ? variants[index] : variants.front(); }
    const Variant& GetConstVariant(uint32_t index) const { return index < variants.size() ? variants[index] : variants.front(); }
};

} // namespace nn
