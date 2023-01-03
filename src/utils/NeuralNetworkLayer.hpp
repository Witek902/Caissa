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

} // namespace nn
