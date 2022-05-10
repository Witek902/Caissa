#pragma once

#include "Common.hpp"

#include <vector>
#include <cmath>

namespace nn {

class NeuralNetwork;

static constexpr uint32_t CurrentVersion = 1;

static constexpr uint32_t FirstLayerSize = 512;
static constexpr uint32_t SecondLayerSize = 32;
static constexpr uint32_t ThirdLayerSize = 64;

// by this value neuron inputs are scaled (so quantized 127 maps to 1.0 float)
static constexpr float ActivationRangeScaling = 127;

static constexpr int32_t WeightScaleShift = 6;
static constexpr int32_t WeightScale = 1 << WeightScaleShift;

static constexpr int32_t OutputScaleShift = 8;
static constexpr int32_t OutputScale = 1 << OutputScaleShift;

static constexpr float InputLayerWeightQuantizationScale = ActivationRangeScaling;
static constexpr float InputLayerBiasQuantizationScale = ActivationRangeScaling;
static constexpr float HiddenLayerWeightQuantizationScale = WeightScale;
static constexpr float HiddenLayerBiasQuantizationScale = WeightScale * ActivationRangeScaling;
static constexpr float OutputLayerWeightQuantizationScale = WeightScale * OutputScale / ActivationRangeScaling;
static constexpr float OutputLayerBiasQuantizationScale = WeightScale * OutputScale;

using FirstLayerWeightType = int16_t;
using FirstLayerBiasType = int16_t;

using HiddenLayerWeightType = int8_t;
using HiddenLayerBiasType = int32_t;

struct alignas(64) Accumulator
{
    FirstLayerWeightType values[FirstLayerSize];

    void Refresh(
        const FirstLayerWeightType* weights, const FirstLayerBiasType* biases,
        uint32_t numInputs, uint32_t numOutputs,
        uint32_t numActiveFeatures, const uint16_t* activeFeatures);

    void Update(
        const Accumulator& source,
        const FirstLayerWeightType* weights,
        uint32_t numInputs, uint32_t numOutputs,
        uint32_t numAddedFeatures, const uint16_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint16_t* removedFeatures);
};

class PackedNeuralNetwork
{
public:

    struct Header
    {
        uint32_t version = 0;
        uint32_t numInputs = 0;
        uint32_t firstLayerSize = 0;
        uint32_t secondLayerSize = 0;
        uint32_t thirdLayerSize = 0;
    };

    friend class NeuralNetwork;

    static constexpr uint32_t MaxNeuronsInLayer = 4096;
    static constexpr uint32_t MaxNumLayers = 4;

    PackedNeuralNetwork();
    ~PackedNeuralNetwork();

    // load from file
    bool Load(const char* filePath);

    // save to file
    bool Save(const char* filePath) const;

    // Calculate neural network output based on incremetally updated accumulator
    int32_t Run(const Accumulator& accumulator) const;

    // Calculate neural network output based on input
    int32_t Run(const uint16_t* activeInputIndices, const uint32_t numActiveInputs) const;

    INLINE uint32_t GetNumInputs() const { return header.numInputs; }
    INLINE const FirstLayerWeightType* GetAccumulatorWeights() const { return layer0_weights; }
    INLINE const FirstLayerBiasType* GetAccumulatorBiases() const { return layer0_biases; }

    bool IsValid() const { return header.numInputs > 0; }

private:

    Header header;

    FirstLayerWeightType* layer0_weights = nullptr;
    FirstLayerBiasType layer0_biases[FirstLayerSize];

    HiddenLayerWeightType layer1_weights[FirstLayerSize * SecondLayerSize];
    HiddenLayerBiasType layer1_biases[SecondLayerSize];

    HiddenLayerWeightType layer2_weights[SecondLayerSize * ThirdLayerSize];
    HiddenLayerBiasType layer2_biases[ThirdLayerSize];

    HiddenLayerWeightType layer3_weights[ThirdLayerSize];
    HiddenLayerBiasType layer3_biases[1];
};

} // namespace nn
