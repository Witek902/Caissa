#pragma once

#include "Common.hpp"

#include <vector>
#include <cmath>

namespace nn {

static constexpr uint32_t FirstLayerSize = 128;
static constexpr uint32_t SecondLayerSize = 32;

// by this value neuron inputs are scaled (so quantized 127 maps to 1.0 float)
static constexpr float InputScale = 127;

static constexpr int32_t WeightScaleShift = 6;
static constexpr int32_t WeightScale = 1 << WeightScaleShift;

static constexpr int32_t OutputScaleShift = 8;
static constexpr int32_t OutputScale = 1 << OutputScaleShift;

using WeightTypeLayer0 = int16_t;
using BiasTypeLayer0 = int16_t;

using WeightTypeLayer12 = int8_t;
using BiasTypeLayer12 = int32_t;

struct LayerData0
{
    const WeightTypeLayer0* weights;
    const BiasTypeLayer0* biases;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;
};

struct LayerData12
{
    const WeightTypeLayer12* weights;
    const BiasTypeLayer12* biases;
    uint32_t numInputs = 0;
    uint32_t numOutputs = 0;
};

struct alignas(64) Accumulator
{
    WeightTypeLayer0 values[FirstLayerSize];

    void Refresh(
        const LayerData0& layer,
        uint32_t numActiveFeatures, const uint32_t* activeFeatures);

    void Update(
        const LayerData0& layer,
        uint32_t numAddedFeatures, const uint32_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint32_t* removedFeatures);
};

class PackedNeuralNetwork
{
public:

    friend class NeuralNetwork;

    static constexpr uint32_t MaxNeuronsInLayer = 256;
    static constexpr uint32_t MaxNumLayers = 5;

    PackedNeuralNetwork();
    ~PackedNeuralNetwork();

    // load from file
    bool Load(const char* filePath);

    // save to file
    bool Save(const char* filePath) const;

    // Calculate neural network output based on input
    int32_t Run(const uint32_t numActiveInputs, const uint32_t* activeInputIndices) const;

    bool IsValid() const { return numInputs > 0; }

private:

    uint32_t numInputs = 0;

    WeightTypeLayer0* layer0_weights = nullptr;
    BiasTypeLayer0 layer0_biases[FirstLayerSize];

    WeightTypeLayer12 layer1_weights[FirstLayerSize * SecondLayerSize];
    BiasTypeLayer12 layer1_biases[SecondLayerSize];

    WeightTypeLayer12 layer2_weights[SecondLayerSize];
    BiasTypeLayer12 layer2_biases[1];
};

} // namespace nn
