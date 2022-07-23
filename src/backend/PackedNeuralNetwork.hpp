#pragma once

#include "Common.hpp"

#include <vector>
#include <cmath>

#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include <Windows.h>

namespace nn {

class NeuralNetwork;

static constexpr uint32_t CurrentVersion = 1;
static constexpr uint32_t MagicNumber = 'CSNN';

static constexpr uint32_t FirstLayerSize = 512;
static constexpr uint32_t OutputSize = 1;

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

struct alignas(CACHELINE_SIZE) Accumulator
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
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t layerSizes[4] = { 0, 0, 0, 0 };
        uint32_t padding[10];
    };

    friend class NeuralNetwork;

    static constexpr uint32_t MaxInputs = 262144;
    static constexpr uint32_t MaxNeuronsInFirstLayer = 1024;
    static constexpr uint32_t MaxNeuronsInLaterLayers = 128;
    static constexpr uint32_t MinNeuronsInLaterLayers = 16;
    static constexpr uint32_t MaxNumLayers = 4;

    PackedNeuralNetwork();
    ~PackedNeuralNetwork();

    // unload weights
    void Release();

    // allocate weights
    bool Resize(uint32_t numInputs, uint32_t l1size, uint32_t l2size, uint32_t l3size);

    // load from file
    bool Load(const char* filePath);

    // save to file
    bool Save(const char* filePath) const;

    // Calculate neural network output based on incremetally updated accumulator
    int32_t Run(const Accumulator& accumulator) const;

    // Calculate neural network output based on input
    int32_t Run(const uint16_t* activeInputIndices, const uint32_t numActiveInputs) const;

    INLINE uint32_t GetNumInputs() const { return header.layerSizes[0]; }
    INLINE uint32_t GetLayerSize(uint32_t i) const { return header.layerSizes[i]; }

    INLINE const FirstLayerWeightType* GetAccumulatorWeights() const
    {
        return reinterpret_cast<const FirstLayerWeightType*>(weightsBuffer);
    }
    INLINE const FirstLayerBiasType* GetAccumulatorBiases() const
    {
        return reinterpret_cast<const FirstLayerWeightType*>(GetAccumulatorWeights() + GetNumInputs() * GetLayerSize(1));
    }

    INLINE const HiddenLayerWeightType* GetLayer1Weights() const
    {
        return reinterpret_cast<const HiddenLayerWeightType*>(GetAccumulatorBiases() + GetLayerSize(1));
    }
    INLINE const HiddenLayerBiasType* GetLayer1Biases() const
    {
        return reinterpret_cast<const HiddenLayerBiasType*>(GetLayer1Weights() + GetLayerSize(1) * GetLayerSize(2));
    }

    INLINE const HiddenLayerWeightType* GetLayer2Weights() const
    {
        return reinterpret_cast<const HiddenLayerWeightType*>(GetLayer1Biases() + GetLayerSize(2));
    }
    INLINE const HiddenLayerBiasType* GetLayer2Biases() const
    {
        return reinterpret_cast<const HiddenLayerBiasType*>(GetLayer2Weights() + GetLayerSize(2) * GetLayerSize(3));
    }

    INLINE const HiddenLayerWeightType* GetLayer3Weights() const
    {
        return reinterpret_cast<const HiddenLayerWeightType*>(GetLayer2Biases() + GetLayerSize(3));
    }
    INLINE const HiddenLayerBiasType* GetLayer3Biases() const
    {
        return reinterpret_cast<const HiddenLayerBiasType*>(GetLayer3Weights() + GetLayerSize(3) * OutputSize);
    }

    // calculate size of all weights buffer
    size_t GetWeightsBufferSize() const;

    bool IsValid() const { return GetNumInputs() > 0; }

private:

    void ReleaseFileMapping();

    Header header;

    // file mapping
    HANDLE fileHandle = INVALID_HANDLE_VALUE;
    HANDLE fileMapping = INVALID_HANDLE_VALUE;
    void* mappedData = nullptr;

    // all weights and biases are stored in this buffer
    uint8_t* weightsBuffer = nullptr;
};

} // namespace nn
