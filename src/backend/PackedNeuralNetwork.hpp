#pragma once

#include "Common.hpp"

#include <cmath>
#include <vector>

#if defined(PLATFORM_WINDOWS)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif // NOMINMAX
    #include <Windows.h>
#endif // PLATFORM_WINDOWS

namespace nn {

class NeuralNetwork;

static constexpr uint32_t CurrentVersion = 6;
static constexpr uint32_t MagicNumber = 'CSNN';

static constexpr uint32_t NumKingBuckets = 3;
static constexpr uint32_t NumNetworkInputs = NumKingBuckets * 12 * 64;
static constexpr uint32_t AccumulatorSize = 768;
static constexpr uint32_t OutputSize = 1;
static constexpr uint32_t NumVariants = 16;

static constexpr uint8_t KingBucketIndex[32] =
{
    0, 0, 1, 1,
    0, 0, 1, 1,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
};

// by this value neuron inputs are scaled (so quantized 127 maps to 1.0 float)
static constexpr float ActivationRangeScaling = 127;

static constexpr int32_t WeightScaleShift = 8; // TODO should be 6 if we clamp weights to [-2,2] range
static constexpr int32_t WeightScale = 1 << WeightScaleShift;

static constexpr int32_t OutputScaleShift = 8;
static constexpr int32_t OutputScale = 1 << OutputScaleShift;

static constexpr float InputLayerWeightQuantizationScale = ActivationRangeScaling;
static constexpr float InputLayerBiasQuantizationScale = ActivationRangeScaling;
static constexpr float HiddenLayerWeightQuantizationScale = WeightScale;
static constexpr float HiddenLayerBiasQuantizationScale = WeightScale * ActivationRangeScaling;
static constexpr float OutputLayerWeightQuantizationScale = WeightScale * OutputScale / ActivationRangeScaling;
static constexpr float OutputLayerBiasQuantizationScale = WeightScale * OutputScale;

using AccumulatorType = int16_t;

using FirstLayerWeightType = int16_t;
using FirstLayerBiasType = int16_t;

using HiddenLayerWeightType = int8_t;
using HiddenLayerBiasType = int32_t;

using LastLayerWeightType = int16_t;
using LastLayerBiasType = int32_t;

struct alignas(CACHELINE_SIZE) Accumulator
{
    AccumulatorType values[AccumulatorSize];

    void Refresh(
        const FirstLayerWeightType* weights, const FirstLayerBiasType* biases,
        uint32_t numActiveFeatures, const uint16_t* activeFeatures);

    void Update(
        const Accumulator& source,
        const FirstLayerWeightType* weights,
        uint32_t numAddedFeatures, const uint16_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint16_t* removedFeatures);
};

class PackedNeuralNetwork
{
public:

    friend class NeuralNetwork;

    static constexpr uint32_t MaxInputs = 262144;
    static constexpr uint32_t MaxNeuronsInHiddenLayers = 128;
    static constexpr uint32_t MinNeuronsInHiddenLayers = 8;
    static constexpr uint32_t MaxNumLayers = 4;

    struct Header
    {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t layerSizes[MaxNumLayers] = { 0, 0, 0, 0 };
        uint32_t layerVariants[MaxNumLayers] = { 0, 0, 0, 0 };
        uint32_t padding[6];
    };

    PackedNeuralNetwork();
    ~PackedNeuralNetwork();

    // unload weights
    void Release();

    // allocate weights
    bool Resize(const std::vector<uint32_t>& layerSizes,
                const std::vector<uint32_t>& numVariantsPerLayer = std::vector<uint32_t>());

    // load from file
    bool Load(const char* filePath);

    // save to file
    bool Save(const char* filePath) const;

    // Calculate neural network output based on incrementally updated accumulators
    int32_t Run(const Accumulator& stmAccum, const Accumulator& nstmAccum, uint32_t variant) const;

    // Calculate neural network output based on input
    int32_t Run(const uint16_t* stmFeatures, const uint32_t stmNumFeatures, const uint16_t* nstmFeatures, const uint32_t nstmNumFeatures, uint32_t variant) const;

    INLINE uint32_t GetNumInputs() const { return header.layerSizes[0]; }
    INLINE uint32_t GetAccumulatorSize() const { return header.layerSizes[1] / 2; }
    INLINE uint32_t GetLayerSize(uint32_t i) const { return header.layerSizes[i]; }

    void GetLayerWeightsAndBiases(uint32_t layerIndex, uint32_t layerVariant, const void*& outWeights, const void*& outBiases) const;

    INLINE const FirstLayerWeightType* GetAccumulatorWeights() const
    {
        return reinterpret_cast<const FirstLayerWeightType*>(layerDataPointers[0]);
    }
    INLINE const FirstLayerBiasType* GetAccumulatorBiases() const
    {
        return reinterpret_cast<const FirstLayerBiasType*>(GetAccumulatorWeights() + GetNumInputs() * GetAccumulatorSize());
    }

    template<typename T>
    INLINE const T* GetLayerWeights(uint32_t index, uint32_t variant) const
    {
        const void* weights;
        const void* biases;
        GetLayerWeightsAndBiases(index, variant, weights, biases);
        return reinterpret_cast<const T*>(weights);
    }

    template<typename T>
    INLINE const T* GetLayerBiases(uint32_t index, uint32_t variant) const
    {
        const void* weights;
        const void* biases;
        GetLayerWeightsAndBiases(index, variant, weights, biases);
        return reinterpret_cast<const T*>(biases);
    }

    // calculate size of all weights buffer
    size_t GetWeightsBufferSize() const;

    bool IsValid() const { return GetNumInputs() > 0; }

private:

    void ReleaseFileMapping();

    void InitLayerDataSizes();
    void InitLayerDataPointers();

    Header header;

    uint32_t numActiveLayers = 0;
    uint32_t layerDataSizes[MaxNumLayers];  // size of each layer (in bytes)
    uint8_t* layerDataPointers[MaxNumLayers];  // base pointer to weights of each layer

    // file mapping
#if defined(PLATFORM_WINDOWS)
    HANDLE fileHandle = INVALID_HANDLE_VALUE;
    HANDLE fileMapping = INVALID_HANDLE_VALUE;
#else
    int fileDesc = -1;
#endif // PLATFORM_WINDOWS

    void* mappedData = nullptr;
    size_t mappedSize = 0;

    // all weights and biases are stored in this buffer
    uint8_t* weightsBuffer = nullptr;
};

} // namespace nn
