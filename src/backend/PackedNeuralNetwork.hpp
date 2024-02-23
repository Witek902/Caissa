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

#ifdef USE_SSE
    #include <immintrin.h>
#endif // USE_SSE

#ifdef USE_ARM_NEON
    #include <arm_neon.h>
#endif // USE_ARM_NEON

namespace nn {

#ifdef USE_VNNI
#define NN_USE_VNNI
#endif // USE_VNNI

#if defined(USE_AVX512)
    #define NN_USE_AVX512
    using Int16VecType = __m512i;
    constexpr const uint32_t VectorRegSize = 512;
    #define Int16VecLoad(ptr) _mm512_load_si512(reinterpret_cast<const Int16VecType*>(ptr))
    #define Int16VecStore(ptr,val) _mm512_store_si512(reinterpret_cast<Int16VecType*>(ptr), (val))
    #define Int16VecAdd _mm512_add_epi16
    #define Int16VecSub _mm512_sub_epi16

#elif defined(USE_AVX2)
    #define NN_USE_AVX2
    using Int16VecType = __m256i;
    constexpr const uint32_t VectorRegSize = 256;
    #define Int16VecLoad(ptr) _mm256_load_si256(reinterpret_cast<const Int16VecType*>(ptr))
    #define Int16VecStore(ptr,val) _mm256_store_si256(reinterpret_cast<Int16VecType*>(ptr), (val))
    #define Int16VecAdd _mm256_add_epi16
    #define Int16VecSub _mm256_sub_epi16

#elif defined(USE_SSE2)
    #define NN_USE_SSE2
    using Int16VecType = __m128i;
    constexpr const uint32_t VectorRegSize = 128;
    #define Int16VecLoad(ptr) _mm_load_si128(reinterpret_cast<const Int16VecType*>(ptr))
    #define Int16VecStore(ptr,val) _mm_store_si128(reinterpret_cast<Int16VecType*>(ptr), (val))
    #define Int16VecAdd _mm_add_epi16
    #define Int16VecSub _mm_sub_epi16

#elif defined(USE_ARM_NEON)
    #define NN_USE_ARM_NEON
    using Int16VecType = int16x8_t;
    constexpr const uint32_t VectorRegSize = 128;
    #define Int16VecLoad(ptr) (*reinterpret_cast<const int16x8_t*>(ptr))
    #define Int16VecStore(ptr,val) ((*reinterpret_cast<int16x8_t*>(ptr)) = (val))
    #define Int16VecAdd vaddq_s16
    #define Int16VecSub vsubq_s16

#endif // USE_ARM_NEON

#ifdef USE_SSE4
    #define NN_USE_SSE4
#endif // USE_SSE

#if defined(NN_USE_AVX512) || defined(NN_USE_AVX2) || defined(NN_USE_SSE2) || defined(NN_USE_ARM_NEON)
    #define NN_USE_SIMD
#endif

#if defined(NN_USE_AVX512)
    constexpr uint32_t OptimalRegisterCount = 16;
#elif defined(NN_USE_AVX2) || defined(NN_USE_SSE2) || defined(NN_USE_ARM_NEON)
    constexpr uint32_t OptimalRegisterCount = 8;
#endif // NN_USE_AVX512 || NN_USE_AVX2 || NN_USE_SSE2 || NN_USE_ARM_NEON


class NeuralNetwork;

static constexpr uint32_t CurrentVersion = 9;
static constexpr uint32_t MagicNumber = 'CSNN';

static constexpr uint32_t NumKingBuckets = 5;
static constexpr uint32_t NumNetworkInputs = NumKingBuckets * 12 * 64;
static constexpr uint32_t AccumulatorSize = 1024;
static constexpr uint32_t OutputSize = 1;
static constexpr uint32_t NumVariants = 8;

static constexpr uint8_t KingBucketIndex[64] =
{
    0, 0, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 3, 3, 3, 3, 2, 2,
    2, 2, 3, 3, 3, 3, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
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

using HiddenLayerWeightType = int8_t;
using HiddenLayerBiasType = int32_t;

using LastLayerWeightType = int16_t;
using LastLayerBiasType = int32_t;

using IntermediateType = int8_t;
using AccumulatorType = int16_t;


struct alignas(CACHELINE_SIZE) Accumulator
{
    AccumulatorType values[AccumulatorSize];

    INLINE void Refresh(
        const Accumulator weights[], const Accumulator& biases,
        uint32_t numActiveFeatures, const uint16_t* activeFeatures)
    {
#ifndef CONFIGURATION_FINAL
        // check for duplicate features
        for (uint32_t i = 0; i < numActiveFeatures; ++i)
        {
            for (uint32_t j = i + 1; j < numActiveFeatures; ++j)
            {
                ASSERT(activeFeatures[i] != activeFeatures[j]);
            }
        }
#endif // CONFIGURATION_FINAL

#if defined(NN_USE_SIMD)

        constexpr uint32_t registerWidth = VectorRegSize / (8 * sizeof(AccumulatorType));
        static_assert(AccumulatorSize % registerWidth == 0);
        ASSERT((size_t)weights % 32 == 0);
        ASSERT((size_t)&biases % 32 == 0);
        ASSERT((size_t)values % 32 == 0);

        constexpr uint32_t numChunks = AccumulatorSize / registerWidth;
        static_assert(numChunks % OptimalRegisterCount == 0, "");
        constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;

        Int16VecType regs[OptimalRegisterCount];
        for (uint32_t tile = 0; tile < numTiles; ++tile)
        {
            const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                regs[i] = Int16VecLoad(biases.values + chunkBase + i * registerWidth);

            for (uint32_t j = 0; j < numActiveFeatures; ++j)
            {
                ASSERT(activeFeatures[j] < NumNetworkInputs);
                const Accumulator& weightsStart = weights[activeFeatures[j]];
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                    regs[i] = Int16VecAdd(regs[i], Int16VecLoad(weightsStart.values + chunkBase + i * registerWidth));
            }

            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                Int16VecStore(values + chunkBase + i * registerWidth, regs[i]);
        }

#else // no SIMD support

        AccumulatorType regs[AccumulatorSize];

        for (uint32_t i = 0; i < AccumulatorSize; ++i)
            regs[i] = biases.values[i];

        for (uint32_t j = 0; j < numActiveFeatures; ++j)
        {
            const uint32_t featureIndex = activeFeatures[j];
            ASSERT(featureIndex < NumNetworkInputs);
            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                ASSERT(int32_t(regs[i]) + int32_t(weights[featureIndex].values[i]) <= std::numeric_limits<AccumulatorType>::max());
                ASSERT(int32_t(regs[i]) + int32_t(weights[featureIndex].values[i]) >= std::numeric_limits<AccumulatorType>::min());
                regs[i] += weights[featureIndex].values[i];
            }
        }

        for (uint32_t i = 0; i < AccumulatorSize; ++i)
            values[i] = static_cast<AccumulatorType>(regs[i]);
#endif // NN_USE_SIMD
    }


    INLINE void Update(
        const Accumulator& source,
        const Accumulator weights[],
        uint32_t numAddedFeatures, const uint16_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint16_t* removedFeatures)
    {
#if defined(NN_USE_SIMD)

        constexpr uint32_t registerWidth = VectorRegSize / (8 * sizeof(AccumulatorType));
        static_assert(AccumulatorSize % registerWidth == 0);
        const uint32_t numChunks = AccumulatorSize / registerWidth;
        static_assert(numChunks % OptimalRegisterCount == 0);
        const uint32_t numTiles = numChunks / OptimalRegisterCount;
        ASSERT((size_t)weights % 32 == 0);
        ASSERT((size_t)source.values % 32 == 0);
        ASSERT((size_t)values % 32 == 0);

        Int16VecType regs[OptimalRegisterCount];
        for (uint32_t tile = 0; tile < numTiles; ++tile)
        {
            const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

            {
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                    regs[i] = Int16VecLoad(source.values + chunkBase + i * registerWidth);
            }

            for (uint32_t j = 0; j < numRemovedFeatures; ++j)
            {
                ASSERT(removedFeatures[j] < NumNetworkInputs);
                const Accumulator& weightsStart = weights[removedFeatures[j]];
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                    regs[i] = Int16VecSub(regs[i], Int16VecLoad(weightsStart.values + chunkBase + i * registerWidth));
            }

            for (uint32_t j = 0; j < numAddedFeatures; ++j)
            {
                ASSERT(addedFeatures[j] < NumNetworkInputs);
                const Accumulator& weightsStart = weights[addedFeatures[j]];
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                    regs[i] = Int16VecAdd(regs[i], Int16VecLoad(weightsStart.values + chunkBase + i * registerWidth));
            }

            {
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                    Int16VecStore(values + chunkBase + i * registerWidth, regs[i]);
            }
        }

#else // no SIMD support
        for (uint32_t i = 0; i < AccumulatorSize; ++i)
            values[i] = source.values[i];

        for (uint32_t j = 0; j < numRemovedFeatures; ++j)
        {
            const uint32_t featureIndex = removedFeatures[j];
            ASSERT(featureIndex < NumNetworkInputs);
            for (uint32_t i = 0; i < AccumulatorSize; ++i)
                values[i] -= weights[featureIndex].values[i];
        }

        for (uint32_t j = 0; j < numAddedFeatures; ++j)
        {
            const uint32_t featureIndex = addedFeatures[j];
            ASSERT(featureIndex < NumNetworkInputs);
            for (uint32_t i = 0; i < AccumulatorSize; ++i)
                values[i] += weights[featureIndex].values[i];
        }
#endif // NN_USE_SIMD
    }

};

struct alignas(CACHELINE_SIZE) LastLayerWeightsBlock
{
    LastLayerWeightType weights[2u * AccumulatorSize];
    LastLayerBiasType bias;
    char __padding[CACHELINE_SIZE - sizeof(LastLayerBiasType)];
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

    // load from file
    bool LoadFromFile(const char* filePath);

    // load from memory
    bool LoadFromMemory(const void* data);

    // save to file
    bool Save(const char* filePath) const;

    // Calculate neural network output based on incrementally updated accumulators
    int32_t Run(const Accumulator& stmAccum, const Accumulator& nstmAccum, uint32_t variant) const;

    // Calculate neural network output based on input
    int32_t Run(const uint16_t* stmFeatures, const uint32_t stmNumFeatures, const uint16_t* nstmFeatures, const uint32_t nstmNumFeatures, uint32_t variant) const;

    INLINE const Accumulator* GetAccumulatorWeights() const { return accumulatorWeights; }
    INLINE const Accumulator& GetAccumulatorBiases() const { return accumulatorBiases; }

    INLINE const LastLayerWeightsBlock& GetLastLayerWeights(uint32_t variant) const { return lastLayerWeights[variant]; }

private:

    Accumulator accumulatorWeights[NumNetworkInputs];
    Accumulator accumulatorBiases;
    LastLayerWeightsBlock lastLayerWeights[NumVariants];
};

} // namespace nn
