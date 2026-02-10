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

#if defined(NN_USE_AVX512)
    constexpr uint32_t OptimalRegisterCount = 16;
#elif defined(NN_USE_AVX2) || defined(NN_USE_SSE2) || defined(NN_USE_ARM_NEON)
    constexpr uint32_t OptimalRegisterCount = 8;
#endif // NN_USE_AVX512 || NN_USE_AVX2 || NN_USE_SSE2 || NN_USE_ARM_NEON


class NeuralNetwork;
struct Accumulator;

static constexpr uint32_t CurrentVersion = 12;
static constexpr uint32_t MagicNumber = 'CSNN';

static constexpr uint32_t NumKingBuckets = 32;
static constexpr uint32_t NumNetworkInputs = NumKingBuckets * 12 * 64;
static constexpr uint32_t AccumulatorSize = 1024;
static constexpr uint32_t OutputSize = 1;
static constexpr uint32_t NumVariants = 8;

static constexpr uint8_t KingBucketIndex[64] =
{
     0, 1, 2, 3,  3, 2, 1, 0,
     4, 5, 6, 7,  7, 6, 5, 4,
     8, 9,10,11, 11,10, 9, 8,
    12,13,14,15, 15,14,13,12,
    16,17,18,19, 19,18,17,16,
    20,21,22,23, 23,22,21,20,
    24,25,26,27, 27,26,25,24,
    28,29,30,31, 31,30,29,28,
};

// by this value neuron inputs are scaled
static constexpr int16_t ActivationRangeScaling = 256;

static constexpr int32_t WeightScaleShift = 8;
static constexpr int32_t WeightScale = 1 << WeightScaleShift;

static constexpr int32_t OutputScaleShift = 10;
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

using LastLayerWeightType = int16_t;
using LastLayerBiasType = int32_t;

using IntermediateType = int8_t;

struct alignas(CACHELINE_SIZE) PackedNeuralNetwork
{
    struct Header
    {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t layerSizes[4] = { 0, 0, 0, 0 };
        uint32_t layerVariants[4] = { 0, 0, 0, 0 };
        uint32_t padding[6];
    };

    struct alignas(CACHELINE_SIZE) LastLayerVariant
    {
        LastLayerWeightType weights[2 * AccumulatorSize];
        LastLayerBiasType bias;
        int32_t padding[15];
    };

    Header header;
    FirstLayerWeightType accumulatorWeights[NumNetworkInputs * AccumulatorSize];
    FirstLayerBiasType accumulatorBiases[AccumulatorSize];
    LastLayerVariant lastLayerVariants[NumVariants];

    // load from file
    bool LoadFromFile(const char* filePath);

    // save to file
    bool SaveToFile(const char* filePath) const;

    // Calculate neural network output based on incrementally updated accumulators
    int32_t Run(const Accumulator& stmAccum, const Accumulator& nstmAccum, uint32_t variant) const;

    // Calculate neural network output based on input
    int32_t Run(const uint16_t* stmFeatures, const uint32_t stmNumFeatures, const uint16_t* nstmFeatures, const uint32_t nstmNumFeatures, uint32_t variant) const;
};

} // namespace nn
