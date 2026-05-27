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

static constexpr uint32_t CurrentVersion = 13;
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

// FT CReLU output scale: QA=255 so that clamp to [0,255] fits exactly in uint8
static constexpr int16_t ActivationRangeScaling = 255;

static constexpr int32_t WeightScaleShift = 8;
static constexpr int32_t WeightScale = 1 << WeightScaleShift;

static constexpr int32_t OutputScaleShift = 10;
static constexpr int32_t OutputScale = 1 << OutputScaleShift;

// Hidden layer sizes
static constexpr uint32_t L1Size = 16;
static constexpr uint32_t L2Size = 32;

// QB=64: hidden-layer weight quantization scale (int8 weight = round(w_float * 64))
static constexpr int32_t HiddenWeightScaleShift = 6;
static constexpr int32_t HiddenWeightScale = 1 << HiddenWeightScaleShift; // 64

// The L1 SIMD inference accumulates uint8 input (<=255) * int8 weight via _mm256_maddubs_epi16,
// which adds adjacent pairs of products with int16 saturation. To guarantee no saturation
// (so quantized inference matches the float trainer) the int8 weight magnitude must satisfy
// 2 * 255 * |w| <= INT16_MAX (32767), i.e. |w| <= 64. The trainer clamps L1/L2 weights to this.
static constexpr int32_t MaddubsSafeWeight = 64;

static constexpr float InputLayerWeightQuantizationScale = ActivationRangeScaling;
static constexpr float InputLayerBiasQuantizationScale = ActivationRangeScaling;
// Weight scale QB=64, bias scale QA*QB = 255*64 = 16320
static constexpr float HiddenLayerWeightQuantizationScale = HiddenWeightScale;
static constexpr float HiddenLayerBiasQuantizationScale = HiddenWeightScale * ActivationRangeScaling;
// QB3 = WeightScale*OutputScale/QA ≈ 1028.235f (non-integer is OK for rounding)
// Evaluator divisor = WeightScale*OutputScale = 262144 (unchanged)
static constexpr float OutputLayerWeightQuantizationScale = WeightScale * OutputScale / ActivationRangeScaling;
static constexpr float OutputLayerBiasQuantizationScale = WeightScale * OutputScale;

using FirstLayerWeightType = int16_t;
using FirstLayerBiasType = int16_t;

using HiddenLayerWeightType = int8_t;
using HiddenLayerBiasType = int32_t;

using LastLayerWeightType = int16_t;
using LastLayerBiasType = int32_t;

// CReLU output of the FT is unsigned [0,255], stored as uint8
using IntermediateType = uint8_t;

struct alignas(CACHELINE_SIZE) PackedNeuralNetwork
{
    struct Header
    {
        uint32_t magic   = 0;
        uint32_t version = 0;
        // padding preserves 64-byte (1 cacheline) struct size;
        // old layerSizes[4] and layerVariants[4] fields removed in version 13
        uint32_t padding[14];
    };

    // Per-bucket subnet: L1(16) + L2(32) + L3(32->1), one per output bucket (variant)
    struct alignas(CACHELINE_SIZE) OutputSubnetVariant
    {
        // L1: FT_CReLU[2*AccumulatorSize] → L1[L1Size]
        // Layout: weights[output * 2*AccumulatorSize + input]  (row-major output-first)
        HiddenLayerWeightType l1Weights[2 * AccumulatorSize * L1Size]; // int8[32768]
        HiddenLayerBiasType   l1Biases[L1Size];                        // int32[16]

        // L2: L1[L1Size] → L2[L2Size]
        HiddenLayerWeightType l2Weights[L1Size * L2Size];              // int8[512]
        HiddenLayerBiasType   l2Biases[L2Size];                        // int32[32]

        // L3: L2[L2Size] → scalar output
        LastLayerWeightType   l3Weights[L2Size];                       // int16[32]
        LastLayerBiasType     l3Bias;                                   // int32

        int32_t padding[15]; // pad to 525*64 = 33600 bytes
    };

    // Compile-time size checks
    static_assert(sizeof(Header) == 64,                         "Header must be 64 bytes");
    static_assert(sizeof(OutputSubnetVariant) == 33600,         "OutputSubnetVariant unexpected size");
    static_assert(sizeof(OutputSubnetVariant) % CACHELINE_SIZE == 0, "OutputSubnetVariant must be cacheline-aligned");

    Header header;
    FirstLayerWeightType accumulatorWeights[NumNetworkInputs * AccumulatorSize];
    FirstLayerBiasType   accumulatorBiases[AccumulatorSize];
    OutputSubnetVariant  outputSubnetVariants[NumVariants];

    PackedNeuralNetwork();

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
