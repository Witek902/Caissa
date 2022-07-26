#include "PackedNeuralNetwork.hpp"
#include "NeuralNetwork.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <intrin.h>

#ifdef USE_AVX2
    #define NN_USE_AVX2
#endif // USE_AVX2

#ifdef USE_SSE2
    #define NN_USE_SSE2
#endif // USE_SSE

#ifdef USE_SSE4
    #define NN_USE_SSE4
#endif // USE_SSE

// assume SSE
constexpr uint32_t OptimalRegisterCount = 8;

namespace nn {

static_assert(sizeof(PackedNeuralNetwork::Header) % CACHELINE_SIZE == 0, "Network header size must be multiple of cacheline size");

using IntermediateType = int8_t;

void Accumulator::Refresh(
    const FirstLayerWeightType* weights, const FirstLayerBiasType* biases,
    uint32_t numInputs, uint32_t numOutputs,
    uint32_t numActiveFeatures, const uint16_t* activeFeatures)
{
    (void)numInputs;
    (void)numOutputs;

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

#if defined(NN_USE_AVX2)

    constexpr uint32_t registerWidth = 256 / (8 * sizeof(FirstLayerWeightType));
    static_assert(FirstLayerSize % registerWidth == 0, "Layer size must be multiple of SIMD register");
    ASSERT(FirstLayerSize == numOutputs);
    ASSERT((size_t)weights % 32 == 0);
    ASSERT((size_t)biases % 32 == 0);
    ASSERT((size_t)values % 32 == 0);

    constexpr uint32_t numChunks = FirstLayerSize / registerWidth;
    static_assert(numChunks % OptimalRegisterCount == 0);
    constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;

    FirstLayerWeightType* valuesStart = values;

    __m256i regs[OptimalRegisterCount];
    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

        {
            //const FirstLayerBiasType* biasesStart = biases + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(biases));
                biases += registerWidth;
            }
        }

        for (uint32_t j = 0; j < numActiveFeatures; ++j)
        {
            ASSERT(activeFeatures[j] < numInputs);
            const FirstLayerWeightType* weightsStart = weights + (chunkBase + activeFeatures[j] * FirstLayerSize);

            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm256_add_epi16(
                    regs[i],
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsStart + i * registerWidth))
                );
            }
        }

        // #TODO keep values as __m256i
        {
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                _mm256_store_si256(reinterpret_cast<__m256i*>(valuesStart), regs[i]);
                valuesStart += registerWidth;
            }
        }
    }

#elif defined(NN_USE_SSE2)

    constexpr uint32_t registerWidth = 128 / (8 * sizeof(FirstLayerWeightType));
    static_assert(FirstLayerSize % registerWidth == 0, "Layer size must be multiple of SIMD register");
    ASSERT(FirstLayerSize == numOutputs);
    ASSERT((size_t)weights % 16 == 0);
    ASSERT((size_t)biases % 16 == 0);

    constexpr uint32_t numChunks = FirstLayerSize / registerWidth;
    static_assert(numChunks % OptimalRegisterCount == 0);
    constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;

    __m128i regs[OptimalRegisterCount];

    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

        {
            const FirstLayerBiasType* biasesStart = biases + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm_load_si128(reinterpret_cast<const __m128i*>(biasesStart + i * registerWidth));
            }
        }

        for (uint32_t j = 0; j < numActiveFeatures; ++j)
        {
            ASSERT(activeFeatures[j] < numInputs);
            const FirstLayerWeightType* weightsStart = weights + (chunkBase + activeFeatures[j] * FirstLayerSize);

            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm_add_epi16(
                    regs[i],
                    _mm_load_si128(reinterpret_cast<const __m128i*>(weightsStart + i * registerWidth))
                );
            }
        }

        // #TODO keep values as __m256i
        {
            FirstLayerWeightType* valuesStart = values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                _mm_store_si128(reinterpret_cast<__m128i*>(valuesStart + i * registerWidth), regs[i]);
            }
        }
    }

#else // no SIMD support

    int32_t regs[FirstLayerSize];

    for (uint32_t i = 0; i < FirstLayerSize; ++i)
    {
        regs[i] = biases[i];
    }

    for (uint32_t j = 0; j < numActiveFeatures; ++j)
    {
        ASSERT(activeFeatures[j] < numInputs);
        const uint32_t weightsDataOffset = activeFeatures[j] * numOutputs;

        for (uint32_t i = 0; i < FirstLayerSize; ++i)
        {
            regs[i] += weights[weightsDataOffset + i];
        }
    }

    for (uint32_t i = 0; i < FirstLayerSize; ++i)
    {
        ASSERT(regs[i] <= std::numeric_limits<FirstLayerWeightType>::max());
        ASSERT(regs[i] >= std::numeric_limits<FirstLayerWeightType>::min());
        values[i] = static_cast<FirstLayerWeightType>(regs[i]);
    }
#endif
}


void Accumulator::Update(
    const Accumulator& source,
    const FirstLayerWeightType* weights,
    uint32_t numInputs, uint32_t numOutputs,
    uint32_t numAddedFeatures, const uint16_t* addedFeatures,
    uint32_t numRemovedFeatures, const uint16_t* removedFeatures)
{
    (void)numInputs;
    (void)numOutputs;
    ASSERT(numOutputs == FirstLayerSize);

#if defined(NN_USE_AVX2)
    constexpr uint32_t registerWidth = 256 / (8 * sizeof(FirstLayerWeightType));
    static_assert(FirstLayerSize % registerWidth == 0, "Layer size must be multiple of SIMD register");
    constexpr uint32_t numChunks = FirstLayerSize / registerWidth;
    static_assert(numChunks % OptimalRegisterCount == 0);
    constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;
    ASSERT((size_t)weights % 32 == 0);
    ASSERT((size_t)source.values % 32 == 0);
    ASSERT((size_t)values % 32 == 0);

    __m256i regs[OptimalRegisterCount];
    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

        {
            const FirstLayerWeightType* valuesStart = source.values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(valuesStart + i * registerWidth));
            }
        }

        for (uint32_t j = 0; j < numRemovedFeatures; ++j)
        {
            ASSERT(removedFeatures[j] < numInputs);
            const FirstLayerWeightType* weightsStart = weights + (chunkBase + removedFeatures[j] * FirstLayerSize);
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm256_sub_epi16(
                    regs[i],
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsStart + i * registerWidth))
                );
            }
        }

        for (uint32_t j = 0; j < numAddedFeatures; ++j)
        {
            ASSERT(addedFeatures[j] < numInputs);
            const FirstLayerWeightType* weightsStart = weights + (chunkBase + addedFeatures[j] * FirstLayerSize);
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm256_add_epi16(
                    regs[i],
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsStart + i * registerWidth))
                );
            }
        }

        {
            FirstLayerWeightType* valuesStart = values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                _mm256_store_si256(reinterpret_cast<__m256i*>(valuesStart + i * registerWidth), regs[i]);
            }
        }
    }

#elif defined(NN_USE_SSE2)
    constexpr uint32_t registerWidth = 256 / (8 * sizeof(FirstLayerWeightType));
    static_assert(FirstLayerSize % registerWidth == 0, "Layer size must be multiple of SIMD register");
    constexpr uint32_t numChunks = FirstLayerSize / registerWidth;
    static_assert(numChunks% OptimalRegisterCount == 0);
    constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;
    ASSERT((size_t)weights % 16 == 0);

    __m128i regs[OptimalRegisterCount];
    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

        {
            const FirstLayerWeightType* valuesStart = source.values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm_load_si128(reinterpret_cast<const __m128i*>(valuesStart + i * registerWidth));
            }
        }

        for (uint32_t j = 0; j < numRemovedFeatures; ++j)
        {
            ASSERT(removedFeatures[j] < numInputs);
            const FirstLayerWeightType* weightsStart = weights + (chunkBase + removedFeatures[j] * FirstLayerSize);
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm_sub_epi16(
                    regs[i],
                    _mm_load_si128(reinterpret_cast<const __m128i*>(weightsStart + i * registerWidth))
                );
            }
        }

        for (uint32_t j = 0; j < numAddedFeatures; ++j)
        {
            ASSERT(addedFeatures[j] < numInputs);
            const FirstLayerWeightType* weightsStart = weights + (chunkBase + addedFeatures[j] * FirstLayerSize);
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm_add_epi16(
                    regs[i],
                    _mm_load_si128(reinterpret_cast<const __m128i*>(weightsStart + i * registerWidth))
                );
            }
        }

        {
            FirstLayerWeightType* valuesStart = values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                _mm_store_si128(reinterpret_cast<__m128i*>(valuesStart + i * registerWidth), regs[i]);
            }
        }
    }
#else // no SIMD support
    for (uint32_t i = 0; i < FirstLayerSize; ++i)
    {
        values[i] = source.values[i];
    }
    for (uint32_t j = 0; j < numRemovedFeatures; ++j)
    {
        ASSERT(removedFeatures[j] < numInputs);
        const uint32_t weightsDataOffset = removedFeatures[j] * numOutputs;

        for (uint32_t i = 0; i < FirstLayerSize; ++i)
        {
            values[i] -= weights[weightsDataOffset + i];
        }
    }
    for (uint32_t j = 0; j < numAddedFeatures; ++j)
    {
        ASSERT(addedFeatures[j] < numInputs);
        const uint32_t weightsDataOffset = addedFeatures[j] * numOutputs;

        for (uint32_t i = 0; i < FirstLayerSize; ++i)
        {
            values[i] += weights[weightsDataOffset + i];
        }
    }
#endif
}

INLINE static void ClippedReLU_16(uint32_t size, IntermediateType* output, const FirstLayerWeightType* input)
{
#if defined(NN_USE_AVX2)
    static_assert(std::is_same_v<FirstLayerWeightType, int16_t>, "Invalid type");
    constexpr uint32_t inRegisterWidth = 256 / 16;
    constexpr uint32_t outRegisterWidth = 256 / 8;
    ASSERT(size % outRegisterWidth == 0);
    const uint32_t numOutChunks = size / outRegisterWidth;
    ASSERT((size_t)output % 32 == 0);
    ASSERT((size_t)input % 32 == 0);

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        const __m256i in0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input));
        input += inRegisterWidth;
        const __m256i in1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input));
        input += inRegisterWidth;

        const __m256i result =
            // packs changes the order, so we need to fix that with a permute
            _mm256_permute4x64_epi64(
                // packs saturates to 127, so we only need to clamp from below
                _mm256_max_epi8(_mm256_packs_epi16(in0, in1), _mm256_setzero_si256()),
                0b11'01'10'00
            );

        _mm256_store_si256(reinterpret_cast<__m256i*>(output), result);
        output += outRegisterWidth;
    }

#elif defined(NN_USE_SSE4)
    static_assert(std::is_same_v<FirstLayerWeightType, int16_t>, "Invalid type");
    constexpr uint32_t inRegisterWidth = 128 / 16;
    constexpr uint32_t outRegisterWidth = 128 / 8;
    ASSERT(size % outRegisterWidth == 0);
    const uint32_t numOutChunks = size / outRegisterWidth;
    ASSERT((size_t)output % 16 == 0);
    ASSERT((size_t)input % 16 == 0);

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        const __m128i in0 = _mm_load_si128(reinterpret_cast<const __m128i*>(input));
        input += inRegisterWidth;
        const __m128i in1 = _mm_load_si128(reinterpret_cast<const __m128i*>(input));
        input += inRegisterWidth;

        // packs saturates to 127, so we only need to clamp from below
        const __m128i result = _mm_max_epi8(_mm_packs_epi16(in0, in1), _mm_setzero_si128());

        _mm_store_si128(reinterpret_cast<__m128i*>(output), result);
        output += outRegisterWidth;
    }

#else // no SIMD support
    for (uint32_t i = 0; i < size; ++i)
    {
        output[i] = (IntermediateType)std::clamp<FirstLayerWeightType>(input[i], 0, std::numeric_limits<IntermediateType>::max());
    }
#endif
}

#ifdef USE_AVX2

INLINE static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b)
{
#if defined (USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
#else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc = _mm256_add_epi32(acc, product0);
#endif
}

INLINE static __m128i m256_haddx4(__m256i a, __m256i b, __m256i c, __m256i d)
{
    a = _mm256_hadd_epi32(a, b);
    c = _mm256_hadd_epi32(c, d);
    a = _mm256_hadd_epi32(a, c);
    const __m128i sum128lo = _mm256_castsi256_si128(a);
    const __m128i sum128hi = _mm256_extracti128_si256(a, 1);
    return _mm_add_epi32(sum128lo, sum128hi);
}

INLINE static int32_t m256_hadd(__m256i a)
{
    const __m256i sum1 = _mm256_hadd_epi32(a, a);
    const __m256i sum2 = _mm256_hadd_epi32(sum1, sum1);
    const __m128i sum3 = _mm256_extracti128_si256(sum2, 1);
    return _mm_cvtsi128_si32(_mm_add_epi32(_mm256_castsi256_si128(sum2), sum3));
}

#endif // USE_AVX2

#ifdef USE_SSE4

INLINE static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b)
{
#if defined (USE_VNNI)
    acc = _mm_dpbusd_epi32(acc, a, b);
#else
    __m128i product0 = _mm_maddubs_epi16(a, b);
    product0 = _mm_madd_epi16(product0, _mm_set1_epi16(1));
    acc = _mm_add_epi32(acc, product0);
#endif
}

INLINE static __m128i m128_haddx4(__m128i a, __m128i b, __m128i c, __m128i d)
{
    return _mm_hadd_epi32(_mm_hadd_epi32(a, b), _mm_hadd_epi32(c, d));
}

INLINE static int32_t m128_hadd(__m128i a)
{
    a = _mm_hadd_epi32(a, a);
    a = _mm_hadd_epi32(a, a);
    return _mm_cvtsi128_si32(a);
}

#endif // USE_SSE4

NO_INLINE static void LinearLayer(
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    uint32_t numInputs, uint32_t numOutputs, int32_t* output, const IntermediateType* input)
{
#if defined(NN_USE_AVX2)
    constexpr uint32_t registerWidth = 256 / 8;
    const uint32_t numOutChunks = numOutputs / 4u;
    ASSERT(numInputs % registerWidth == 0);
    ASSERT(numOutputs % 4u == 0);
    ASSERT((size_t)weights % 32 == 0);
    ASSERT((size_t)biases % 32 == 0);
    ASSERT((size_t)output % 32 == 0);
    ASSERT((size_t)input % 32 == 0);

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        // Prepare weight offsets. One offset for one row of weights.
        // This is a simple index into a 2d array.
        const uint32_t offset0 = (i * 4u + 0u) * numInputs;
        const uint32_t offset1 = (i * 4u + 1u) * numInputs;
        const uint32_t offset2 = (i * 4u + 2u) * numInputs;
        const uint32_t offset3 = (i * 4u + 3u) * numInputs;

        // Accumulation starts from 0, we add the bias only at the end.
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
        for (uint32_t j = 0; j < numInputs; j += registerWidth)
        {
            // We unroll by 4 so that we can reuse this value, reducing the number of memory operations required.
            const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + j));

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            const HiddenLayerWeightType* weightsBase = weights + j;
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsBase + offset0)));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsBase + offset1)));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsBase + offset2)));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weightsBase + offset3)));
        }

        const __m128i bias = _mm_load_si128(reinterpret_cast<const __m128i*>(&biases[i * 4u]));
        // This function adds horizontally 8 values from each sum together, producing 4 int32 values.
        __m128i outVal = m256_haddx4(sum0, sum1, sum2, sum3);
        outVal = _mm_add_epi32(outVal, _mm_set1_epi32(WeightScale / 2)); // divide with rounding to nearest
        outVal = _mm_add_epi32(outVal, bias);
        outVal = _mm_srai_epi32(outVal, WeightScaleShift);
        _mm_store_si128(reinterpret_cast<__m128i*>(&output[i * 4]), outVal);
    }

#elif defined(NN_USE_SSE4)
    constexpr uint32_t registerWidth = 128 / 8;
    const uint32_t numOutChunks = numOutputs / 4u;
    ASSERT(numInputs % registerWidth == 0);
    ASSERT(numOutputs % 4u == 0);
    ASSERT((size_t)weights % 16 == 0);
    ASSERT((size_t)biases % 16 == 0);
    ASSERT((size_t)output % 16 == 0);
    ASSERT((size_t)input % 16 == 0);

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        // Prepare weight offsets. One offset for one row of weights.
        // This is a simple index into a 2d array.
        const uint32_t offset0 = (i * 4u + 0u) * numInputs;
        const uint32_t offset1 = (i * 4u + 1u) * numInputs;
        const uint32_t offset2 = (i * 4u + 2u) * numInputs;
        const uint32_t offset3 = (i * 4u + 3u) * numInputs;

        // Accumulation starts from 0, we add the bias only at the end.
        __m128i sum0 = _mm_setzero_si128();
        __m128i sum1 = _mm_setzero_si128();
        __m128i sum2 = _mm_setzero_si128();
        __m128i sum3 = _mm_setzero_si128();

        // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
        for (uint32_t j = 0; j < numInputs; j += registerWidth)
        {
            // We unroll by 4 so that we can reuse this value, reducing the number of memory operations required.
            const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(input + j));

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            const HiddenLayerWeightType* weightsBase = weights + j;
            m128_add_dpbusd_epi32(sum0, in, _mm_load_si128(reinterpret_cast<const __m128i*>(weightsBase + offset0)));
            m128_add_dpbusd_epi32(sum1, in, _mm_load_si128(reinterpret_cast<const __m128i*>(weightsBase + offset1)));
            m128_add_dpbusd_epi32(sum2, in, _mm_load_si128(reinterpret_cast<const __m128i*>(weightsBase + offset2)));
            m128_add_dpbusd_epi32(sum3, in, _mm_load_si128(reinterpret_cast<const __m128i*>(weightsBase + offset3)));
        }

        const __m128i bias = _mm_load_si128(reinterpret_cast<const __m128i*>(&biases[i * 4u]));
        // This function adds horizontally 8 values from each sum together, producing 4 int32 values.
        __m128i outVal = m128_haddx4(sum0, sum1, sum2, sum3);
        outVal = _mm_add_epi32(outVal, _mm_set1_epi32(WeightScale / 2)); // divide with rounding to nearest
        outVal = _mm_add_epi32(outVal, bias);
        outVal = _mm_srai_epi32(outVal, WeightScaleShift);
        _mm_store_si128(reinterpret_cast<__m128i*>(&output[i * 4]), outVal);
    }
#else // no SIMD support
    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        int32_t val = biases[i];
        for (uint32_t j = 0; j < numInputs; ++j)
        {
            val += weights[i * numInputs + j] * (int32_t)input[j];
        }
        // divide with rounding to nearest
        output[i] = (val + (WeightScale / 2)) >> WeightScaleShift;
    }
#endif
}

INLINE static void ClippedReLU_32(uint32_t size, IntermediateType* output, const int32_t* input)
{
#if defined(NN_USE_AVX2)
    constexpr uint32_t inRegisterWidth = 256 / 32;
    constexpr uint32_t outRegisterWidth = 256 / 8;
    ASSERT(size % outRegisterWidth == 0);
    const uint32_t numOutChunks = size / outRegisterWidth;
    ASSERT((size_t)output % 32 == 0);
    ASSERT((size_t)input % 32 == 0);

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        __m256i in0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input)); input += inRegisterWidth;
        __m256i in1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input)); input += inRegisterWidth;
        __m256i in2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input)); input += inRegisterWidth;
        __m256i in3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input)); input += inRegisterWidth;

        in0 = _mm256_packs_epi32(in0, in1);
        in1 = _mm256_packs_epi32(in2, in3);

        const __m256i result =
            _mm256_permutevar8x32_epi32(
                // packs saturates to 127, so we only need to clamp from below
                _mm256_max_epi8(
                    _mm256_packs_epi16(in0, in1),
                    _mm256_setzero_si256()
                ),
                _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0)
            );

        _mm256_store_si256(reinterpret_cast<__m256i*>(output), result);
        output += outRegisterWidth;
    }

#elif defined(NN_USE_SSE4)
    constexpr uint32_t inRegisterWidth = 128 / 32;
    constexpr uint32_t outRegisterWidth = 128 / 8;
    ASSERT(size % outRegisterWidth == 0);
    const uint32_t numOutChunks = size / outRegisterWidth;
    ASSERT((size_t)output % 16 == 0);
    ASSERT((size_t)input % 16 == 0);

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        __m128i in0 = _mm_load_si128(reinterpret_cast<const __m128i*>(input)); input += inRegisterWidth;
        __m128i in1 = _mm_load_si128(reinterpret_cast<const __m128i*>(input)); input += inRegisterWidth;
        __m128i in2 = _mm_load_si128(reinterpret_cast<const __m128i*>(input)); input += inRegisterWidth;
        __m128i in3 = _mm_load_si128(reinterpret_cast<const __m128i*>(input)); input += inRegisterWidth;

        in0 = _mm_packs_epi32(in0, in1);
        in1 = _mm_packs_epi32(in2, in3);

        // packs saturates to 127, so we only need to clamp from below
        const __m128i result = _mm_max_epi8(_mm_packs_epi16(in0, in1), _mm_setzero_si128());

        _mm_store_si128(reinterpret_cast<__m128i*>(output), result);
        output += outRegisterWidth;
    }

#else // no SIMD support
    for (uint32_t i = 0; i < size; ++i)
    {
        output[i] = (IntermediateType)std::clamp<int32_t>(input[i], 0, std::numeric_limits<IntermediateType>::max());
    }
#endif
}

INLINE static int32_t LinearLayer_SingleOutput(
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    uint32_t numInputs, uint32_t numOutputs,
    const IntermediateType* input)
{
    (void)numInputs;
    (void)numOutputs;
    ASSERT(numOutputs == 1);

    int32_t val = biases[0];

#if defined(NN_USE_AVX2)
    constexpr uint32_t registerWidth = 256 / 8;
    ASSERT(numInputs % registerWidth == 0);
    ASSERT((size_t)weights % 32 == 0);
    ASSERT((size_t)biases % 32 == 0);

    __m256i sum = _mm256_setzero_si256();
    for (uint32_t j = 0; j < numInputs; j += registerWidth)
    {
        const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + j));
        m256_add_dpbusd_epi32(sum, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + j)));
    }

    // add 8 int32s horizontally
    val += m256_hadd(sum);

#elif defined(NN_USE_SSE4)
    constexpr uint32_t registerWidth = 128 / 8;
    ASSERT(numInputs % registerWidth == 0);
    ASSERT((size_t)weights % 16 == 0);
    ASSERT((size_t)biases % 16 == 0);

    __m128i sum = _mm_setzero_si128();
    for (uint32_t j = 0; j < numInputs; j += registerWidth)
    {
        const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(input + j));
        m128_add_dpbusd_epi32(sum, in, _mm_load_si128(reinterpret_cast<const __m128i*>(weights + j)));
    }

    // add 8 int32s horizontally
    val += m128_hadd(sum);

#else

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        val += (int32_t)input[i] * (int32_t)weights[i];
    }

#endif

    // divide with rounding to nearest
    return (val + (WeightScale / 2)) >> WeightScaleShift;
}

///

PackedNeuralNetwork::PackedNeuralNetwork()
{
}

PackedNeuralNetwork::~PackedNeuralNetwork()
{


    Release();
}

void PackedNeuralNetwork::Release()
{
    if (mappedData)
    {
        ReleaseFileMapping();
        weightsBuffer = nullptr;
    }

    if (weightsBuffer)
    {
        AlignedFree(weightsBuffer);
        weightsBuffer = nullptr;
    }

    header = Header{};
}

size_t PackedNeuralNetwork::GetWeightsBufferSize() const
{
    size_t size = 0;

    size += header.layerSizes[0] * header.layerSizes[1] * sizeof(FirstLayerWeightType);
    size += header.layerSizes[1] * sizeof(FirstLayerBiasType);

    size += header.layerSizes[1] * header.layerSizes[2] * sizeof(HiddenLayerWeightType);
    size += header.layerSizes[2] * sizeof(HiddenLayerBiasType);

    size += header.layerSizes[2] * header.layerSizes[3] * sizeof(HiddenLayerWeightType);
    size += header.layerSizes[3] * sizeof(HiddenLayerBiasType);

    size += header.layerSizes[3] * OutputSize * sizeof(HiddenLayerWeightType);
    size += OutputSize * sizeof(HiddenLayerBiasType);

    return size;
}

bool PackedNeuralNetwork::Resize(uint32_t numInputs, uint32_t l1size, uint32_t l2size, uint32_t l3size)
{
    Release();

    header.magic = MagicNumber;
    header.version = CurrentVersion;
    header.layerSizes[0] = numInputs;
    header.layerSizes[1] = l1size;
    header.layerSizes[2] = l2size;
    header.layerSizes[3] = l3size;

    const size_t weightsSize = GetWeightsBufferSize();
    weightsBuffer = (uint8_t*)AlignedMalloc(weightsSize, CACHELINE_SIZE);

    if (!weightsBuffer)
    {
        Release();
        std::cerr << "Failed to allocate weights buffer" << std::endl;
        return false;
    }

    return true;
}

bool PackedNeuralNetwork::Save(const char* filePath) const
{
    if (!IsValid())
    {
        std::cerr << "Failed to save neural network: " << "invalid network" << std::endl;
        return false;
    }

    FILE* file = fopen(filePath, "wb");
    if (!file)
    {
        std::cerr << "Failed to save neural network: " << "cannot open file" << std::endl;
        return false;
    }

    if (1 != fwrite(&header, sizeof(Header), 1, file))
    {
        fclose(file);
        std::cerr << "Failed to save neural network: " << "cannot write header" << std::endl;
        return false;
    }
    
    if (1 != fwrite(weightsBuffer, GetWeightsBufferSize(), 1, file))
    {
        fclose(file);
        std::cerr << "Failed to save neural network: " << "cannot write weights" << std::endl;
        return false;
    }

    fclose(file);
    return true;
}

void PackedNeuralNetwork::ReleaseFileMapping()
{
    if (fileMapping == INVALID_HANDLE_VALUE)
    {
        CloseHandle(fileMapping);
        fileMapping = INVALID_HANDLE_VALUE;
    }

    if (fileHandle == INVALID_HANDLE_VALUE)
    {
        CloseHandle(fileHandle);
        fileHandle = INVALID_HANDLE_VALUE;
    }

    mappedData = nullptr;
}

bool PackedNeuralNetwork::Load(const char* filePath)
{
    Release();
    
    // open file
    {
#ifdef _UNICODE
        wchar_t wideFilePath[4096];
        size_t len = 0;
        mbstowcs_s(&len, wideFilePath, 4096, filePath, _TRUNCATE);
        fileHandle = ::CreateFile(wideFilePath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#else
        fileHandle = ::CreateFile(filePath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#endif

        if (fileHandle == INVALID_HANDLE_VALUE)
        {
            fprintf(stderr, "CreateFile() failed, error = %lu.\n", GetLastError());
            return false;
        }
    }

    DWORD sizeLow = 0, sizeHigh = 0;
    sizeLow = ::GetFileSize(fileHandle, &sizeHigh);
    fileMapping = ::CreateFileMapping(fileHandle, NULL, PAGE_READONLY, sizeHigh, sizeLow, NULL);
    if (fileMapping == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "CreateFileMapping() failed, error = %lu.\n", GetLastError());
        Release();
        return false;
    }

    mappedData = (void*)MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
    if (mappedData == nullptr)
    {
        fprintf(stderr, "MapViewOfFile() failed, error = %lu.\n", GetLastError());
        Release();
        return false;
    }

    memcpy(&header, mappedData, sizeof(Header));

    if (sizeof(Header) + GetWeightsBufferSize() > (sizeLow + ((uint64_t)sizeHigh << 32)))
    {
        std::cerr << "Failed to load neural network: " << "file it too small" << std::endl;
        Release();
        return false;
    }

    if (header.magic != MagicNumber)
    {
        std::cerr << "Failed to load neural network: " << "invalid magic" << std::endl;
        Release();
        return false;
    }

    if (header.version != CurrentVersion)
    {
        std::cerr << "Failed to load neural network: " << "unsupported version" << std::endl;
        Release();
        return false;
    }

    if (header.layerSizes[0] == 0 || header.layerSizes[0] > MaxInputs)
    {
        std::cerr << "Failed to load neural network: " << "invalid number of inputs" << std::endl;
        Release();
        return false;
    }

    for (uint32_t i = 1; i < MaxNumLayers; ++i)
    {
        const uint32_t maxNeurons = i > 1 ? MaxNeuronsInLaterLayers : MaxNeuronsInFirstLayer;
        if ((header.layerSizes[i] == 0) || (header.layerSizes[i] % MinNeuronsInLaterLayers != 0) || (header.layerSizes[i] > maxNeurons))
        {
            std::cerr << "Failed to load neural network: " << "invalid number of inputs" << std::endl;
            Release();
            return false;
        }
    }

    weightsBuffer = (uint8_t*)mappedData + sizeof(Header);

    return true;
}

int32_t PackedNeuralNetwork::Run(const Accumulator& accumulator) const
{
    ASSERT(GetLayerSize(1) <= MaxNeuronsInFirstLayer);
    ASSERT(GetLayerSize(2) <= MaxNeuronsInLaterLayers);
    ASSERT(GetLayerSize(3) <= MaxNeuronsInLaterLayers);

    alignas(CACHELINE_SIZE) IntermediateType tempA[MaxNeuronsInFirstLayer];
    alignas(CACHELINE_SIZE) int32_t tempB[MaxNeuronsInLaterLayers];

    ClippedReLU_16(GetLayerSize(1), tempA, accumulator.values);

    LinearLayer(GetLayer1Weights(), GetLayer1Biases(), GetLayerSize(1), GetLayerSize(2), tempB, tempA);
    ClippedReLU_32(GetLayerSize(2), tempA, tempB);

    LinearLayer(GetLayer2Weights(), GetLayer2Biases(), GetLayerSize(2), GetLayerSize(3), tempB, tempA);
    ClippedReLU_32(GetLayerSize(3), tempA, tempB);

    return LinearLayer_SingleOutput(GetLayer3Weights(), GetLayer3Biases(), GetLayerSize(3), 1, tempA);
}

int32_t PackedNeuralNetwork::Run(const uint16_t* activeInputIndices, const uint32_t numActiveInputs) const
{
    Accumulator accumulator;
    accumulator.Refresh(GetAccumulatorWeights(), GetAccumulatorBiases(), GetNumInputs(), GetLayerSize(1), numActiveInputs, activeInputIndices);

    return Run(accumulator);
}

} // namespace nn
