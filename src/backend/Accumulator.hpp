#pragma once

#include "PackedNeuralNetwork.hpp"

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

using AccumulatorType = int16_t;

struct alignas(CACHELINE_SIZE) Accumulator
{
    AccumulatorType values[AccumulatorSize];

    INLINE void Refresh(
        const FirstLayerWeightType* weights, const FirstLayerBiasType* biases,
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

#if defined(NN_USE_AVX512) || defined(NN_USE_AVX2) || defined(NN_USE_SSE2) || defined(NN_USE_ARM_NEON)

        constexpr uint32_t registerWidth = VectorRegSize / (8 * sizeof(AccumulatorType));
        static_assert(AccumulatorSize % registerWidth == 0);
        ASSERT((size_t)weights % 32 == 0);
        ASSERT((size_t)biases % 32 == 0);
        ASSERT((size_t)values % 32 == 0);

        constexpr uint32_t numChunks = AccumulatorSize / registerWidth;
        static_assert(numChunks % OptimalRegisterCount == 0, "");
        constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;

        AccumulatorType* valuesStart = values;

        Int16VecType regs[OptimalRegisterCount];
        for (uint32_t tile = 0; tile < numTiles; ++tile)
        {
            const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = Int16VecLoad(biases);
                biases += registerWidth;
            }

            for (uint32_t j = 0; j < numActiveFeatures; ++j)
            {
                ASSERT(activeFeatures[j] < NumNetworkInputs);
                const FirstLayerWeightType* weightsStart = weights + (chunkBase + activeFeatures[j] * AccumulatorSize);
                ASSERT((size_t)weightsStart % 32 == 0); // make sure loads are aligned

                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    regs[i] = Int16VecAdd(regs[i], Int16VecLoad(weightsStart + i * registerWidth));
                }
            }

            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                Int16VecStore(valuesStart, regs[i]);
                valuesStart += registerWidth;
            }
        }

#else // no SIMD support

        int16_t regs[AccumulatorSize];

        for (uint32_t i = 0; i < AccumulatorSize; ++i)
        {
            regs[i] = biases[i];
        }

        for (uint32_t j = 0; j < numActiveFeatures; ++j)
        {
            const uint32_t weightsDataOffset = activeFeatures[j] * AccumulatorSize;

            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                ASSERT(int32_t(regs[i]) + int32_t(weights[weightsDataOffset + i]) <= std::numeric_limits<AccumulatorType>::max());
                ASSERT(int32_t(regs[i]) + int32_t(weights[weightsDataOffset + i]) >= std::numeric_limits<AccumulatorType>::min());

                regs[i] += weights[weightsDataOffset + i];
            }
        }

        for (uint32_t i = 0; i < AccumulatorSize; ++i)
        {
            values[i] = static_cast<AccumulatorType>(regs[i]);
        }
#endif
    }

    template<bool WithExtraTarget>
    INLINE static void UpdateImpl(Accumulator& target, Accumulator* extraTarget, const Accumulator& source,
        const FirstLayerWeightType* weights,
        uint32_t numAddedFeatures, const uint16_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint16_t* removedFeatures)
    {
#if defined(NN_USE_AVX512) || defined(NN_USE_AVX2) || defined(NN_USE_SSE2) || defined(NN_USE_ARM_NEON)

        constexpr uint32_t registerWidth = VectorRegSize / (8 * sizeof(AccumulatorType));
        static_assert(AccumulatorSize % registerWidth == 0);
        const uint32_t numChunks = AccumulatorSize / registerWidth;
        static_assert(numChunks % OptimalRegisterCount == 0);
        const uint32_t numTiles = numChunks / OptimalRegisterCount;
        ASSERT((size_t)weights % 32 == 0);
        ASSERT((size_t)source.values % 32 == 0);
        ASSERT((size_t)target.values % 32 == 0);
        if constexpr (WithExtraTarget) ASSERT((size_t)extraTarget->values % 32 == 0);

        Int16VecType regs[OptimalRegisterCount];
        for (uint32_t tile = 0; tile < numTiles; ++tile)
        {
            const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

            {
                const AccumulatorType* valuesStart = source.values + chunkBase;
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    regs[i] = Int16VecLoad(valuesStart + i * registerWidth);
                }
            }

            for (uint32_t j = 0; j < numRemovedFeatures; ++j)
            {
                ASSERT(removedFeatures[j] < NumNetworkInputs);
                const FirstLayerWeightType* weightsStart = weights + (chunkBase + removedFeatures[j] * AccumulatorSize);
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    regs[i] = Int16VecSub(regs[i], Int16VecLoad(weightsStart + i * registerWidth));
                }
            }

            for (uint32_t j = 0; j < numAddedFeatures; ++j)
            {
                ASSERT(addedFeatures[j] < NumNetworkInputs);
                const FirstLayerWeightType* weightsStart = weights + (chunkBase + addedFeatures[j] * AccumulatorSize);
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    regs[i] = Int16VecAdd(regs[i], Int16VecLoad(weightsStart + i * registerWidth));
                }
            }

            {
                AccumulatorType* valuesStart = target.values + chunkBase;
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    Int16VecStore(valuesStart + i * registerWidth, regs[i]);
                }
            }

            if constexpr (WithExtraTarget)
            {
                AccumulatorType* extraValuesStart = extraTarget->values + chunkBase;
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    Int16VecStore(extraValuesStart + i * registerWidth, regs[i]);
                }
            }
        }

#else // no SIMD support
        for (uint32_t i = 0; i < AccumulatorSize; ++i)
        {
            target.values[i] = source.values[i];
        }
        for (uint32_t j = 0; j < numRemovedFeatures; ++j)
        {
            ASSERT(removedFeatures[j] < NumNetworkInputs);
            const uint32_t weightsDataOffset = removedFeatures[j] * AccumulatorSize;

            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                target.values[i] -= weights[weightsDataOffset + i];
            }
        }
        for (uint32_t j = 0; j < numAddedFeatures; ++j)
        {
            ASSERT(addedFeatures[j] < NumNetworkInputs);
            const uint32_t weightsDataOffset = addedFeatures[j] * AccumulatorSize;

            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                target.values[i] += weights[weightsDataOffset + i];
            }
        }
        if constexpr (WithExtraTarget)
        {
            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                extraTarget->values[i] = target.values[i];
            }
        }
#endif
    }

    INLINE static void Update(Accumulator& target, const Accumulator& source,
        const FirstLayerWeightType* weights,
        uint32_t numAddedFeatures, const uint16_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint16_t* removedFeatures)
    {
        UpdateImpl<false>(target, nullptr, source, weights, numAddedFeatures, addedFeatures, numRemovedFeatures, removedFeatures);
    }

    INLINE static void Update(Accumulator& target, Accumulator& extraTarget, const Accumulator& source,
        const FirstLayerWeightType* weights,
        uint32_t numAddedFeatures, const uint16_t* addedFeatures,
        uint32_t numRemovedFeatures, const uint16_t* removedFeatures)
    {
        UpdateImpl<true>(target, &extraTarget, source, weights, numAddedFeatures, addedFeatures, numRemovedFeatures, removedFeatures);
    }
};

} // namespace nn
