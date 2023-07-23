#pragma once

#include "PackedNeuralNetwork.hpp"

namespace nn {

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


    INLINE void Update(
        const Accumulator& source,
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
        ASSERT((size_t)values % 32 == 0);

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
                AccumulatorType* valuesStart = values + chunkBase;
                for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
                {
                    Int16VecStore(valuesStart + i * registerWidth, regs[i]);
                }
            }
        }

#else // no SIMD support
        for (uint32_t i = 0; i < AccumulatorSize; ++i)
        {
            values[i] = source.values[i];
        }
        for (uint32_t j = 0; j < numRemovedFeatures; ++j)
        {
            ASSERT(removedFeatures[j] < NumNetworkInputs);
            const uint32_t weightsDataOffset = removedFeatures[j] * AccumulatorSize;

            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                values[i] -= weights[weightsDataOffset + i];
            }
        }
        for (uint32_t j = 0; j < numAddedFeatures; ++j)
        {
            ASSERT(addedFeatures[j] < NumNetworkInputs);
            const uint32_t weightsDataOffset = addedFeatures[j] * AccumulatorSize;

            for (uint32_t i = 0; i < AccumulatorSize; ++i)
            {
                values[i] += weights[weightsDataOffset + i];
            }
        }
#endif
    }

};

} // namespace nn
