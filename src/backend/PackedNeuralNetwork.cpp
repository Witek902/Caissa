#include "PackedNeuralNetwork.hpp"
#include "NeuralNetwork.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_AVX2
    #include <intrin.h>
    #define NN_USE_AVX2
#endif // USE_AVX2

#if defined(NN_USE_AVX2)
constexpr uint32_t OptimalRegisterCount = 16;
#elif defined(NN_USE_SSE)
constexpr uint32_t OptimalRegisterCount = 8;
#else
constexpr uint32_t OptimalRegisterCount = 1;
#endif

namespace nn {

using IntermediateType = int8_t;


void Accumulator::Refresh(
    const FirstLayerWeightType* weights, const FirstLayerBiasType* biases,
    uint32_t numInputs, uint32_t numOutputs,
    uint32_t numActiveFeatures, const uint16_t* activeFeatures)
{
    (void)numInputs;
    (void)numOutputs;

#ifdef NN_USE_AVX2
    constexpr uint32_t registerWidth = 256 / 16;
    static_assert(FirstLayerSize % registerWidth == 0, "We're processing 16 elements at a time");
    ASSERT(FirstLayerSize == numOutputs);

    constexpr uint32_t numChunks = FirstLayerSize / registerWidth;
    static_assert(numChunks % OptimalRegisterCount == 0);
    constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;

    __m256i regs[OptimalRegisterCount];

    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

        {
            const FirstLayerBiasType* biasesStart = biases + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                regs[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(biasesStart + i * registerWidth));
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
            FirstLayerWeightType* valuesStart = values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                _mm256_store_si256(reinterpret_cast<__m256i*>(valuesStart + i * registerWidth), regs[i]);
            }
        }
    }

#else
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

#ifdef NN_USE_AVX2
    constexpr uint32_t registerWidth = 256 / 16;
    static_assert(FirstLayerSize % registerWidth == 0, "We're processing 16 elements at a time");

    constexpr uint32_t numChunks = FirstLayerSize / registerWidth;
    static_assert(numChunks % OptimalRegisterCount == 0);
    constexpr uint32_t numTiles = numChunks / OptimalRegisterCount;

    __m256i regs[OptimalRegisterCount];

    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * OptimalRegisterCount * registerWidth;

        // #TODO keep values as __m256i
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

        // #TODO keep values as __m256i
        {
            FirstLayerWeightType* valuesStart = values + chunkBase;
            for (uint32_t i = 0; i < OptimalRegisterCount; ++i)
            {
                _mm256_store_si256(reinterpret_cast<__m256i*>(valuesStart + i * registerWidth), regs[i]);
            }
        }
    }

#else
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
#ifdef NN_USE_AVX2
    static_assert(std::is_same_v<FirstLayerWeightType, int16_t>, "Invalid type");
    constexpr uint32_t inRegisterWidth = 256 / 16;
    constexpr uint32_t outRegisterWidth = 256 / 8;
    ASSERT(size % outRegisterWidth == 0);
    const uint32_t numOutChunks = size / outRegisterWidth;

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        const __m256i in0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&input[inRegisterWidth * (i * 2 + 0)]));
        const __m256i in1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&input[inRegisterWidth * (i * 2 + 1)]));

        const __m256i result =
            // packs changes the order, so we need to fix that with a permute
            _mm256_permute4x64_epi64(
                // packs saturates to 127, so we only need to clamp from below
                _mm256_max_epi8(
                    _mm256_packs_epi16(in0, in1),
                    _mm256_setzero_si256()
                ),
                0b11011000
            );

        _mm256_store_si256(reinterpret_cast<__m256i*>(&output[i * outRegisterWidth]), result);
    }
#else
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

INLINE static void LinearLayer(
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    uint32_t numInputs, uint32_t numOutputs, int32_t* output, const IntermediateType* input)
{
#ifdef NN_USE_AVX2
    constexpr uint32_t registerWidth = 256 / 8;
    ASSERT(numInputs % registerWidth == 0);
    ASSERT(numOutputs % 4u == 0);
    const uint32_t numOutChunks = numOutputs / 4u;

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
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + offset0 + j)));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + offset1 + j)));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + offset2 + j)));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + offset3 + j)));
        }

        const __m128i bias = _mm_load_si128(reinterpret_cast<const __m128i*>(&biases[i * 4u]));
        // This function adds horizontally 8 values from each sum together, producing 4 int32 values.
        __m128i outVal = m256_haddx4(sum0, sum1, sum2, sum3);
        outVal = _mm_add_epi32(outVal, _mm_set1_epi32(WeightScale / 2)); // divide with rounding to nearest
        outVal = _mm_add_epi32(outVal, bias);
        outVal = _mm_srai_epi32(outVal, WeightScaleShift);
        _mm_store_si128(reinterpret_cast<__m128i*>(&output[i * 4]), outVal);
    }
#else
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
#ifdef NN_USE_AVX2
    constexpr uint32_t inRegisterWidth = 256 / 32;
    constexpr uint32_t outRegisterWidth = 256 / 8;
    ASSERT(size % outRegisterWidth == 0);
    const uint32_t numOutChunks = size / outRegisterWidth;

    for (uint32_t i = 0; i < numOutChunks; ++i)
    {
        const __m256i in0 =
            _mm256_packs_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(&input[inRegisterWidth * (i * 4 + 0)])),
                _mm256_load_si256(reinterpret_cast<const __m256i*>(&input[inRegisterWidth * (i * 4 + 1)]))
            );

        const __m256i in1 =
            _mm256_packs_epi32(
                _mm256_load_si256(reinterpret_cast<const __m256i*>(&input[inRegisterWidth * (i * 4 + 2)])),
                _mm256_load_si256(reinterpret_cast<const __m256i*>(&input[inRegisterWidth * (i * 4 + 3)]))
            );

        const __m256i result =
            _mm256_permutevar8x32_epi32(
                // packs saturates to 127, so we only need to clamp from below
                _mm256_max_epi8(
                    _mm256_packs_epi16(in0, in1),
                    _mm256_setzero_si256()
                ),
                _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0)
            );

        _mm256_store_si256(reinterpret_cast<__m256i*>(&output[i * outRegisterWidth]), result);
    }
#else
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

#ifdef NN_USE_AVX2
    constexpr uint32_t registerWidth = 256 / 8;
    ASSERT(numInputs % registerWidth == 0);

    // Accumulation starts from 0, we add the bias only at the end.
    __m256i sum = _mm256_setzero_si256();

    for (uint32_t j = 0; j < numInputs; j += registerWidth)
    {
        const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + j));
        // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
        m256_add_dpbusd_epi32(sum, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + j)));
    }

    // add 8 int32s horizontally
    val += m256_hadd(sum);
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
    header.version = CurrentVersion;
    header.firstLayerSize = FirstLayerSize;
    header.secondLayerSize = SecondLayerSize;
    header.thirdLayerSize = ThirdLayerSize;
}

PackedNeuralNetwork::~PackedNeuralNetwork()
{
    if (layer0_weights)
    {
        AlignedFree(layer0_weights);
        layer0_weights = nullptr;
    }
}

bool PackedNeuralNetwork::Save(const char* filePath) const
{
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
    
    const size_t layer0WeightsSize = header.numInputs * FirstLayerSize * sizeof(FirstLayerWeightType);
    if (1 != fwrite(layer0_weights, layer0WeightsSize, 1, file))
    {
        fclose(file);
        std::cerr << "Failed to save neural network: " << "cannot write weights" << std::endl;
        return false;
    }

    const size_t dataSize = sizeof(layer0_biases) +
        sizeof(layer1_weights) + sizeof(layer1_biases) +
        sizeof(layer2_weights) + sizeof(layer2_biases) +
        sizeof(layer3_weights) + sizeof(layer3_biases);
    if (1 != fwrite(layer0_biases, dataSize, 1, file))
    {
        fclose(file);
        std::cerr << "Failed to save neural network: " << "cannot write weights" << std::endl;
        return false;
    }

    fclose(file);
    return true;
}

bool PackedNeuralNetwork::Load(const char* filePath)
{
    FILE* file = fopen(filePath, "rb");
    if (!file)
    {
        std::cerr << "Failed to load neural network: " << "cannot open file" << std::endl;
        return false;
    }

    header = Header{};

    if (1 != fread(&header, sizeof(Header), 1, file))
    {
        fclose(file);
        header.numInputs = 0;
        std::cerr << "Failed to load neural network: " << "cannot read header" << std::endl;
        return false;
    }

    if (header.version != CurrentVersion)
    {
        fclose(file);
        header = Header{};
        std::cerr << "Failed to load neural network: " << "unsupported version" << std::endl;
        return false;
    }

    if (header.numInputs == 0 || header.numInputs > MaxNeuronsInLayer)
    {
        fclose(file);
        header = Header{};
        std::cerr << "Failed to load neural network: " << "invalid number of inputs" << std::endl;
        return false;
    }

    const size_t layer0WeightsSize = header.numInputs * FirstLayerSize * sizeof(FirstLayerWeightType);
    layer0_weights = (FirstLayerWeightType*)AlignedMalloc(layer0WeightsSize, 64);
    
    if (1 != fread(layer0_weights, layer0WeightsSize, 1, file))
    {
        fclose(file);
        header = Header{};
        std::cerr << "Failed to load neural network: " << "cannot read weights" << std::endl;
        return false;
    }

    const size_t dataSize = sizeof(layer0_biases) +
        sizeof(layer1_weights) + sizeof(layer1_biases) +
        sizeof(layer2_weights) + sizeof(layer2_biases) +
        sizeof(layer3_weights) + sizeof(layer3_biases);
    if (1 != fread(layer0_biases, dataSize, 1, file))
    {
        fclose(file);
        header = Header{};
        std::cerr << "Failed to load neural network: " << "cannot read weights" << std::endl;
        return false;
    }

    fclose(file);
    return true;
}

int32_t PackedNeuralNetwork::Run(const Accumulator& accumulator) const
{
    constexpr uint32_t bufferSize = std::max(FirstLayerSize, std::max(SecondLayerSize, ThirdLayerSize));
    constexpr uint32_t linearLayerBufferSize = std::max(SecondLayerSize, ThirdLayerSize);

    IntermediateType tempA[bufferSize];
    int32_t tempB[linearLayerBufferSize];

    ClippedReLU_16(FirstLayerSize, tempA, accumulator.values);

    LinearLayer(layer1_weights, layer1_biases, FirstLayerSize, SecondLayerSize, tempB, tempA);
    ClippedReLU_32(SecondLayerSize, tempA, tempB);

    LinearLayer(layer2_weights, layer2_biases, SecondLayerSize, ThirdLayerSize, tempB, tempA);
    ClippedReLU_32(ThirdLayerSize, tempA, tempB);

    return LinearLayer_SingleOutput(layer3_weights, layer3_biases, ThirdLayerSize, 1, tempA);
}

int32_t PackedNeuralNetwork::Run(const uint16_t* activeInputIndices, const uint32_t numActiveInputs) const
{
    Accumulator accumulator;
    accumulator.Refresh(layer0_weights, layer0_biases, header.numInputs, FirstLayerSize, numActiveInputs, activeInputIndices);

    return Run(accumulator);
}

} // namespace nn
