#include "PackedNeuralNetwork.hpp"
#include "Accumulator.hpp"
#include "Memory.hpp"
#include "Math.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>

#if defined(PLATFORM_LINUX)
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
#endif // PLATFORM_LINUX


namespace nn {

static_assert(sizeof(PackedNeuralNetwork::Header) % CACHELINE_SIZE == 0, "Network header size must be multiple of cacheline size");

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE4)

// For each 8-bit mask, the positions of its set bits. Used to append the indices of the non-zero
// 4-byte L1 input groups to the sparse index list eight at a time.
struct NnzIndexTable
{
    alignas(CACHELINE_SIZE) uint16_t indices[256][8];
};

static constexpr NnzIndexTable c_nnzIndexTable = []()
{
    NnzIndexTable table{};
    for (uint32_t mask = 0; mask < 256; ++mask)
    {
        uint32_t count = 0;
        for (uint32_t bit = 0; bit < 8; ++bit)
        {
            if (mask & (1u << bit))
                table.indices[mask][count++] = (uint16_t)bit;
        }
    }
    return table;
}();

// Appends the group indices selected by an 8-bit mask, offset by the block's first group index.
// Always stores 8 entries; the caller's buffer is sized so the extra ones land in unused slots.
INLINE static void AppendNnzIndices(uint16_t* nnzIndices, uint32_t& nnzCount, uint32_t mask, __m128i groupIndexBase)
{
    const __m128i indices = _mm_load_si128(reinterpret_cast<const __m128i*>(c_nnzIndexTable.indices[mask]));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(nnzIndices + nnzCount), _mm_add_epi16(indices, groupIndexBase));
    nnzCount += PopCount(mask);
}

// Horizontal sum of 4 x int32 using shuffle+add (avoids slow phaddd)
INLINE static int32_t m128_hadd(__m128i a)
{
    const __m128i hi64 = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
    a = _mm_add_epi32(a, hi64);
    const __m128i hi32 = _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
    a = _mm_add_epi32(a, hi32);
    return _mm_cvtsi128_si32(a);
}

#if defined(USE_AVX2) || defined(USE_AVX512)

// Horizontal sum of 8 x int32 (avoids slow vphaddd)
INLINE static int32_t m256_hadd(__m256i a)
{
    const __m128i lo = _mm256_castsi256_si128(a);
    const __m128i hi = _mm256_extracti128_si256(a, 1);
    return m128_hadd(_mm_add_epi32(lo, hi));
}

// Adds the per-lane dot products of the 4-byte groups (uint8 a x int8 b) to the int32 sums. The
// inputs are at most HiddenActivationMax, so the intermediate int16 pair sums cannot saturate.
INLINE static __m256i m256_dpbusd(__m256i sum, __m256i a, __m256i b)
{
    return _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1)));
}

// Two registers of int32 sums -> one register of int16 hidden activations in natural lane order:
// (sum >> HiddenWeightScaleShift) clamped to [0, HiddenActivationMax]. The pack interleaves the
// 128-bit lanes, so the permute restores the order; its saturation cannot change the clamped result.
INLINE static __m256i m256_requantize(__m256i sumLo, __m256i sumHi)
{
    __m256i packed = _mm256_packs_epi32(_mm256_srai_epi32(sumLo, HiddenWeightScaleShift), _mm256_srai_epi32(sumHi, HiddenWeightScaleShift));
    packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm256_min_epi16(_mm256_max_epi16(packed, _mm256_setzero_si256()), _mm256_set1_epi16(HiddenActivationMax));
}

// Pairwise activation of one accumulator: the two halves are clipped to [0, QA] and multiplied,
// then shifted back into uint8. Output length is AccumulatorSize/2. Also appends the indices of the
// non-zero 4-byte output groups to the sparse L1 index list.
INLINE static void FT_PairwiseCReLU(
    IntermediateType* output, const AccumulatorType* accumulator,
    uint16_t* nnzIndices, uint32_t& nnzCount, uint32_t firstGroupIndex)
{
    constexpr uint32_t halfSize = AccumulatorSize / 2;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i maxActivation = _mm256_set1_epi16(ActivationRangeScaling);
    __m128i groupIndexBase = _mm_set1_epi16((int16_t)firstGroupIndex);

    for (uint32_t i = 0; i < halfSize; i += 32)
    {
        __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + i));
        __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + i + 16));
        __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + halfSize + i));
        __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + halfSize + i + 16));

        // Only the first factor needs the lower clamp: a negative second factor makes the product
        // negative, which the unsigned pack below clips to zero
        a0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), maxActivation);
        a1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), maxActivation);
        b0 = _mm256_min_epi16(b0, maxActivation);
        b1 = _mm256_min_epi16(b1, maxActivation);

        // (a << (16 - PairwiseShift)) * b >> 16 == (a * b) >> PairwiseShift
        const __m256i p0 = _mm256_mulhi_epi16(_mm256_slli_epi16(a0, 16 - PairwiseShift), b0);
        const __m256i p1 = _mm256_mulhi_epi16(_mm256_slli_epi16(a1, 16 - PairwiseShift), b1);

        // the pack interleaves the 128-bit lanes, so restore the natural order
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + i), packed);

        // one bit per non-zero 4-byte group
        const uint32_t nnzMask = ~(uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(packed, zero))) & 0xFFu;
        AppendNnzIndices(nnzIndices, nnzCount, nnzMask, groupIndexBase);
        groupIndexBase = _mm_add_epi16(groupIndexBase, _mm_set1_epi16(8));
    }
}

// L1 over the non-zero input groups only. A group's weight block holds 4 int8 weights for every
// output, so its two 32-byte halves cover outputs 0-7 and 8-15.
INLINE static void HiddenLayerL1(
    IntermediateType* output, const IntermediateType* input,
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    const uint16_t* nnzIndices, uint32_t nnzCount)
{
    static_assert(L1Size == 16, "Invalid L1 size");

    __m256i sum0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(biases));
    __m256i sum1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(biases + 8));

    for (uint32_t k = 0; k < nnzCount; ++k)
    {
        const uint32_t group = nnzIndices[k];
        int32_t inputGroup;
        memcpy(&inputGroup, input + 4 * group, sizeof(inputGroup));
        const __m256i in = _mm256_set1_epi32(inputGroup);
        const __m256i* w = reinterpret_cast<const __m256i*>(weights + group * (4 * L1Size));
        sum0 = m256_dpbusd(sum0, in, _mm256_load_si256(w));
        sum1 = m256_dpbusd(sum1, in, _mm256_load_si256(w + 1));
    }

    const __m256i activations = m256_requantize(sum0, sum1);
    _mm_store_si128(reinterpret_cast<__m128i*>(output),
        _mm_packus_epi16(_mm256_castsi256_si128(activations), _mm256_extracti128_si256(activations, 1)));
}

// L2 over all four input groups (32 outputs per group block), then L3 on the requantized sums
INLINE static int32_t HiddenLayerL2_LastLayer(const IntermediateType* input, const PackedNeuralNetwork::OutputSubnetVariant& subnet)
{
    static_assert(L1Size == 16, "Invalid L1 size");
    static_assert(L2Size == 32, "Invalid L2 size");

    __m256i sums[4];
    for (uint32_t j = 0; j < 4; ++j)
    {
        sums[j] = _mm256_load_si256(reinterpret_cast<const __m256i*>(subnet.l2Biases + 8 * j));
    }

    for (uint32_t group = 0; group < L1Size / 4; ++group)
    {
        int32_t inputGroup;
        memcpy(&inputGroup, input + 4 * group, sizeof(inputGroup));
        const __m256i in = _mm256_set1_epi32(inputGroup);
        const __m256i* w = reinterpret_cast<const __m256i*>(subnet.l2Weights + group * (4 * L2Size));
        for (uint32_t j = 0; j < 4; ++j)
        {
            sums[j] = m256_dpbusd(sums[j], in, _mm256_load_si256(w + j));
        }
    }

    const __m256i in0 = m256_requantize(sums[0], sums[1]);
    const __m256i in1 = m256_requantize(sums[2], sums[3]);
    const __m256i w0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(subnet.l3Weights));
    const __m256i w1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(subnet.l3Weights + 16));
    return subnet.l3Bias + m256_hadd(_mm256_add_epi32(_mm256_madd_epi16(in0, w0), _mm256_madd_epi16(in1, w1)));
}

#else // USE_SSE4

// Adds the per-lane dot products of the 4-byte groups (uint8 a x int8 b) to the int32 sums. The
// inputs are at most HiddenActivationMax, so the intermediate int16 pair sums cannot saturate.
INLINE static __m128i m128_dpbusd(__m128i sum, __m128i a, __m128i b)
{
    return _mm_add_epi32(sum, _mm_madd_epi16(_mm_maddubs_epi16(a, b), _mm_set1_epi16(1)));
}

// Two registers of int32 sums -> one register of int16 hidden activations:
// (sum >> HiddenWeightScaleShift) clamped to [0, HiddenActivationMax]. The pack's saturation cannot
// change the clamped result.
INLINE static __m128i m128_requantize(__m128i sumLo, __m128i sumHi)
{
    const __m128i packed = _mm_packs_epi32(_mm_srai_epi32(sumLo, HiddenWeightScaleShift), _mm_srai_epi32(sumHi, HiddenWeightScaleShift));
    return _mm_min_epi16(_mm_max_epi16(packed, _mm_setzero_si128()), _mm_set1_epi16(HiddenActivationMax));
}

// Pairwise activation of one accumulator: the two halves are clipped to [0, QA] and multiplied,
// then shifted back into uint8. Output length is AccumulatorSize/2. Also appends the indices of the
// non-zero 4-byte output groups to the sparse L1 index list.
INLINE static void FT_PairwiseCReLU(
    IntermediateType* output, const AccumulatorType* accumulator,
    uint16_t* nnzIndices, uint32_t& nnzCount, uint32_t firstGroupIndex)
{
    constexpr uint32_t halfSize = AccumulatorSize / 2;
    const __m128i zero = _mm_setzero_si128();
    const __m128i maxActivation = _mm_set1_epi16(ActivationRangeScaling);
    __m128i groupIndexBase = _mm_set1_epi16((int16_t)firstGroupIndex);

    // 32 outputs per iteration, so that one 8-bit mask covers the block's 4-byte groups
    for (uint32_t i = 0; i < halfSize; i += 32)
    {
        uint32_t nnzMask = 0;
        for (uint32_t j = 0; j < 32; j += 16)
        {
            __m128i a0 = _mm_load_si128(reinterpret_cast<const __m128i*>(accumulator + i + j));
            __m128i a1 = _mm_load_si128(reinterpret_cast<const __m128i*>(accumulator + i + j + 8));
            __m128i b0 = _mm_load_si128(reinterpret_cast<const __m128i*>(accumulator + halfSize + i + j));
            __m128i b1 = _mm_load_si128(reinterpret_cast<const __m128i*>(accumulator + halfSize + i + j + 8));

            // Only the first factor needs the lower clamp: a negative second factor makes the
            // product negative, which the unsigned pack below clips to zero
            a0 = _mm_min_epi16(_mm_max_epi16(a0, zero), maxActivation);
            a1 = _mm_min_epi16(_mm_max_epi16(a1, zero), maxActivation);
            b0 = _mm_min_epi16(b0, maxActivation);
            b1 = _mm_min_epi16(b1, maxActivation);

            // (a << (16 - PairwiseShift)) * b >> 16 == (a * b) >> PairwiseShift
            const __m128i p0 = _mm_mulhi_epi16(_mm_slli_epi16(a0, 16 - PairwiseShift), b0);
            const __m128i p1 = _mm_mulhi_epi16(_mm_slli_epi16(a1, 16 - PairwiseShift), b1);

            const __m128i packed = _mm_packus_epi16(p0, p1);
            _mm_store_si128(reinterpret_cast<__m128i*>(output + i + j), packed);

            // one bit per non-zero 4-byte group
            nnzMask |= (~(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(packed, zero))) & 0xFu) << (j / 4);
        }
        AppendNnzIndices(nnzIndices, nnzCount, nnzMask, groupIndexBase);
        groupIndexBase = _mm_add_epi16(groupIndexBase, _mm_set1_epi16(8));
    }
}

// L1 over the non-zero input groups only. A group's weight block holds 4 int8 weights for every
// output, so its four 16-byte quarters cover outputs 0-3, 4-7, 8-11 and 12-15.
INLINE static void HiddenLayerL1(
    IntermediateType* output, const IntermediateType* input,
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    const uint16_t* nnzIndices, uint32_t nnzCount)
{
    static_assert(L1Size == 16, "");

    __m128i sums[4];
    for (uint32_t j = 0; j < 4; ++j)
    {
        sums[j] = _mm_load_si128(reinterpret_cast<const __m128i*>(biases + 4 * j));
    }

    for (uint32_t k = 0; k < nnzCount; ++k)
    {
        const uint32_t group = nnzIndices[k];
        int32_t inputGroup;
        memcpy(&inputGroup, input + 4 * group, sizeof(inputGroup));
        const __m128i in = _mm_set1_epi32(inputGroup);
        const __m128i* w = reinterpret_cast<const __m128i*>(weights + group * (4 * L1Size));
        for (uint32_t j = 0; j < 4; ++j)
        {
            sums[j] = m128_dpbusd(sums[j], in, _mm_load_si128(w + j));
        }
    }

    _mm_store_si128(reinterpret_cast<__m128i*>(output),
        _mm_packus_epi16(m128_requantize(sums[0], sums[1]), m128_requantize(sums[2], sums[3])));
}

// L2 over all four input groups (32 outputs per group block), then L3 on the requantized sums
INLINE static int32_t HiddenLayerL2_LastLayer(const IntermediateType* input, const PackedNeuralNetwork::OutputSubnetVariant& subnet)
{
    static_assert(L1Size == 16 && L2Size == 32, "");

    __m128i sums[8];
    for (uint32_t j = 0; j < 8; ++j)
    {
        sums[j] = _mm_load_si128(reinterpret_cast<const __m128i*>(subnet.l2Biases + 4 * j));
    }

    for (uint32_t group = 0; group < L1Size / 4; ++group)
    {
        int32_t inputGroup;
        memcpy(&inputGroup, input + 4 * group, sizeof(inputGroup));
        const __m128i in = _mm_set1_epi32(inputGroup);
        const __m128i* w = reinterpret_cast<const __m128i*>(subnet.l2Weights + group * (4 * L2Size));
        for (uint32_t j = 0; j < 8; ++j)
        {
            sums[j] = m128_dpbusd(sums[j], in, _mm_load_si128(w + j));
        }
    }

    __m128i total = _mm_setzero_si128();
    for (uint32_t j = 0; j < 4; ++j)
    {
        const __m128i activations = m128_requantize(sums[2 * j], sums[2 * j + 1]);
        const __m128i w = _mm_load_si128(reinterpret_cast<const __m128i*>(subnet.l3Weights + 8 * j));
        total = _mm_add_epi32(total, _mm_madd_epi16(activations, w));
    }

    return subnet.l3Bias + m128_hadd(total);
}

#endif // USE_AVX2 || USE_AVX512

#else // no SIMD support

// Pairwise activation of one accumulator: the two halves are clipped to [0, QA] and multiplied,
// then shifted back into uint8. Output length is AccumulatorSize/2.
INLINE static void FT_PairwiseCReLU(IntermediateType* output, const AccumulatorType* accumulator)
{
    constexpr uint32_t halfSize = AccumulatorSize / 2;

    for (uint32_t i = 0; i < halfSize; ++i)
    {
        const int32_t a = std::clamp<int32_t>(accumulator[i], 0, ActivationRangeScaling);
        const int32_t b = std::clamp<int32_t>(accumulator[i + halfSize], 0, ActivationRangeScaling);
        output[i] = (IntermediateType)((a * b) >> PairwiseShift);
    }
}

// uint8 input x int8 weights -> int32, requantized back to the uint8 activation range.
// The bias carries the combined input and weight scale, so a single shift by the weight scale
// returns the value to the activation scale.
template<uint32_t InputSize, uint32_t OutputSize>
INLINE static void HiddenLayer(
    IntermediateType* output,
    const IntermediateType* input,
    const HiddenLayerWeightType* weights,
    const HiddenLayerBiasType* biases)
{
    constexpr uint32_t blockSize = 4 * OutputSize;

    // One partial sum per weight slot of a group block, with the group's 4 inputs repeated across
    // the block, so the inner loop is an elementwise multiply-accumulate over contiguous memory
    // that the compiler can vectorize
    int32_t partials[blockSize] = {};
    for (uint32_t group = 0; group < InputSize / 4; ++group)
    {
        int32_t inputGroup;
        memcpy(&inputGroup, input + 4 * group, sizeof(inputGroup));
        if (inputGroup == 0)
            continue;

        IntermediateType blockInput[blockSize];
        for (uint32_t i = 0; i < OutputSize; ++i)
            memcpy(blockInput + 4 * i, &inputGroup, sizeof(inputGroup));

        const HiddenLayerWeightType* blockWeights = weights + group * blockSize;
        for (uint32_t n = 0; n < blockSize; ++n)
            partials[n] += (int32_t)blockInput[n] * (int32_t)blockWeights[n];
    }

    for (uint32_t i = 0; i < OutputSize; ++i)
    {
        const int32_t sum = biases[i] + partials[4 * i] + partials[4 * i + 1] + partials[4 * i + 2] + partials[4 * i + 3];
        output[i] = (IntermediateType)std::clamp(sum >> HiddenWeightScaleShift, 0, HiddenActivationMax);
    }
}

// Last layer: no activation, the caller divides by WeightScale * OutputScale
INLINE static int32_t LastLayer(const IntermediateType* input, const LastLayerWeightType* weights, LastLayerBiasType bias)
{
    int32_t sum = bias;
    for (uint32_t j = 0; j < L2Size; ++j)
    {
        sum += (int32_t)input[j] * (int32_t)weights[j];
    }
    return sum;
}

#endif // USE_AVX2 || USE_AVX512 || USE_SSE4

///

// Converts a hidden layer weight matrix from output-major to the grouped layout, in place
// TODO remove this when version 13 is no longer supported
static void RegroupHiddenWeights(HiddenLayerWeightType* weights, uint32_t numInputs, uint32_t numOutputs)
{
    const std::vector<HiddenLayerWeightType> outputMajor(weights, weights + numInputs * numOutputs);
    for (uint32_t output = 0; output < numOutputs; ++output)
    {
        for (uint32_t input = 0; input < numInputs; ++input)
        {
            weights[HiddenWeightIndex(input, output, numOutputs)] = outputMajor[output * numInputs + input];
        }
    }
}

///

PackedNeuralNetwork::PackedNeuralNetwork()
{
    header.magic = MagicNumber;
    header.version = CurrentVersion;
}

bool PackedNeuralNetwork::SaveToFile(const char* filePath) const
{
    FILE* file = fopen(filePath, "wb");
    if (!file)
    {
        std::cerr << "Failed to save neural network: " << "cannot open file" << std::endl;
        return false;
    }

    if (1 != fwrite(this, sizeof(PackedNeuralNetwork), 1, file))
    {
        fclose(file);
        std::cerr << "Failed to save neural network: " << "cannot write header" << std::endl;
        return false;
    }

    fclose(file);
    return true;
}

bool PackedNeuralNetwork::LoadFromFile(const char* filePath)
{
    FILE* file = fopen(filePath, "rb");
    if (!file)
    {
        std::cerr << "Failed to load neural network: " << "cannot open file" << std::endl;
        return false;
    }

    if (1 != fread(this, sizeof(PackedNeuralNetwork), 1, file))
    {
        fclose(file);
        std::cerr << "Failed to load neural network: " << "cannot read header" << std::endl;
        return false;
    }

    fclose(file);

    if (header.magic != MagicNumber)
    {
        std::cerr << "Failed to load neural network: " << "not a network file" << std::endl;
        return false;
    }

    // version 13 differs from 14 only in the hidden layer weight layout
    if (header.version == 13)
    {
        static_assert(CurrentVersion == 14, "Remove this code when version 13 is no longer supported");
        for (OutputSubnetVariant& subnet : outputSubnetVariants)
        {
            RegroupHiddenWeights(subnet.l1Weights, L1InputSize, L1Size);
            RegroupHiddenWeights(subnet.l2Weights, L1Size, L2Size);
        }
        header.version = CurrentVersion;
    }
    else if (header.version != CurrentVersion)
    {
        std::cerr << "Failed to load neural network: " << "unsupported version " << header.version << std::endl;
        return false;
    }

    return true;
}

int32_t PackedNeuralNetwork::Run(const Accumulator& stmAccum, const Accumulator& nstmAccum, uint32_t variant) const
{
    ASSERT(variant < NumVariants);
    const OutputSubnetVariant& subnet = outputSubnetVariants[variant];

    // side to move first, so the subnet sees whose move it is
    alignas(CACHELINE_SIZE) IntermediateType l1Input[L1InputSize];
    alignas(CACHELINE_SIZE) IntermediateType l2Input[L1Size];

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE4)

    // Indices of the non-zero 4-byte groups of l1Input. Every 8-group block stores 8 entries starting
    // at the running count, which never exceeds the block's own first slot, so the buffer needs no slack.
    alignas(16) uint16_t nnzIndices[L1InputSize / 4];
    uint32_t nnzCount = 0;
    FT_PairwiseCReLU(l1Input, stmAccum.values, nnzIndices, nnzCount, 0);
    FT_PairwiseCReLU(l1Input + AccumulatorSize / 2, nstmAccum.values, nnzIndices, nnzCount, AccumulatorSize / 2 / 4);

    HiddenLayerL1(l2Input, l1Input, subnet.l1Weights, subnet.l1Biases, nnzIndices, nnzCount);
    return HiddenLayerL2_LastLayer(l2Input, subnet);

#else // no SIMD support

    FT_PairwiseCReLU(l1Input, stmAccum.values);
    FT_PairwiseCReLU(l1Input + AccumulatorSize / 2, nstmAccum.values);

    HiddenLayer<L1InputSize, L1Size>(l2Input, l1Input, subnet.l1Weights, subnet.l1Biases);

    alignas(CACHELINE_SIZE) IntermediateType l3Input[L2Size];
    HiddenLayer<L1Size, L2Size>(l3Input, l2Input, subnet.l2Weights, subnet.l2Biases);

    return LastLayer(l3Input, subnet.l3Weights, subnet.l3Bias);

#endif
}

int32_t PackedNeuralNetwork::Run(const uint16_t* stmFeatures, const uint32_t stmNumFeatures, const uint16_t* nstmFeatures, const uint32_t nstmNumFeatures, uint32_t variant) const
{
    Accumulator stmAccum;
    stmAccum.Refresh(accumulatorWeights, accumulatorBiases, stmNumFeatures, stmFeatures);

    Accumulator nstmAccum;
    nstmAccum.Refresh(accumulatorWeights, accumulatorBiases, nstmNumFeatures, nstmFeatures);

    return Run(stmAccum, nstmAccum, variant);
}

} // namespace nn
