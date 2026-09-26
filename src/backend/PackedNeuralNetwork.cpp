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

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE4) || defined(USE_ARM_NEON)

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

// Number of independent accumulator sets in the sparse L1 loop. A fused dot product (VNNI, NEON
// dotprod) carries its latency through the accumulator, so several groups are accumulated in
// parallel and summed at the end; with maddubs+madd the only loop-carried op is a one-cycle add.
#if defined(NN_USE_VNNI) || defined(__ARM_FEATURE_DOTPROD)
    static constexpr uint32_t L1AccumulatorSets = 4;
#else
    static constexpr uint32_t L1AccumulatorSets = 1;
#endif

// The 4 bytes of an input group as one int32, ready to be broadcast
INLINE static int32_t LoadInputGroup(const IntermediateType* input, uint32_t group)
{
    int32_t inputGroup;
    memcpy(&inputGroup, input + 4 * group, sizeof(inputGroup));
    return inputGroup;
}

#endif // USE_AVX2 || USE_AVX512 || USE_SSE4 || USE_ARM_NEON

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE4)

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
// inputs are at most HiddenActivationMax, so the intermediate int16 pair sums cannot saturate and
// the result is the same with and without VNNI.
INLINE static __m256i m256_dpbusd(__m256i sum, __m256i a, __m256i b)
{
#if defined(NN_USE_VNNI) && defined(USE_AVX512)
    // AVX512VNNI + AVX512VL form: these CPUs may lack the VEX-encoded AVX-VNNI
    return _mm256_dpbusd_epi32(sum, a, b);
#elif defined(NN_USE_VNNI)
    return _mm256_dpbusd_avx_epi32(sum, a, b);
#else
    return _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1)));
#endif // NN_USE_VNNI
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

#if defined(USE_AVX512)

// Same as m256_dpbusd
INLINE static __m512i m512_dpbusd(__m512i sum, __m512i a, __m512i b)
{
#if defined(NN_USE_VNNI)
    return _mm512_dpbusd_epi32(sum, a, b);
#else
    return _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_maddubs_epi16(a, b), _mm512_set1_epi16(1)));
#endif // NN_USE_VNNI
}

// Pairwise activation of one accumulator: the two halves are clipped to [0, QA] and multiplied,
// then shifted back into uint8. Output length is AccumulatorSize/2. Also appends the indices of the
// non-zero 4-byte output groups to the sparse L1 index list.
INLINE static void FT_PairwiseCReLU(
    IntermediateType* output, const AccumulatorType* accumulator,
    uint16_t* nnzIndices, uint32_t& nnzCount, uint32_t firstGroupIndex)
{
    constexpr uint32_t halfSize = AccumulatorSize / 2;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i maxActivation = _mm512_set1_epi16(ActivationRangeScaling);
    // the pack interleaves the 128-bit lanes of its two inputs; this permutation restores the order
    const __m512i packOrder = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
#if defined(USE_VBMI2)
    __m256i groupIndices = _mm256_add_epi16(_mm256_set1_epi16((int16_t)firstGroupIndex),
        _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
#else
    __m128i groupIndexBase = _mm_set1_epi16((int16_t)firstGroupIndex);
#endif // USE_VBMI2

    for (uint32_t i = 0; i < halfSize; i += 64)
    {
        __m512i a0 = _mm512_load_si512(accumulator + i);
        __m512i a1 = _mm512_load_si512(accumulator + i + 32);
        __m512i b0 = _mm512_load_si512(accumulator + halfSize + i);
        __m512i b1 = _mm512_load_si512(accumulator + halfSize + i + 32);

        // Only the first factor needs the lower clamp: a negative second factor makes the product
        // negative, which the unsigned pack below clips to zero
        a0 = _mm512_min_epi16(_mm512_max_epi16(a0, zero), maxActivation);
        a1 = _mm512_min_epi16(_mm512_max_epi16(a1, zero), maxActivation);
        b0 = _mm512_min_epi16(b0, maxActivation);
        b1 = _mm512_min_epi16(b1, maxActivation);

        // mulhrs is (x * y + (1 << 14)) >> 15, so this is (a * b) >> PairwiseShift rounded to nearest
        const __m512i p0 = _mm512_mulhrs_epi16(_mm512_slli_epi16(a0, 15 - PairwiseShift), b0);
        const __m512i p1 = _mm512_mulhrs_epi16(_mm512_slli_epi16(a1, 15 - PairwiseShift), b1);

        const __m512i packed = _mm512_permutexvar_epi64(packOrder, _mm512_packus_epi16(p0, p1));
        _mm512_store_si512(output + i, packed);

        // one bit per non-zero 4-byte group, 16 groups per iteration
        const uint32_t nnzMask = _mm512_cmpneq_epi32_mask(packed, zero);
#if defined(USE_VBMI2)
        // Always stores 16 entries, like AppendNnzIndices stores 8.
        // Compress into a register and then store: the compress-to-memory form is microcoded on Zen 4.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(nnzIndices + nnzCount), _mm256_maskz_compress_epi16((__mmask16)nnzMask, groupIndices));
        nnzCount += PopCount(nnzMask);
        groupIndices = _mm256_add_epi16(groupIndices, _mm256_set1_epi16(16));
#else
        AppendNnzIndices(nnzIndices, nnzCount, nnzMask & 0xFFu, groupIndexBase);
        groupIndexBase = _mm_add_epi16(groupIndexBase, _mm_set1_epi16(8));
        AppendNnzIndices(nnzIndices, nnzCount, nnzMask >> 8, groupIndexBase);
        groupIndexBase = _mm_add_epi16(groupIndexBase, _mm_set1_epi16(8));
#endif // USE_VBMI2
    }
}

// Adds one L1 input group: its 4 bytes are broadcast against the group's weight block, which holds
// 4 int8 weights for each of the 16 outputs, so one register covers all of them
INLINE static void AccumulateL1Group(__m512i& sum, const IntermediateType* input, const HiddenLayerWeightType* weights, uint32_t group)
{
    const __m512i in = _mm512_set1_epi32(LoadInputGroup(input, group));
    sum = m512_dpbusd(sum, in, _mm512_load_si512(weights + group * (4 * L1Size)));
}

// L1 over the non-zero input groups only
INLINE static void HiddenLayerL1(
    IntermediateType* output, const IntermediateType* input,
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    const uint16_t* nnzIndices, uint32_t nnzCount)
{
    static_assert(L1Size == 16, "Invalid L1 size");

    __m512i sums[L1AccumulatorSets];
    sums[0] = _mm512_load_si512(biases);
    for (uint32_t s = 1; s < L1AccumulatorSets; ++s)
    {
        sums[s] = _mm512_setzero_si512();
    }

    uint32_t k = 0;
    for (; k + L1AccumulatorSets <= nnzCount; k += L1AccumulatorSets)
    {
        for (uint32_t s = 0; s < L1AccumulatorSets; ++s)
        {
            AccumulateL1Group(sums[s], input, weights, nnzIndices[k + s]);
        }
    }
    for (; k < nnzCount; ++k)
    {
        AccumulateL1Group(sums[0], input, weights, nnzIndices[k]);
    }
    for (uint32_t s = 1; s < L1AccumulatorSets; ++s)
    {
        sums[0] = _mm512_add_epi32(sums[0], sums[s]);
    }

    const __m256i activations = m256_requantize(_mm512_castsi512_si256(sums[0]), _mm512_extracti64x4_epi64(sums[0], 1));
    _mm_store_si128(reinterpret_cast<__m128i*>(output),
        _mm_packus_epi16(_mm256_castsi256_si128(activations), _mm256_extracti128_si256(activations, 1)));
}

#else // USE_AVX2

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

        // mulhrs is (x * y + (1 << 14)) >> 15, so this is (a * b) >> PairwiseShift rounded to nearest
        const __m256i p0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(a0, 15 - PairwiseShift), b0);
        const __m256i p1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(a1, 15 - PairwiseShift), b1);

        // the pack interleaves the 128-bit lanes, so restore the natural order
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + i), packed);

        // one bit per non-zero 4-byte group
        const uint32_t nnzMask = ~(uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(packed, zero))) & 0xFFu;
        AppendNnzIndices(nnzIndices, nnzCount, nnzMask, groupIndexBase);
        groupIndexBase = _mm_add_epi16(groupIndexBase, _mm_set1_epi16(8));
    }
}

// Adds one L1 input group: its 4 bytes are broadcast against the group's weight block, which holds
// 4 int8 weights for every output, so the block's two halves cover outputs 0-7 and 8-15
INLINE static void AccumulateL1Group(__m256i& sum0, __m256i& sum1, const IntermediateType* input, const HiddenLayerWeightType* weights, uint32_t group)
{
    const __m256i in = _mm256_set1_epi32(LoadInputGroup(input, group));
    const __m256i* w = reinterpret_cast<const __m256i*>(weights + group * (4 * L1Size));
    sum0 = m256_dpbusd(sum0, in, _mm256_load_si256(w));
    sum1 = m256_dpbusd(sum1, in, _mm256_load_si256(w + 1));
}

// L1 over the non-zero input groups only
INLINE static void HiddenLayerL1(
    IntermediateType* output, const IntermediateType* input,
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    const uint16_t* nnzIndices, uint32_t nnzCount)
{
    static_assert(L1Size == 16, "Invalid L1 size");

    __m256i sums0[L1AccumulatorSets];
    __m256i sums1[L1AccumulatorSets];
    sums0[0] = _mm256_load_si256(reinterpret_cast<const __m256i*>(biases));
    sums1[0] = _mm256_load_si256(reinterpret_cast<const __m256i*>(biases + 8));
    for (uint32_t s = 1; s < L1AccumulatorSets; ++s)
    {
        sums0[s] = _mm256_setzero_si256();
        sums1[s] = _mm256_setzero_si256();
    }

    uint32_t k = 0;
    for (; k + L1AccumulatorSets <= nnzCount; k += L1AccumulatorSets)
    {
        for (uint32_t s = 0; s < L1AccumulatorSets; ++s)
        {
            AccumulateL1Group(sums0[s], sums1[s], input, weights, nnzIndices[k + s]);
        }
    }
    for (; k < nnzCount; ++k)
    {
        AccumulateL1Group(sums0[0], sums1[0], input, weights, nnzIndices[k]);
    }
    for (uint32_t s = 1; s < L1AccumulatorSets; ++s)
    {
        sums0[0] = _mm256_add_epi32(sums0[0], sums0[s]);
        sums1[0] = _mm256_add_epi32(sums1[0], sums1[s]);
    }

    const __m256i activations = m256_requantize(sums0[0], sums1[0]);
    _mm_store_si128(reinterpret_cast<__m128i*>(output),
        _mm_packus_epi16(_mm256_castsi256_si128(activations), _mm256_extracti128_si256(activations, 1)));
}

#endif // USE_AVX512

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
        const __m256i in = _mm256_set1_epi32(LoadInputGroup(input, group));
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

            // mulhrs is (x * y + (1 << 14)) >> 15, so this is (a * b) >> PairwiseShift rounded to nearest
            const __m128i p0 = _mm_mulhrs_epi16(_mm_slli_epi16(a0, 15 - PairwiseShift), b0);
            const __m128i p1 = _mm_mulhrs_epi16(_mm_slli_epi16(a1, 15 - PairwiseShift), b1);

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
    static_assert(L1Size == 16, "Invalid L1 size");

    __m128i sums[4];
    for (uint32_t j = 0; j < 4; ++j)
    {
        sums[j] = _mm_load_si128(reinterpret_cast<const __m128i*>(biases + 4 * j));
    }

    for (uint32_t k = 0; k < nnzCount; ++k)
    {
        const uint32_t group = nnzIndices[k];
        const __m128i in = _mm_set1_epi32(LoadInputGroup(input, group));
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
    static_assert(L1Size == 16, "Invalid L1 size");
    static_assert(L2Size == 32, "Invalid L2 size");

    __m128i sums[8];
    for (uint32_t j = 0; j < 8; ++j)
    {
        sums[j] = _mm_load_si128(reinterpret_cast<const __m128i*>(subnet.l2Biases + 4 * j));
    }

    for (uint32_t group = 0; group < L1Size / 4; ++group)
    {
        const __m128i in = _mm_set1_epi32(LoadInputGroup(input, group));
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

#elif defined(USE_ARM_NEON)

// Appends the group indices selected by an 8-bit mask, offset by the block's first group index.
// Always stores 8 entries; the caller's buffer is sized so the extra ones land in unused slots.
INLINE static void AppendNnzIndices(uint16_t* nnzIndices, uint32_t& nnzCount, uint32_t mask, uint16x8_t groupIndexBase)
{
    const uint16x8_t indices = vld1q_u16(c_nnzIndexTable.indices[mask]);
    vst1q_u16(nnzIndices + nnzCount, vaddq_u16(indices, groupIndexBase));
    nnzCount += PopCount(mask);
}

// Bit k is set when the k-th 4-byte group of the register is non-zero
INLINE static uint32_t neon_nnzMask(uint8x16_t packed)
{
    alignas(16) static constexpr uint32_t laneBits[4] = { 1, 2, 4, 8 };
    const uint32x4_t groups = vreinterpretq_u32_u8(packed);
    return vaddvq_u32(vandq_u32(vtstq_u32(groups, groups), vld1q_u32(laneBits)));
}

// Adds the per-lane dot products of the 4-byte groups (uint8 a x int8 b) to the int32 sums. The
// inputs are at most HiddenActivationMax, so they are passed as int8 unchanged and the int16 pair
// sums cannot overflow.
INLINE static int32x4_t neon_dpbusd(int32x4_t sum, int8x16_t a, int8x16_t b)
{
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(sum, a, b);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t hi = vmull_high_s8(a, b);
    return vpadalq_s16(sum, vpaddq_s16(lo, hi));
#endif // __ARM_FEATURE_DOTPROD
}

// Two registers of int32 sums -> one register of int16 hidden activations:
// (sum >> HiddenWeightScaleShift) clamped to [0, HiddenActivationMax]. The narrowing saturation
// cannot change the clamped result.
INLINE static int16x8_t neon_requantize(int32x4_t sumLo, int32x4_t sumHi)
{
    const int16x8_t packed = vcombine_s16(
        vqmovn_s32(vshrq_n_s32(sumLo, HiddenWeightScaleShift)),
        vqmovn_s32(vshrq_n_s32(sumHi, HiddenWeightScaleShift)));
    return vminq_s16(vmaxq_s16(packed, vdupq_n_s16(0)), vdupq_n_s16(HiddenActivationMax));
}

// Pairwise activation of one accumulator: the two halves are clipped to [0, QA] and multiplied,
// then shifted back into uint8. Output length is AccumulatorSize/2. Also appends the indices of the
// non-zero 4-byte output groups to the sparse L1 index list.
INLINE static void FT_PairwiseCReLU(
    IntermediateType* output, const AccumulatorType* accumulator,
    uint16_t* nnzIndices, uint32_t& nnzCount, uint32_t firstGroupIndex)
{
    constexpr uint32_t halfSize = AccumulatorSize / 2;
    const int16x8_t zero = vdupq_n_s16(0);
    const int16x8_t maxActivation = vdupq_n_s16(ActivationRangeScaling);
    uint16x8_t groupIndexBase = vdupq_n_u16((uint16_t)firstGroupIndex);

    // 32 outputs per iteration, so that one 8-bit mask covers the block's 4-byte groups
    for (uint32_t i = 0; i < halfSize; i += 32)
    {
        uint32_t nnzMask = 0;
        for (uint32_t j = 0; j < 32; j += 16)
        {
            int16x8_t a0 = vld1q_s16(accumulator + i + j);
            int16x8_t a1 = vld1q_s16(accumulator + i + j + 8);
            int16x8_t b0 = vld1q_s16(accumulator + halfSize + i + j);
            int16x8_t b1 = vld1q_s16(accumulator + halfSize + i + j + 8);

            // Only the first factor needs the lower clamp: a negative second factor makes the
            // product negative, which the unsigned narrowing below clips to zero
            a0 = vminq_s16(vmaxq_s16(a0, zero), maxActivation);
            a1 = vminq_s16(vmaxq_s16(a1, zero), maxActivation);
            b0 = vminq_s16(b0, maxActivation);
            b1 = vminq_s16(b1, maxActivation);

            // The rounding doubling multiply-high is (2 * x * y + (1 << 15)) >> 16, the same as mulhrs
            // on x86, so this is (a * b) >> PairwiseShift rounded to nearest
            const int16x8_t p0 = vqrdmulhq_s16(vshlq_n_s16(a0, 15 - PairwiseShift), b0);
            const int16x8_t p1 = vqrdmulhq_s16(vshlq_n_s16(a1, 15 - PairwiseShift), b1);

            const uint8x16_t packed = vcombine_u8(vqmovun_s16(p0), vqmovun_s16(p1));
            vst1q_u8(output + i + j, packed);

            nnzMask |= neon_nnzMask(packed) << (j / 4);
        }
        AppendNnzIndices(nnzIndices, nnzCount, nnzMask, groupIndexBase);
        groupIndexBase = vaddq_u16(groupIndexBase, vdupq_n_u16(8));
    }
}

// Adds one L1 input group: its 4 bytes are broadcast against the group's weight block, which holds
// 4 int8 weights for every output, so the block's four quarters cover outputs 0-3, 4-7, 8-11 and 12-15
INLINE static void AccumulateL1Group(int32x4_t* sums, const IntermediateType* input, const HiddenLayerWeightType* weights, uint32_t group)
{
    const int8x16_t in = vreinterpretq_s8_s32(vdupq_n_s32(LoadInputGroup(input, group)));
    const HiddenLayerWeightType* w = weights + group * (4 * L1Size);
    for (uint32_t j = 0; j < 4; ++j)
    {
        sums[j] = neon_dpbusd(sums[j], in, vld1q_s8(w + 16 * j));
    }
}

// L1 over the non-zero input groups only
INLINE static void HiddenLayerL1(
    IntermediateType* output, const IntermediateType* input,
    const HiddenLayerWeightType* weights, const HiddenLayerBiasType* biases,
    const uint16_t* nnzIndices, uint32_t nnzCount)
{
    static_assert(L1Size == 16, "Invalid L1 size");

    int32x4_t sums[L1AccumulatorSets][4];
    for (uint32_t j = 0; j < 4; ++j)
    {
        sums[0][j] = vld1q_s32(biases + 4 * j);
        for (uint32_t s = 1; s < L1AccumulatorSets; ++s)
        {
            sums[s][j] = vdupq_n_s32(0);
        }
    }

    uint32_t k = 0;
    for (; k + L1AccumulatorSets <= nnzCount; k += L1AccumulatorSets)
    {
        for (uint32_t s = 0; s < L1AccumulatorSets; ++s)
        {
            AccumulateL1Group(sums[s], input, weights, nnzIndices[k + s]);
        }
    }
    for (; k < nnzCount; ++k)
    {
        AccumulateL1Group(sums[0], input, weights, nnzIndices[k]);
    }
    for (uint32_t s = 1; s < L1AccumulatorSets; ++s)
    {
        for (uint32_t j = 0; j < 4; ++j)
        {
            sums[0][j] = vaddq_s32(sums[0][j], sums[s][j]);
        }
    }

    const int16x8_t activations0 = neon_requantize(sums[0][0], sums[0][1]);
    const int16x8_t activations1 = neon_requantize(sums[0][2], sums[0][3]);
    vst1q_u8(output, vcombine_u8(vqmovun_s16(activations0), vqmovun_s16(activations1)));
}

// L2 over all four input groups (32 outputs per group block), then L3 on the requantized sums
INLINE static int32_t HiddenLayerL2_LastLayer(const IntermediateType* input, const PackedNeuralNetwork::OutputSubnetVariant& subnet)
{
    static_assert(L1Size == 16, "Invalid L1 size");
    static_assert(L2Size == 32, "Invalid L2 size");

    int32x4_t sums[8];
    for (uint32_t j = 0; j < 8; ++j)
    {
        sums[j] = vld1q_s32(subnet.l2Biases + 4 * j);
    }

    for (uint32_t group = 0; group < L1Size / 4; ++group)
    {
        const int8x16_t in = vreinterpretq_s8_s32(vdupq_n_s32(LoadInputGroup(input, group)));
        const HiddenLayerWeightType* w = subnet.l2Weights + group * (4 * L2Size);
        for (uint32_t j = 0; j < 8; ++j)
        {
            sums[j] = neon_dpbusd(sums[j], in, vld1q_s8(w + 16 * j));
        }
    }

    int32x4_t total = vdupq_n_s32(0);
    for (uint32_t j = 0; j < 4; ++j)
    {
        const int16x8_t activations = neon_requantize(sums[2 * j], sums[2 * j + 1]);
        const int16x8_t w = vld1q_s16(subnet.l3Weights + 8 * j);
        total = vmlal_s16(total, vget_low_s16(activations), vget_low_s16(w));
        total = vmlal_high_s16(total, activations, w);
    }

    return subnet.l3Bias + vaddvq_s32(total);
}

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
        output[i] = (IntermediateType)((a * b + PairwiseRounding) >> PairwiseShift);
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

    if (header.version != CurrentVersion)
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

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE4) || defined(USE_ARM_NEON)

    // Indices of the non-zero 4-byte groups of l1Input. Every block of 8 groups (16 with VBMI2) stores one
    // entry per group starting at the running count, which never exceeds the block's own first slot, so
    // the buffer needs no slack.
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
