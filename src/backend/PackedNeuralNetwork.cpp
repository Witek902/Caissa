#include "PackedNeuralNetwork.hpp"
#include "Accumulator.hpp"
#include "Memory.hpp"
#include "Math.hpp"

#include <cassert>
#include <cmath>
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

#ifdef USE_SSE4
// Horizontal sum of 4 x int32 using shuffle+add (avoids slow phaddd)
INLINE static int32_t m128_hadd(__m128i a)
{
    const __m128i hi64 = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
    a = _mm_add_epi32(a, hi64);
    const __m128i hi32 = _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
    a = _mm_add_epi32(a, hi32);
    return _mm_cvtsi128_si32(a);
}
#endif // USE_SSE4

#ifdef USE_AVX2
// Horizontal sum of 8 x int32 using extract+shuffle+add (avoids slow vphaddd)
INLINE static int32_t m256_hadd(__m256i a)
{
    const __m128i lo = _mm256_castsi256_si128(a);
    const __m128i hi = _mm256_extracti128_si256(a, 1);
    return m128_hadd(_mm_add_epi32(lo, hi));
}

// Accumulate uint8 x int8 products into int32 (32 elements per call)
INLINE static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b)
{
#ifdef NN_USE_VNNI
    acc = _mm256_dpbusd_epi32(acc, a, b);
#else
    __m256i product = _mm256_maddubs_epi16(a, b);             // uint8*int8 → int16
    product = _mm256_madd_epi16(product, _mm256_set1_epi16(1)); // int16 → int32
    acc = _mm256_add_epi32(acc, product);
#endif
}

// Reduce 4 independent int32 AVX2 sums → one __m128i of 4 int32 values, adding bias
INLINE static __m128i m256_haddx4(__m256i s0, __m256i s1, __m256i s2, __m256i s3, __m128i bias)
{
    s0 = _mm256_hadd_epi32(s0, s1);
    s2 = _mm256_hadd_epi32(s2, s3);
    s0 = _mm256_hadd_epi32(s0, s2);
    const __m128i lo = _mm256_castsi256_si128(s0);
    const __m128i hi = _mm256_extracti128_si256(s0, 1);
    return _mm_add_epi32(_mm_add_epi32(lo, hi), bias);
}
#endif // USE_AVX2

#ifdef USE_AVX512
INLINE static int32_t m512_hadd(__m512i v)
{
    const __m256i sum256 = _mm256_add_epi32(
        _mm512_castsi512_si256(v),
        _mm512_extracti64x4_epi64(v, 1));
    return m256_hadd(sum256);
}

// Accumulate uint8 x int8 products into int32 (64 elements per call)
INLINE static void m512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b)
{
#ifdef NN_USE_VNNI
    acc = _mm512_dpbusd_epi32(acc, a, b);
#else
    __m512i product = _mm512_maddubs_epi16(a, b);
    product = _mm512_madd_epi16(product, _mm512_set1_epi16(1));
    acc = _mm512_add_epi32(acc, product);
#endif
}
#endif // USE_AVX512

// Pack both FT accumulators (int16, pre-CReLU) into a single uint8 buffer
// Output layout: [stm[0..AccumulatorSize-1] || nstm[0..AccumulatorSize-1]]
// CReLU clamp: max(0, min(255, x)) via unsigned saturation
INLINE static void FT_CReLU_Pack(
    uint8_t* output,
    const AccumulatorType* stm,
    const AccumulatorType* nstm)
{
#if defined(NN_USE_AVX512)
    constexpr uint32_t regW = 32; // int16 per 512-bit register
    const __m512i zero = _mm512_setzero_si512();
    const __m512i max255 = _mm512_set1_epi16(255);
    for (uint32_t j = 0; j < AccumulatorSize; j += regW)
    {
        __m512i a = Int16VecLoad(stm + j);
        a = _mm512_min_epi16(_mm512_max_epi16(a, zero), max255);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output + j),
            _mm512_cvtepi16_epi8(a));
    }
    for (uint32_t j = 0; j < AccumulatorSize; j += regW)
    {
        __m512i b = Int16VecLoad(nstm + j);
        b = _mm512_min_epi16(_mm512_max_epi16(b, zero), max255);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output + AccumulatorSize + j),
            _mm512_cvtepi16_epi8(b));
    }
#elif defined(NN_USE_AVX2)
    constexpr uint32_t regW = 16; // int16 per 256-bit register
    const __m256i zero = _mm256_setzero_si256();
    constexpr int perm = 0b11011000; // fix AVX2 lane order after packus
    for (uint32_t j = 0; j < AccumulatorSize; j += regW * 2)
    {
        // Process 32 elements at once (two 16-element registers)
        __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(stm + j));
        __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(stm + j + regW));
        a0 = _mm256_max_epi16(a0, zero);
        a1 = _mm256_max_epi16(a1, zero);
        // packus_epi16: unsigned saturation clips >255→255, negative→0 already clamped
        __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(a0, a1), perm);
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + j), packed);
    }
    for (uint32_t j = 0; j < AccumulatorSize; j += regW * 2)
    {
        __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(nstm + j));
        __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(nstm + j + regW));
        b0 = _mm256_max_epi16(b0, zero);
        b1 = _mm256_max_epi16(b1, zero);
        __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(b0, b1), perm);
        _mm256_store_si256(reinterpret_cast<__m256i*>(output + AccumulatorSize + j), packed);
    }
#elif defined(NN_USE_SSE4)
    const __m128i zero = _mm_setzero_si128();
    for (uint32_t j = 0; j < AccumulatorSize; j += 8)
    {
        __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(stm + j));
        a = _mm_max_epi16(a, zero);
        // packus_epi16 with zero: produces [a[0..7], 0..0]
        __m128i packed = _mm_packus_epi16(a, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + j), packed);
    }
    for (uint32_t j = 0; j < AccumulatorSize; j += 8)
    {
        __m128i b = _mm_load_si128(reinterpret_cast<const __m128i*>(nstm + j));
        b = _mm_max_epi16(b, zero);
        __m128i packed = _mm_packus_epi16(b, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + AccumulatorSize + j), packed);
    }
#elif defined(NN_USE_ARM_NEON)
    for (uint32_t j = 0; j < AccumulatorSize; j += 8)
        vst1_u8(output + j, vqmovun_s16(vld1q_s16(stm + j)));
    for (uint32_t j = 0; j < AccumulatorSize; j += 8)
        vst1_u8(output + AccumulatorSize + j, vqmovun_s16(vld1q_s16(nstm + j)));
#else
    for (uint32_t j = 0; j < AccumulatorSize; ++j)
        output[j]                   = (uint8_t)std::clamp<int16_t>(stm[j],  0, 255);
    for (uint32_t j = 0; j < AccumulatorSize; ++j)
        output[j + AccumulatorSize] = (uint8_t)std::clamp<int16_t>(nstm[j], 0, 255);
#endif
}

// L1: uint8[2*AccumulatorSize] × int8[L1Size × 2*AccumulatorSize] → uint8[L1Size]
// Weights layout: row-major output-first: weights[out * 2*AccumulatorSize + in]
// Bias scale = QA*QB = 16320; after shift >>QB: output = clamp(sum>>6, 0, 255)
INLINE static void LinearLayer_L1(
    uint8_t* output,
    const uint8_t* input,
    const HiddenLayerWeightType* weights,
    const HiddenLayerBiasType* biases)
{
    static_assert(L1Size % 4 == 0, "L1Size must be divisible by 4 for SIMD");
    constexpr uint32_t InputSize = 2 * AccumulatorSize; // 2048

#if defined(NN_USE_AVX2)
    constexpr uint32_t RegW = 32;
    constexpr uint32_t NumChunks = InputSize / RegW; // 64

    alignas(16) int32_t int32_buf[L1Size];
    for (uint32_t i = 0; i < L1Size; i += 4)
    {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        const HiddenLayerWeightType* w0 = weights + (i + 0) * InputSize;
        const HiddenLayerWeightType* w1 = weights + (i + 1) * InputSize;
        const HiddenLayerWeightType* w2 = weights + (i + 2) * InputSize;
        const HiddenLayerWeightType* w3 = weights + (i + 3) * InputSize;

        for (uint32_t j = 0; j < NumChunks; ++j)
        {
            const __m256i in = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + j * RegW));
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(w0 + j * RegW)));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(w1 + j * RegW)));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(w2 + j * RegW)));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(reinterpret_cast<const __m256i*>(w3 + j * RegW)));
        }

        const __m128i bias = _mm_load_si128(reinterpret_cast<const __m128i*>(biases + i));
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
        outval = _mm_srai_epi32(outval, HiddenWeightScaleShift); // >>6
        _mm_store_si128(reinterpret_cast<__m128i*>(int32_buf + i), outval);
    }
    // Pack int32 → uint8 with CReLU clamp [0,255]
    for (uint32_t k = 0; k < L1Size; ++k)
        output[k] = (uint8_t)std::clamp(int32_buf[k], 0, 255);

#elif defined(NN_USE_AVX512)
    constexpr uint32_t RegW = 64;
    constexpr uint32_t NumChunks = InputSize / RegW; // 32

    alignas(16) int32_t int32_buf[L1Size];
    for (uint32_t i = 0; i < L1Size; i += 4)
    {
        __m512i sum0 = _mm512_setzero_si512();
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();

        const HiddenLayerWeightType* w0 = weights + (i + 0) * InputSize;
        const HiddenLayerWeightType* w1 = weights + (i + 1) * InputSize;
        const HiddenLayerWeightType* w2 = weights + (i + 2) * InputSize;
        const HiddenLayerWeightType* w3 = weights + (i + 3) * InputSize;

        for (uint32_t j = 0; j < NumChunks; ++j)
        {
            const __m512i in = _mm512_load_si512(reinterpret_cast<const __m512i*>(input + j * RegW));
            m512_add_dpbusd_epi32(sum0, in, _mm512_load_si512(reinterpret_cast<const __m512i*>(w0 + j * RegW)));
            m512_add_dpbusd_epi32(sum1, in, _mm512_load_si512(reinterpret_cast<const __m512i*>(w1 + j * RegW)));
            m512_add_dpbusd_epi32(sum2, in, _mm512_load_si512(reinterpret_cast<const __m512i*>(w2 + j * RegW)));
            m512_add_dpbusd_epi32(sum3, in, _mm512_load_si512(reinterpret_cast<const __m512i*>(w3 + j * RegW)));
        }

        int32_buf[i + 0] = std::clamp(m512_hadd(sum0) + biases[i + 0], 0, 65536) >> HiddenWeightScaleShift;
        int32_buf[i + 1] = std::clamp(m512_hadd(sum1) + biases[i + 1], 0, 65536) >> HiddenWeightScaleShift;
        int32_buf[i + 2] = std::clamp(m512_hadd(sum2) + biases[i + 2], 0, 65536) >> HiddenWeightScaleShift;
        int32_buf[i + 3] = std::clamp(m512_hadd(sum3) + biases[i + 3], 0, 65536) >> HiddenWeightScaleShift;
    }
    for (uint32_t k = 0; k < L1Size; ++k)
        output[k] = (uint8_t)std::clamp(int32_buf[k], 0, 255);

#else // scalar / SSE4 / NEON
    for (uint32_t i = 0; i < L1Size; ++i)
    {
        int32_t sum = biases[i];
        for (uint32_t j = 0; j < InputSize; ++j)
            sum += (int32_t)input[j] * (int32_t)weights[i * InputSize + j];
        output[i] = (uint8_t)std::clamp(sum >> HiddenWeightScaleShift, 0, 255);
    }
#endif
}

// L2: uint8[L1Size] × int8[L2Size × L1Size] → uint8[L2Size]
// Same quantization as L1 (QA*QB bias, >>QB shift, clamp to uint8)
INLINE static void LinearLayer_L2(
    uint8_t* output,
    const uint8_t* input,
    const HiddenLayerWeightType* weights,
    const HiddenLayerBiasType* biases)
{
    for (uint32_t i = 0; i < L2Size; ++i)
    {
        int32_t sum = biases[i];
        for (uint32_t j = 0; j < L1Size; ++j)
            sum += (int32_t)input[j] * (int32_t)weights[i * L1Size + j];
        output[i] = (uint8_t)std::clamp(sum >> HiddenWeightScaleShift, 0, 255);
    }
}

// L3: uint8[L2Size] × int16[L2Size] → int32 (no activation; divided by evaluator later)
INLINE static int32_t LinearLayer_L3(
    const uint8_t* input,
    const LastLayerWeightType* weights,
    const LastLayerBiasType* bias)
{
    int32_t sum = *bias;
    for (uint32_t j = 0; j < L2Size; ++j)
        sum += (int32_t)input[j] * (int32_t)weights[j];
    return sum;
}

///

PackedNeuralNetwork::PackedNeuralNetwork()
{
    header.magic   = MagicNumber;
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
    return true;
}

int32_t PackedNeuralNetwork::Run(const Accumulator& stmAccum, const Accumulator& nstmAccum, uint32_t variant) const
{
    ASSERT(variant < NumVariants);
    const OutputSubnetVariant& subnet = outputSubnetVariants[variant];

    // Stage 1: FT CReLU → uint8[2*AccumulatorSize]
    alignas(CACHELINE_SIZE) uint8_t l1Input[2 * AccumulatorSize];
    FT_CReLU_Pack(l1Input, stmAccum.values, nstmAccum.values);

    // Stage 2: L1
    alignas(16) uint8_t l2Input[L1Size];
    LinearLayer_L1(l2Input, l1Input, subnet.l1Weights, subnet.l1Biases);

    // Stage 3: L2
    alignas(16) uint8_t l3Input[L2Size];
    LinearLayer_L2(l3Input, l2Input, subnet.l2Weights, subnet.l2Biases);

    // Stage 4: L3 (no activation; output scaled by QA*QB3 = 262144)
    return LinearLayer_L3(l3Input, subnet.l3Weights, &subnet.l3Bias);
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
