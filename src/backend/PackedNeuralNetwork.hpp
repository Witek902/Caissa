#pragma once

#include "Common.hpp"

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

using FirstLayerWeightType = int16_t;
using FirstLayerBiasType = int16_t;

using HiddenLayerWeightType = int8_t;
using HiddenLayerBiasType = int32_t;

using LastLayerWeightType = int16_t;
using LastLayerBiasType = int32_t;

using IntermediateType = int8_t;

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

    PackedNeuralNetwork();
    ~PackedNeuralNetwork();

    // unload weights
    void Release();

    // allocate weights
    bool Resize(const std::vector<uint32_t>& layerSizes,
                const std::vector<uint32_t>& numVariantsPerLayer = std::vector<uint32_t>());

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

    INLINE uint32_t GetNumInputs() const { return header.layerSizes[0]; }
    INLINE uint32_t GetAccumulatorSize() const { return header.layerSizes[1] / 2; }
    INLINE uint32_t GetLayerSize(uint32_t i) const { return header.layerSizes[i]; }

    void GetLayerWeightsAndBiases(uint32_t layerIndex, uint32_t layerVariant, const void*& outWeights, const void*& outBiases) const;

    INLINE const FirstLayerWeightType* GetAccumulatorWeights() const
    {
        return reinterpret_cast<const FirstLayerWeightType*>(layerDataPointers[0]);
    }
    INLINE const FirstLayerBiasType* GetAccumulatorBiases() const
    {
        return reinterpret_cast<const FirstLayerBiasType*>(GetAccumulatorWeights() + GetNumInputs() * GetAccumulatorSize());
    }

    template<typename T>
    INLINE const T* GetLayerWeights(uint32_t index, uint32_t variant) const
    {
        const void* weights;
        const void* biases;
        GetLayerWeightsAndBiases(index, variant, weights, biases);
        return reinterpret_cast<const T*>(weights);
    }

    template<typename T>
    INLINE const T* GetLayerBiases(uint32_t index, uint32_t variant) const
    {
        const void* weights;
        const void* biases;
        GetLayerWeightsAndBiases(index, variant, weights, biases);
        return reinterpret_cast<const T*>(biases);
    }

    // calculate size of all weights buffer
    size_t GetWeightsBufferSize() const;

    bool IsValid() const { return GetNumInputs() > 0; }

private:

    void ReleaseFileMapping();

    void InitLayerDataSizes();
    void InitLayerDataPointers();

    Header header;

    uint32_t numActiveLayers = 0;
    uint32_t layerDataSizes[MaxNumLayers];  // size of each layer (in bytes)
    const uint8_t* layerDataPointers[MaxNumLayers];  // base pointer to weights of each layer

    // file mapping
#if defined(PLATFORM_WINDOWS)
    HANDLE fileHandle = INVALID_HANDLE_VALUE;
    HANDLE fileMapping = INVALID_HANDLE_VALUE;
#else
    int fileDesc = -1;
#endif // PLATFORM_WINDOWS

    void* mappedData = nullptr;
    size_t mappedSize = 0;

    void* allocatedData = nullptr;

    // all weights and biases are stored in this buffer
    const uint8_t* weightsBuffer = nullptr;
};

} // namespace nn
