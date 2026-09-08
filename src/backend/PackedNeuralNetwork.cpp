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
    for (uint32_t i = 0; i < OutputSize; ++i)
    {
        const HiddenLayerWeightType* weightsRow = weights + i * InputSize;

        int32_t sum = biases[i];
        for (uint32_t j = 0; j < InputSize; ++j)
            sum += (int32_t)input[j] * (int32_t)weightsRow[j];

        output[i] = (IntermediateType)std::clamp(sum >> HiddenWeightScaleShift, 0, HiddenActivationMax);
    }
}

// Last layer: no activation, the caller divides by WeightScale * OutputScale
INLINE static int32_t LastLayer(
    const IntermediateType* input,
    const LastLayerWeightType* weights,
    LastLayerBiasType bias)
{
    int32_t sum = bias;
    for (uint32_t j = 0; j < L2Size; ++j)
        sum += (int32_t)input[j] * (int32_t)weights[j];
    return sum;
}

///

PackedNeuralNetwork::PackedNeuralNetwork()
{
    header.magic = MagicNumber;
    header.version = CurrentVersion;

    header.layerSizes[0] = NumNetworkInputs;
    header.layerSizes[1] = L1InputSize;
    header.layerSizes[2] = L1Size;
    header.layerSizes[3] = L2Size;

    header.layerVariants[0] = 1;
    header.layerVariants[1] = NumVariants;
    header.layerVariants[2] = NumVariants;
    header.layerVariants[3] = NumVariants;
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

    // side to move first, so the subnet sees whose move it is
    alignas(CACHELINE_SIZE) IntermediateType l1Input[L1InputSize];
    FT_PairwiseCReLU(l1Input, stmAccum.values);
    FT_PairwiseCReLU(l1Input + AccumulatorSize / 2, nstmAccum.values);

    alignas(CACHELINE_SIZE) IntermediateType l2Input[L1Size];
    HiddenLayer<L1InputSize, L1Size>(l2Input, l1Input, subnet.l1Weights, subnet.l1Biases);

    alignas(CACHELINE_SIZE) IntermediateType l3Input[L2Size];
    HiddenLayer<L1Size, L2Size>(l3Input, l2Input, subnet.l2Weights, subnet.l2Biases);

    return LastLayer(l3Input, subnet.l3Weights, subnet.l3Bias);
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
