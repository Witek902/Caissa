#pragma once

// Interface between the host code (.cpp) and the CUDA kernels (.cu). Kept free of the STL so the
// .cu files compile quickly.

#include "CudaCommon.hpp"
#include "../TrainingEntry.hpp"
#include "../../backend/PackedNeuralNetwork.hpp"

// Input factorizer: a shared 768-feature weight block (king-bucket independent) that is added to the
// weights of every king bucket during training and folded into them when the net is packed.
// TODO make it runtime switch instead of compile-time
#define USE_FACTORIZER 1

namespace nn {
namespace cuda {

static constexpr uint32_t FactorizerInputs = 12 * 64;
#if USE_FACTORIZER
static constexpr uint32_t FeatureTransformerInputs = nn::NumNetworkInputs + FactorizerInputs;
#else
static constexpr uint32_t FeatureTransformerInputs = nn::NumNetworkInputs;
#endif // USE_FACTORIZER

// factorizer weights are clipped so that their sum with the (unclipped) bucket weights stays bounded
static constexpr float FactorizerWeightRange = 0.99f;

// Quantization-aware training scales of one layer (a scale of 0 disables fake quantization)
struct QuantScales
{
    float weightScale;
    float biasScale;
    float invWeightScale;
    float invBiasScale;
};

struct AdamUpdateParams
{
    float* weights;
    float* moment1;
    float* moment2;
    const float* gradients;
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t numWeights;
    float learningRate;
    float weightDecay;
    float maxWeightRange;
    float maxBiasRange;
    uint32_t factorizerFirstWeight;
    float maxFactorizerRange;
    float beta1;
    float beta2;
    float biasCorrection1; // 1 / (1 - beta1^t)
    float biasCorrection2; // 1 / (1 - beta2^t)
};

// Device pointer of one layer's weights and its fake-quantization scales
struct DeviceLayer
{
    const float* weights;
    QuantScales quant;
};

struct CudaBatchData
{
    CudaBuffer<TrainingEntry> trainingVectors;
    CudaBuffer<float> networkOutputs;       // post-sigmoid output
    CudaBuffer<float> outputErrors;         // dLoss / d(pre-sigmoid output)
    CudaBuffer<float> lossSum;              // sum of squared output errors, accumulated across batches

    // Forward activations. accumulatorBuffer holds the raw (pre-activation) feature transformer
    // output; every later buffer holds the post-activation value.
    CudaBuffer<float> accumulatorBuffer;    // [batch][2][AccumulatorSize]
    CudaBuffer<float> pairwiseBuffer;       // [batch][L1InputSize]
    CudaBuffer<float> l1Buffer;             // [batch][L1Size]
    CudaBuffer<float> l2Buffer;             // [batch][L2Size]
    CudaBuffer<float> l3Buffer;             // [batch] (pre-sigmoid)

    // dLoss / d(pre-activation) of the hidden layers
    CudaBuffer<float> l1PreErrors;          // [batch][L1Size]
    CudaBuffer<float> l2PreErrors;          // [batch][L2Size]

    // Weight gradients. Each must match its weight storage element count exactly,
    // (inputSize + 1) * outputSize * numVariants - the "+1" row holds the per-output biases.
    CudaBuffer<float> l1Gradients;
    CudaBuffer<float> l2Gradients;
    CudaBuffer<float> l3Gradients;
    CudaBuffer<float> featureTransformerGradients;

    uint32_t batchSize;

    void Allocate(uint32_t size)
    {
        batchSize = size;

        trainingVectors.Allocate(batchSize);
        networkOutputs.Allocate(batchSize);
        outputErrors.Allocate(batchSize);
        lossSum.Allocate(1);

        accumulatorBuffer.Allocate(batchSize * 2 * nn::AccumulatorSize);
        pairwiseBuffer.Allocate(batchSize * nn::L1InputSize);
        l1Buffer.Allocate(batchSize * nn::L1Size);
        l2Buffer.Allocate(batchSize * nn::L2Size);
        l3Buffer.Allocate(batchSize);

        l1PreErrors.Allocate(batchSize * nn::L1Size);
        l2PreErrors.Allocate(batchSize * nn::L2Size);

        l1Gradients.Allocate((nn::L1InputSize + 1) * nn::L1Size * nn::NumVariants);
        l2Gradients.Allocate((nn::L1Size + 1) * nn::L2Size * nn::NumVariants);
        l3Gradients.Allocate((nn::L2Size + 1) * 1 * nn::NumVariants);
        featureTransformerGradients.Allocate((FeatureTransformerInputs + 1) * nn::AccumulatorSize);
    }
};

void LaunchAdamUpdate(const AdamUpdateParams& params, cudaStream_t stream);

// Forward pass
void LaunchSparseBinaryInput(CudaBatchData& batch, const DeviceLayer& ft, cudaStream_t stream);
void LaunchL1Forward(CudaBatchData& batch, const DeviceLayer& l1, cudaStream_t stream);
void LaunchL2Forward(CudaBatchData& batch, const DeviceLayer& l2, cudaStream_t stream);
void LaunchL3Forward(CudaBatchData& batch, const DeviceLayer& l3, cudaStream_t stream);
void LaunchSigmoidActivation(CudaBatchData& batch, cudaStream_t stream);

// Backward pass. Each LaunchLxBackward computes the layer's weight gradients, then the errors of its
// inputs' pre-activations.
void LaunchSigmoidDerivative(CudaBatchData& batch, cudaStream_t stream);
void LaunchL3Backward(CudaBatchData& batch, const DeviceLayer& l3, cudaStream_t stream);
void LaunchL2Backward(CudaBatchData& batch, const DeviceLayer& l2, cudaStream_t stream);
void LaunchL1Gradients(CudaBatchData& batch, cudaStream_t stream);
void LaunchFeatureTransformerGradients(CudaBatchData& batch, const DeviceLayer& l1, cudaStream_t stream);

#ifdef __CUDACC__

// Device functions shared by the forward and backward kernels

__device__ __forceinline__ float CReLU(float x)
{
    return fminf(1.0f, fmaxf(0.0f, x));
}

// Quantization-aware training: fake-quantize a weight/bias on read, matching the round-to-grid
// done at pack time. The float master is left untouched; the straight-through estimator means the
// gradient flows back to the float master as if no rounding happened. A scale of 0 disables it.
__device__ __forceinline__ float FakeQuantize(float w, float scale, float invScale)
{
    return (scale > 0.0f) ? (roundf(w * scale) * invScale) : w;
}

#endif // __CUDACC__

} // namespace cuda
} // namespace nn
