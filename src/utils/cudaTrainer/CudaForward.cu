#include "CudaKernels.hpp"

namespace nn {
namespace cuda {

// Threads per block of L1ForwardKernel: one warp per L1 output.
static constexpr uint32_t c_l1ForwardBlockSize = 32 * nn::L1Size;

// Activation functions
__device__ __forceinline__ float Sigmoid(float x)
{
    if (x >= 0.0f)
    {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    else
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

// L1 forward: one block per batch element. The pairwise activations are staged in shared memory so
// the block reads them from global memory once. Thread t owns output (t % L1Size) and every
// (t / L1Size)-th input, which makes its weight index exactly t + k * blockDim - so a warp reads 32
// consecutive weights instead of striding one output row at a time. Partial dot products are
// reduced through shared memory.
__global__ void L1ForwardKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ outputs,
    float weightScale, float biasScale, float invWeightScale, float invBiasScale
)
{
    constexpr uint32_t inputsPerThread = c_l1ForwardBlockSize / nn::L1Size;

    __shared__ float s_input[nn::L1InputSize];
    __shared__ float s_partial[c_l1ForwardBlockSize];

    const uint32_t batchIdx = blockIdx.x;

    for (uint32_t i = threadIdx.x; i < nn::L1InputSize; i += c_l1ForwardBlockSize)
        s_input[i] = inputs[batchIdx * nn::L1InputSize + i];
    __syncthreads();

    const uint32_t variant = trainingVectors[batchIdx].variant;
    const float* __restrict__ w = weights + variant * (nn::L1InputSize + 1) * nn::L1Size;

    float sum = 0.0f;
    for (uint32_t k = 0; k < nn::L1InputSize / inputsPerThread; ++k)
    {
        const uint32_t i = threadIdx.x / nn::L1Size + k * inputsPerThread;
        sum += s_input[i] * FakeQuantize(w[threadIdx.x + k * c_l1ForwardBlockSize], weightScale, invWeightScale);
    }

    s_partial[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x < nn::L1Size)
    {
        float total = 0.0f;
        for (uint32_t k = 0; k < inputsPerThread; ++k)
            total += s_partial[threadIdx.x + k * nn::L1Size];

        const float bias = FakeQuantize(w[nn::L1InputSize * nn::L1Size + threadIdx.x], biasScale, invBiasScale);
        outputs[batchIdx * nn::L1Size + threadIdx.x] = CReLU(total + bias);
    }
}

// Forward pass of a narrow dense layer (L2 and the scalar output layer): one thread per
// (batch element, output).
template<uint32_t InputSize, uint32_t OutputSize, bool ApplyOutputCReLU>
__global__ void SmallDenseForwardKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ outputs,
    uint32_t batchSize,
    float weightScale, float biasScale, float invWeightScale, float invBiasScale
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * OutputSize) return;

    const uint32_t batchIdx = idx / OutputSize;
    const uint32_t outputIdx = idx % OutputSize;

    const uint32_t variant = trainingVectors[batchIdx].variant;
    const float* __restrict__ w = weights + variant * (InputSize + 1) * OutputSize;

    float sum = FakeQuantize(w[InputSize * OutputSize + outputIdx], biasScale, invBiasScale);
    for (uint32_t i = 0; i < InputSize; ++i)
        sum += inputs[batchIdx * InputSize + i] * FakeQuantize(w[i * OutputSize + outputIdx], weightScale, invWeightScale);

    outputs[idx] = ApplyOutputCReLU ? CReLU(sum) : sum;
}

// CUDA kernel for sigmoid activation
__global__ void SigmoidActivationKernel(
    const float* __restrict__ inputs,
    float* __restrict__ outputs,
    uint32_t batchSize
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    outputs[idx] = Sigmoid(inputs[idx]);
}

// Sparse binary input accumulation and the pairwise activation of its output. Thread (x, y) owns
// accumulator neurons x and x + AccumulatorSize / 2 of batch element y, which form one pair, so it
// writes both accumulator halves and their product for L1. The two perspectives are concatenated,
// side to move first, into L1InputSize values.
__global__ void SparseBinaryInputKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ weights,
    float* __restrict__ accumulators,
    float* __restrict__ pairwiseOutputs,
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t accumulatorSize,
    float weightScale, float biasScale, float invWeightScale, float invBiasScale
)
{
    const uint32_t halfSize = accumulatorSize / 2;
    const uint32_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize || pairIdx >= halfSize) return;

    const TrainingEntry* trainingVector = trainingVectors + batchIdx;

    // biases are stored right after the weight matrix, and every sum starts from its bias
    const auto accumulate = [&](uint32_t accumulatorIdx, const uint16_t* features, uint32_t numFeatures)
    {
        float sum = FakeQuantize(weights[inputSize * accumulatorSize + accumulatorIdx], biasScale, invBiasScale);
        for (uint32_t i = 0; i < numFeatures; ++i)
        {
            const uint32_t feature = features[i];
            if (feature >= inputSize) continue;

            // Effective weight of a feature: the bucket weight plus the shared factorizer weight,
            // fake-quantized as a sum because that is what gets packed.
            float w = weights[feature * accumulatorSize + accumulatorIdx];
#if USE_FACTORIZER
            w += weights[(nn::NumNetworkInputs + feature % FactorizerInputs) * accumulatorSize + accumulatorIdx];
#endif // USE_FACTORIZER
            sum += FakeQuantize(w, weightScale, invWeightScale);
        }
        return sum;
    };

    const uint32_t whiteBase = 2 * batchIdx * accumulatorSize;
    const float whiteLow = accumulate(pairIdx, trainingVector->whiteFeatures, trainingVector->numWhiteFeatures);
    const float whiteHigh = accumulate(pairIdx + halfSize, trainingVector->whiteFeatures, trainingVector->numWhiteFeatures);
    accumulators[whiteBase + pairIdx] = whiteLow;
    accumulators[whiteBase + pairIdx + halfSize] = whiteHigh;

    const uint32_t blackBase = whiteBase + accumulatorSize;
    const float blackLow = accumulate(pairIdx, trainingVector->blackFeatures, trainingVector->numBlackFeatures);
    const float blackHigh = accumulate(pairIdx + halfSize, trainingVector->blackFeatures, trainingVector->numBlackFeatures);
    accumulators[blackBase + pairIdx] = blackLow;
    accumulators[blackBase + pairIdx + halfSize] = blackHigh;

    pairwiseOutputs[batchIdx * accumulatorSize + pairIdx] = CReLU(whiteLow) * CReLU(whiteHigh);
    pairwiseOutputs[batchIdx * accumulatorSize + halfSize + pairIdx] = CReLU(blackLow) * CReLU(blackHigh);
}

void LaunchSparseBinaryInput(CudaBatchData& batch, const DeviceLayer& ft, cudaStream_t stream)
{
    const dim3 blockSize(32, 16);
    const dim3 gridSize(
        (nn::AccumulatorSize / 2 + blockSize.x - 1) / blockSize.x,
        (batch.batchSize + blockSize.y - 1) / blockSize.y);

    SparseBinaryInputKernel<<<gridSize, blockSize, 0, stream>>>(
        batch.trainingVectors.Get(),
        ft.weights,
        batch.accumulatorBuffer.Get(),
        batch.pairwiseBuffer.Get(),
        batch.batchSize,
        FeatureTransformerInputs,
        nn::AccumulatorSize,
        ft.quant.weightScale,
        ft.quant.biasScale,
        ft.quant.invWeightScale,
        ft.quant.invBiasScale
    );
    CUDA_CHECK(cudaGetLastError());
}

void LaunchL1Forward(CudaBatchData& batch, const DeviceLayer& l1, cudaStream_t stream)
{
    L1ForwardKernel<<<batch.batchSize, c_l1ForwardBlockSize, 0, stream>>>(
        batch.trainingVectors.Get(),
        batch.pairwiseBuffer.Get(),
        l1.weights,
        batch.l1Buffer.Get(),
        l1.quant.weightScale,
        l1.quant.biasScale,
        l1.quant.invWeightScale,
        l1.quant.invBiasScale
    );
    CUDA_CHECK(cudaGetLastError());
}

void LaunchL2Forward(CudaBatchData& batch, const DeviceLayer& l2, cudaStream_t stream)
{
    const dim3 blockSize(256);
    const dim3 gridSize((batch.batchSize * nn::L2Size + blockSize.x - 1) / blockSize.x);

    SmallDenseForwardKernel<nn::L1Size, nn::L2Size, true><<<gridSize, blockSize, 0, stream>>>(
        batch.trainingVectors.Get(),
        batch.l1Buffer.Get(),
        l2.weights,
        batch.l2Buffer.Get(),
        batch.batchSize,
        l2.quant.weightScale,
        l2.quant.biasScale,
        l2.quant.invWeightScale,
        l2.quant.invBiasScale
    );
    CUDA_CHECK(cudaGetLastError());
}

// scalar output, no activation
void LaunchL3Forward(CudaBatchData& batch, const DeviceLayer& l3, cudaStream_t stream)
{
    const dim3 blockSize(256);
    const dim3 gridSize((batch.batchSize + blockSize.x - 1) / blockSize.x);

    SmallDenseForwardKernel<nn::L2Size, 1, false><<<gridSize, blockSize, 0, stream>>>(
        batch.trainingVectors.Get(),
        batch.l2Buffer.Get(),
        l3.weights,
        batch.l3Buffer.Get(),
        batch.batchSize,
        l3.quant.weightScale,
        l3.quant.biasScale,
        l3.quant.invWeightScale,
        l3.quant.invBiasScale
    );
    CUDA_CHECK(cudaGetLastError());
}

void LaunchSigmoidActivation(CudaBatchData& batch, cudaStream_t stream)
{
    const dim3 blockSize(256);
    const dim3 gridSize((batch.batchSize + blockSize.x - 1) / blockSize.x);

    SigmoidActivationKernel<<<gridSize, blockSize, 0, stream>>>(
        batch.l3Buffer.Get(),
        batch.networkOutputs.Get(),
        batch.batchSize
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace nn
