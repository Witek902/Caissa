#include "CudaKernels.hpp"

namespace nn {
namespace cuda {

// Number of slices the batch is split into when reducing dense weight gradients. Each slice is a
// separate block, so the reduction has enough parallelism for the small output subnet layers.
static constexpr uint32_t c_l1GradientBatchSplits = 16;
static constexpr uint32_t c_l2GradientBatchSplits = 64;
static constexpr uint32_t c_l3GradientBatchSplits = 256;

// Gate for backpropagating through CReLU, evaluated on the post-activation value.
__device__ __forceinline__ float CReLUGate(float activated)
{
    return (activated > 0.0f && activated < 1.0f) ? 1.0f : 0.0f;
}

#if USE_FACTORIZER
// The factorizer weight of a feature is shared by all king buckets, so its gradient is the sum of
// the bucket gradients of that feature. Computed from the accumulated bucket gradients instead of
// adding a second atomic per feature to FeatureTransformerGradientsKernel.
__global__ void FactorizerGradientsKernel(
    float* __restrict__ weightGradients,
    uint32_t accumulatorSize
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= FactorizerInputs * accumulatorSize) return;

    float sum = 0.0f;
    for (uint32_t bucket = 0; bucket < nn::NumKingBuckets; ++bucket)
        sum += weightGradients[bucket * FactorizerInputs * accumulatorSize + idx];

    weightGradients[nn::NumNetworkInputs * accumulatorSize + idx] = sum;
}
#endif // USE_FACTORIZER

// Backward pass kernels
static constexpr uint32_t c_sigmoidDerivativeBlockSize = 256;

// Also accumulates the batch's squared-error sum into lossSum (one atomic per block).
__global__ void SigmoidDerivativeKernel(
    const float* __restrict__ outputs,
    const TrainingEntry* __restrict__ trainingVectors,
    float* __restrict__ outputErrors,
    float* __restrict__ lossSum,
    uint32_t batchSize
)
{
    __shared__ float s_squaredErrors[c_sigmoidDerivativeBlockSize];

    const uint32_t batchIdx = blockIdx.x * blockDim.x + threadIdx.x;

    float squaredError = 0.0f;
    if (batchIdx < batchSize)
    {
        const float output = outputs[batchIdx];
        const float target = trainingVectors[batchIdx].targetOutput;
        const float derivative = output * (1.0f - output);
        const float diff = output - target;
        outputErrors[batchIdx] = 2.0f * diff * derivative;
        squaredError = diff * diff;
    }

    s_squaredErrors[threadIdx.x] = squaredError;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            s_squaredErrors[threadIdx.x] += s_squaredErrors[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(lossSum, s_squaredErrors[0]);
}

// Weight and bias gradients of one dense output-subnet layer, for all variants at once.
//
// threadIdx spans (input within a tile, output); blockIdx.x selects the input tile and blockIdx.y
// a slice of the batch. The slice is walked in tiles staged in shared memory, so the block reads
// each input and output gradient from global memory once instead of once per thread that needs it.
// Per-variant partials live in registers - the unrolled compare keeps that array out of local
// memory. The bias gradient comes for free from the same output-gradient read, and is accumulated
// only by the first input tile so it is not counted once per tile.
template<uint32_t OutputSize, uint32_t InputTileSize>
__global__ void DenseWeightGradientsKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ activatedInputs, // [batchSize][inputSize]
    const float* __restrict__ outputGrads,     // [batchSize][OutputSize]
    float* __restrict__ weightGradients,       // [numVariants][(inputSize + 1) * OutputSize]
    uint32_t batchSize,
    uint32_t inputSize)
{
    constexpr uint32_t batchTileSize = 32;
    constexpr uint32_t blockThreads = InputTileSize * OutputSize;

    __shared__ float s_inputs[batchTileSize * InputTileSize];
    __shared__ float s_grads[batchTileSize * OutputSize];
    __shared__ uint8_t s_variants[batchTileSize];

    const uint32_t inputIdx = blockIdx.x * InputTileSize + threadIdx.x;
    const uint32_t outputIdx = threadIdx.y;
    const uint32_t threadIdxFlat = threadIdx.y * InputTileSize + threadIdx.x;

    // The bias does not depend on the input, so only one input tile accumulates it. The condition
    // is block-uniform, so skipping the work costs no divergence.
    const bool accumulateBias = (blockIdx.x == 0);

    float weightGrad[nn::NumVariants];
    float biasGrad[nn::NumVariants];
    #pragma unroll
    for (uint32_t v = 0; v < nn::NumVariants; ++v) { weightGrad[v] = 0.0f; biasGrad[v] = 0.0f; }

    const uint32_t sliceSize = (batchSize + gridDim.y - 1) / gridDim.y;
    const uint32_t begin = blockIdx.y * sliceSize;
    const uint32_t end = min(begin + sliceSize, batchSize);

    for (uint32_t base = begin; base < end; base += batchTileSize)
    {
        const uint32_t tileCount = min(batchTileSize, end - base);

        __syncthreads();
        for (uint32_t n = threadIdxFlat; n < tileCount * InputTileSize; n += blockThreads)
            s_inputs[n] = activatedInputs[(base + n / InputTileSize) * inputSize + blockIdx.x * InputTileSize + n % InputTileSize];
        for (uint32_t n = threadIdxFlat; n < tileCount * OutputSize; n += blockThreads)
            s_grads[n] = outputGrads[(base + n / OutputSize) * OutputSize + n % OutputSize];
        for (uint32_t n = threadIdxFlat; n < tileCount; n += blockThreads)
            s_variants[n] = trainingVectors[base + n].variant;
        __syncthreads();

        for (uint32_t b = 0; b < tileCount; ++b)
        {
            const uint32_t variant = s_variants[b];
            const float outputGrad = s_grads[b * OutputSize + outputIdx];
            const float input = s_inputs[b * InputTileSize + threadIdx.x];

            #pragma unroll
            for (uint32_t v = 0; v < nn::NumVariants; ++v)
                if (variant == v) weightGrad[v] += input * outputGrad;

            if (accumulateBias)
            {
                #pragma unroll
                for (uint32_t v = 0; v < nn::NumVariants; ++v)
                    if (variant == v) biasGrad[v] += outputGrad;
            }
        }
    }

    const uint32_t variantStride = (inputSize + 1) * OutputSize;

    #pragma unroll
    for (uint32_t v = 0; v < nn::NumVariants; ++v)
        atomicAdd(&weightGradients[v * variantStride + inputIdx * OutputSize + outputIdx], weightGrad[v]);

    if (accumulateBias && threadIdx.x == 0)
    {
        #pragma unroll
        for (uint32_t v = 0; v < nn::NumVariants; ++v)
            atomicAdd(&weightGradients[v * variantStride + inputSize * OutputSize + outputIdx], biasGrad[v]);
    }
}

// Backpropagates a dense layer's output gradient to its input's pre-activation, gating by the
// CReLU that produced the (post-activation) input.
template<uint32_t InputSize, uint32_t OutputSize>
__global__ void BackpropToHiddenKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ outputGrads,     // [batchSize][OutputSize]
    const float* __restrict__ weights,         // [numVariants][(InputSize + 1) * OutputSize]
    const float* __restrict__ activatedInputs, // [batchSize][InputSize]
    float* __restrict__ inputErrors,           // [batchSize][InputSize]
    uint32_t batchSize,
    float weightScale, float invWeightScale)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * InputSize) return;

    const uint32_t batchIdx = idx / InputSize;
    const uint32_t inputIdx = idx % InputSize;

    const uint32_t variant = trainingVectors[batchIdx].variant;
    const float* __restrict__ w = weights + variant * (InputSize + 1) * OutputSize + inputIdx * OutputSize;

    float error = 0.0f;
    for (uint32_t o = 0; o < OutputSize; ++o)
        error += outputGrads[batchIdx * OutputSize + o] * FakeQuantize(w[o], weightScale, invWeightScale);

    inputErrors[idx] = error * CReLUGate(activatedInputs[idx]);
}

// Feature transformer gradients, with each accumulator neuron's error backpropagated from L1 through
// the pairwise product in the same thread. Thread (x, y) owns accumulator neuron x of the block's
// y-th batch element; the L1 errors of the block's elements are staged in shared memory once.
static constexpr uint32_t c_ftGradientBlockRows = 16;

__global__ void FeatureTransformerGradientsKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ l1PreErrors,  // [batchSize][L1Size]
    const float* __restrict__ l1Weights,    // [numVariants][(L1InputSize + 1) * L1Size]
    const float* __restrict__ accumulators, // [batchSize][2][AccumulatorSize]
    float* __restrict__ weightGradients,
    uint32_t batchSize,
    uint32_t inputSize,
    float l1WeightScale, float l1InvWeightScale
)
{
    constexpr uint32_t accumulatorSize = nn::AccumulatorSize;
    constexpr uint32_t halfSize = accumulatorSize / 2;

    __shared__ float s_error[c_ftGradientBlockRows * nn::L1Size];

    const uint32_t accumulatorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t errorRow = threadIdx.y * nn::L1Size;

    if (threadIdx.x < nn::L1Size && batchIdx < batchSize)
        s_error[errorRow + threadIdx.x] = l1PreErrors[batchIdx * nn::L1Size + threadIdx.x];
    __syncthreads();

    if (batchIdx >= batchSize || accumulatorIdx >= accumulatorSize) return;

    const TrainingEntry* trainingVector = trainingVectors + batchIdx;
    const float* __restrict__ variantWeights = l1Weights + trainingVector->variant * (nn::L1InputSize + 1) * nn::L1Size;
    const uint32_t pairIdx = accumulatorIdx % halfSize;
    const uint32_t partnerIdx = accumulatorIdx < halfSize ? accumulatorIdx + halfSize : accumulatorIdx - halfSize;

    // The error of the pair's L1 input, times the derivative of the product with respect to this
    // neuron: the partner's activation, gated by this neuron's clipping.
    const auto accumulatorError = [&](uint32_t perspective)
    {
        // one output row is L1Size contiguous weights; read it four at a time
        const float4* __restrict__ w4 = reinterpret_cast<const float4*>(variantWeights + (perspective * halfSize + pairIdx) * nn::L1Size);

        float error = 0.0f;
        #pragma unroll
        for (uint32_t o = 0; o < nn::L1Size / 4; ++o)
        {
            const float4 wv = w4[o];
            error += s_error[errorRow + 4 * o + 0] * FakeQuantize(wv.x, l1WeightScale, l1InvWeightScale);
            error += s_error[errorRow + 4 * o + 1] * FakeQuantize(wv.y, l1WeightScale, l1InvWeightScale);
            error += s_error[errorRow + 4 * o + 2] * FakeQuantize(wv.z, l1WeightScale, l1InvWeightScale);
            error += s_error[errorRow + 4 * o + 3] * FakeQuantize(wv.w, l1WeightScale, l1InvWeightScale);
        }

        const uint32_t base = (2 * batchIdx + perspective) * accumulatorSize;
        return error * CReLU(accumulators[base + partnerIdx]) * CReLUGate(accumulators[base + accumulatorIdx]);
    };

    // Process white features
    const float whitesError = accumulatorError(0);
    if (whitesError != 0.0f)
    {
        for (uint32_t i = 0; i < trainingVector->numWhiteFeatures; ++i)
        {
            const uint32_t feature = trainingVector->whiteFeatures[i];
            if (feature >= inputSize) continue;

            atomicAdd(&weightGradients[feature * accumulatorSize + accumulatorIdx], whitesError);
        }
    }

    // Process black features
    const float blacksError = accumulatorError(1);
    if (blacksError != 0.0f)
    {
        for (uint32_t i = 0; i < trainingVector->numBlackFeatures; ++i)
        {
            const uint32_t feature = trainingVector->blackFeatures[i];
            if (feature >= inputSize) continue;

            atomicAdd(&weightGradients[feature * accumulatorSize + accumulatorIdx], blacksError);
        }
    }

    // bias gradient
    atomicAdd(&weightGradients[inputSize * accumulatorSize + accumulatorIdx], whitesError + blacksError);
}

void LaunchSigmoidDerivative(CudaBatchData& batch, cudaStream_t stream)
{
    const dim3 blockSize(c_sigmoidDerivativeBlockSize);
    const dim3 gridSize((batch.batchSize + blockSize.x - 1) / blockSize.x);

    SigmoidDerivativeKernel<<<gridSize, blockSize, 0, stream>>>(
        batch.networkOutputs.Get(),
        batch.trainingVectors.Get(),
        batch.outputErrors.Get(),
        batch.lossSum.Get(),
        batch.batchSize
    );
    CUDA_CHECK(cudaGetLastError());
}

void LaunchL3Backward(CudaBatchData& batch, const DeviceLayer& l3, cudaStream_t stream)
{
    {
        const dim3 blockSize(nn::L2Size, 1);
        const dim3 gridSize(1, c_l3GradientBatchSplits);

        DenseWeightGradientsKernel<1, nn::L2Size><<<gridSize, blockSize, 0, stream>>>(
            batch.trainingVectors.Get(),
            batch.l2Buffer.Get(),
            batch.outputErrors.Get(),
            batch.l3Gradients.Get(),
            batch.batchSize,
            nn::L2Size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batch.batchSize * nn::L2Size + blockSize.x - 1) / blockSize.x);

        BackpropToHiddenKernel<nn::L2Size, 1><<<gridSize, blockSize, 0, stream>>>(
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            l3.weights,
            batch.l2Buffer.Get(),
            batch.l2PreErrors.Get(),
            batch.batchSize,
            l3.quant.weightScale,
            l3.quant.invWeightScale
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

void LaunchL2Backward(CudaBatchData& batch, const DeviceLayer& l2, cudaStream_t stream)
{
    {
        const dim3 blockSize(nn::L1Size, nn::L2Size);
        const dim3 gridSize(1, c_l2GradientBatchSplits);

        DenseWeightGradientsKernel<nn::L2Size, nn::L1Size><<<gridSize, blockSize, 0, stream>>>(
            batch.trainingVectors.Get(),
            batch.l1Buffer.Get(),
            batch.l2PreErrors.Get(),
            batch.l2Gradients.Get(),
            batch.batchSize,
            nn::L1Size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batch.batchSize * nn::L1Size + blockSize.x - 1) / blockSize.x);

        BackpropToHiddenKernel<nn::L1Size, nn::L2Size><<<gridSize, blockSize, 0, stream>>>(
            batch.trainingVectors.Get(),
            batch.l2PreErrors.Get(),
            l2.weights,
            batch.l1Buffer.Get(),
            batch.l1PreErrors.Get(),
            batch.batchSize,
            l2.quant.weightScale,
            l2.quant.invWeightScale
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

void LaunchL1Gradients(CudaBatchData& batch, cudaStream_t stream)
{
    constexpr uint32_t inputTileSize = 16;
    const dim3 blockSize(inputTileSize, nn::L1Size);
    const dim3 gridSize(nn::L1InputSize / inputTileSize, c_l1GradientBatchSplits);

    DenseWeightGradientsKernel<nn::L1Size, inputTileSize><<<gridSize, blockSize, 0, stream>>>(
        batch.trainingVectors.Get(),
        batch.pairwiseBuffer.Get(),
        batch.l1PreErrors.Get(),
        batch.l1Gradients.Get(),
        batch.batchSize,
        nn::L1InputSize
    );
    CUDA_CHECK(cudaGetLastError());
}

void LaunchFeatureTransformerGradients(CudaBatchData& batch, const DeviceLayer& l1, cudaStream_t stream)
{
    {
        const dim3 blockSize(32, c_ftGradientBlockRows);
        const dim3 gridSize(
            (nn::AccumulatorSize + blockSize.x - 1) / blockSize.x,
            (batch.batchSize + blockSize.y - 1) / blockSize.y);

        FeatureTransformerGradientsKernel<<<gridSize, blockSize, 0, stream>>>(
            batch.trainingVectors.Get(),
            batch.l1PreErrors.Get(),
            l1.weights,
            batch.accumulatorBuffer.Get(),
            batch.featureTransformerGradients.Get(),
            batch.batchSize,
            FeatureTransformerInputs,
            l1.quant.weightScale,
            l1.quant.invWeightScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

#if USE_FACTORIZER
    {
        const dim3 blockSize(256);
        const dim3 gridSize((FactorizerInputs * nn::AccumulatorSize + blockSize.x - 1) / blockSize.x);

        FactorizerGradientsKernel<<<gridSize, blockSize, 0, stream>>>(
            batch.featureTransformerGradients.Get(),
            nn::AccumulatorSize
        );
        CUDA_CHECK(cudaGetLastError());
    }
#endif // USE_FACTORIZER
}

} // namespace cuda
} // namespace nn
