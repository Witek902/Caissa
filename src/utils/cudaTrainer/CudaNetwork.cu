#include "CudaNetwork.hpp"

#include <algorithm>

namespace nn {
namespace cuda {

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

__device__ __forceinline__ float SCReLU(float x)
{
    const float y = fminf(1.0f, fmaxf(0.0f, x));
    return y * y;
}

__device__ __forceinline__ float SCReLUDerivative(float x)
{
    return (x > 0.0f && x < 1.0f) ? (2.0f * x) : 0.0f;
}

// Quantization-aware training: fake-quantize a weight/bias on read, matching the round-to-grid
// done at pack time. The float master is left untouched; the straight-through estimator means the
// gradient flows back to the float master as if no rounding happened. A scale of 0 disables it.
__device__ __forceinline__ float FakeQuantize(float w, float scale, float invScale)
{
    return (scale > 0.0f) ? (roundf(w * scale) * invScale) : w;
}

CudaNeuralNetwork::CudaNeuralNetwork()
{
    CUDA_CHECK(cudaEventCreateWithFlags(&m_ftGradConsumedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_ftGradClearedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_trainConsumedEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_copyDoneEvent, cudaEventDisableTiming));

    // Pre-record so the first iteration's cross-stream waits are already satisfied.
    CUDA_CHECK(cudaEventRecord(m_ftGradConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_ftGradClearedEvent, m_auxStream.Get()));
    CUDA_CHECK(cudaEventRecord(m_trainConsumedEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventRecord(m_copyDoneEvent, m_copyStream.Get()));
}

CudaNeuralNetwork::~CudaNeuralNetwork()
{
    cudaEventDestroy(m_ftGradConsumedEvent);
    cudaEventDestroy(m_ftGradClearedEvent);
    cudaEventDestroy(m_trainConsumedEvent);
    cudaEventDestroy(m_copyDoneEvent);
}

void CudaNeuralNetwork::Init(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights)
{
    // Create CUDA weight storages based on host network
    m_featureTransformerWeights = std::make_shared<CudaWeightsStorage>(
        c_numNetworkInputs, c_accumulatorSize, 1
    );
    m_featureTransformerWeights->Init(c_numNetworkInputs);

    m_lastLayerWeights = std::make_shared<CudaWeightsStorage>(
        2 * c_accumulatorSize, 1, c_numVariants
    );
    m_lastLayerWeights->Init(2 * c_accumulatorSize);

    // Copy initial weights from host
    m_featureTransformerWeights->CopyFromHost(*featureTransformerWeights);
    m_lastLayerWeights->CopyFromHost(*lastLayerWeights);

    CopyWeightsFromHost(featureTransformerWeights, lastLayerWeights);

    // Quantization-aware training: weights/biases are fake-quantized on read in the forward/backward
    // kernels using the same scales as the final pack step.
    m_featureTransformerWeights->m_weightQuantScale = nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasQuantScale = nn::InputLayerBiasQuantizationScale;
    m_lastLayerWeights->m_weightQuantScale = nn::OutputLayerWeightQuantizationScale;
    m_lastLayerWeights->m_biasQuantScale = nn::OutputLayerBiasQuantizationScale;
}

void CudaNeuralNetwork::SetWeightDecay(float featureTransformerDecay, float lastLayerDecay)
{
    m_featureTransformerWeights->m_weightDecay = featureTransformerDecay;
    m_lastLayerWeights->m_weightDecay = lastLayerDecay;
}

void CudaNeuralNetwork::CopyWeightsFromHost(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights)
{
    m_featureTransformerWeights->CopyFromHost(*featureTransformerWeights);
    m_lastLayerWeights->CopyFromHost(*lastLayerWeights);
}

void CudaNeuralNetwork::CopyWeightsToHost(const nn::WeightsStoragePtr& featureTransformerWeights, const nn::WeightsStoragePtr& lastLayerWeights) const
{
    m_featureTransformerWeights->CopyToHost(*featureTransformerWeights);
    m_lastLayerWeights->CopyToHost(*lastLayerWeights);
}

// CUDA kernel for sparse binary input accumulation
__global__ void SparseBinaryInputKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ weights,
    float* __restrict__ accumulators,
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t accumulatorSize,
    float weightScale, float biasScale, float invWeightScale, float invBiasScale
)
{
    const uint32_t accumulatorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize || accumulatorIdx >= accumulatorSize) return;

    const TrainingEntry* trainingVector = trainingVectors + batchIdx;

    // biases are stored right after the weight matrix
    const float bias = FakeQuantize(weights[inputSize * accumulatorSize + accumulatorIdx], biasScale, invBiasScale);

    // Process white features
    float whiteSum = bias;
    for (uint32_t i = 0; i < trainingVector->numWhiteFeatures; ++i)
    {
        const uint16_t feature = trainingVector->whiteFeatures[i];
        if (feature >= inputSize) continue;

        whiteSum += FakeQuantize(weights[feature * accumulatorSize + accumulatorIdx], weightScale, invWeightScale);
    }
    accumulators[2 * batchIdx * accumulatorSize + accumulatorIdx] = whiteSum;

    // Process black features
    float blackSum = bias;
    for (uint32_t i = 0; i < trainingVector->numBlackFeatures; ++i)
    {
        const uint16_t feature = trainingVector->blackFeatures[i];
        if (feature >= inputSize) continue;

        blackSum += FakeQuantize(weights[feature * accumulatorSize + accumulatorIdx], weightScale, invWeightScale);
    }
    accumulators[2 * batchIdx * accumulatorSize + accumulatorSize + accumulatorIdx] = blackSum;
}

// CUDA kernel for fully connected layer (last layer)
__global__ void FullyConnectedKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ outputs,
    uint32_t batchSize,
    uint32_t inputSize,
    float weightScale, float biasScale, float invWeightScale, float invBiasScale
)
{
    // One warp per batch element: lanes stride the input dimension (coalesced reads),
    // then a warp reduction combines the partial dot products.
    const uint32_t batchIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const uint32_t lane = threadIdx.x & 31u;
    if (batchIdx >= batchSize) return;

    const uint32_t variant = trainingVectors[batchIdx].variant;
    const float* __restrict__ in = inputs + batchIdx * inputSize;
    const float* __restrict__ w = weights + variant * (inputSize + 1);

    float sum = 0.0f;
    for (uint32_t i = lane; i < inputSize; i += 32u)
    {
        sum += SCReLU(in[i]) * FakeQuantize(w[i], weightScale, invWeightScale);
    }

    // Warp reduction
    #pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    if (lane == 0u)
    {
        outputs[batchIdx] = sum + FakeQuantize(w[inputSize], biasScale, invBiasScale); // add bias
    }
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

void CudaNeuralNetwork::Forward(CudaBatchData& batch)
{
    const uint32_t batchSize = batch.batchSize;

    // Clear the FT gradient buffer on a separate stream so it overlaps the forward pass.
    // Wait until the previous iteration's FT Adam finished reading the buffer, clear it, then
    // signal completion for the backward FT-gradient accumulation to wait on.
    CUDA_CHECK(cudaStreamWaitEvent(m_auxStream.Get(), m_ftGradConsumedEvent, 0));
    batch.featureTransformerGradients.ClearAsync(m_auxStream.Get());
    CUDA_CHECK(cudaEventRecord(m_ftGradClearedEvent, m_auxStream.Get()));

    // The forward pass reads the training vectors; wait for this batch's copy to complete.
    CUDA_CHECK(cudaStreamWaitEvent(m_stream.Get(), m_copyDoneEvent, 0));

    // Sparse binary input accumulation (also initializes accumulators from biases)
    {
        const dim3 blockSize(32, 16);
        const dim3 gridSize(
            (c_accumulatorSize + blockSize.x - 1) / blockSize.x,
            (batchSize + blockSize.y - 1) / blockSize.y);

        SparseBinaryInputKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            m_featureTransformerWeights->m_weights.Get(),
            batch.accumulatorBuffer.Get(),
            batchSize,
            c_numNetworkInputs,
            c_accumulatorSize,
            m_featureTransformerWeights->m_weightQuantScale,
            m_featureTransformerWeights->m_biasQuantScale,
            1.0f / m_featureTransformerWeights->m_weightQuantScale,
            1.0f / m_featureTransformerWeights->m_biasQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Fully connected layer (last layer)
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize * 32 + blockSize.x - 1) / blockSize.x); // one warp per batch element

        FullyConnectedKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.accumulatorBuffer.Get(),
            m_lastLayerWeights->m_weights.Get(),
            batch.hiddenBuffer.Get(),
            batchSize,
            2 * c_accumulatorSize,
            m_lastLayerWeights->m_weightQuantScale,
            m_lastLayerWeights->m_biasQuantScale,
            1.0f / m_lastLayerWeights->m_weightQuantScale,
            1.0f / m_lastLayerWeights->m_biasQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Sigmoid activation (final output)
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);

        SigmoidActivationKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.hiddenBuffer.Get(),
            batch.networkOutputs.Get(),
            batchSize
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

// Backward pass kernels
__global__ void SigmoidDerivativeKernel(
    const float* __restrict__ outputs,
    const TrainingEntry* __restrict__ trainingVectors,
    float* __restrict__ outputErrors,
    uint32_t batchSize
)
{
    const uint32_t batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize) return;

    const float output = outputs[batchIdx];
    const float target = trainingVectors[batchIdx].targetOutput;
    const float derivative = output * (1.0f - output);
    outputErrors[batchIdx] = 2.0f * (output - target) * derivative;
}

__global__ void LastLayerGradientsKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ accumulatorBuffer,
    const float* __restrict__ outputErrors,
    float* __restrict__ weightGradients,
    uint32_t batchSize,
    uint32_t inputSize
)
{
    // threadIdx.x selects the input (inputSize is the bias slot); threadIdx.y splits the
    // batch reduction so we read each accumulator value once across all variants instead of
    // re-scanning the whole batch per variant. Per-variant partials are kept in registers and
    // combined across threadIdx.y via shared memory.
    const uint32_t inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t numSplits = blockDim.y;
    const bool active = (inputIdx <= inputSize);
    const bool isBias = (inputIdx == inputSize);

    float gradients[nn::NumVariants];
    #pragma unroll
    for (uint32_t v = 0; v < nn::NumVariants; ++v) gradients[v] = 0.0f;

    if (active)
    {
        for (uint32_t batchIdx = threadIdx.y; batchIdx < batchSize; batchIdx += numSplits)
        {
            const uint32_t variant = trainingVectors[batchIdx].variant;
            const float error = outputErrors[batchIdx];
            const float contribution = isBias ? error
                : SCReLU(accumulatorBuffer[batchIdx * inputSize + inputIdx]) * error;

            #pragma unroll
            for (uint32_t v = 0; v < nn::NumVariants; ++v)
            {
                if (variant == v) gradients[v] += contribution;
            }
        }
    }

    // Reduce per-variant partials across threadIdx.y. Layout: [blockDim.y][blockDim.x][NumVariants].
    extern __shared__ float shared[];
    const uint32_t slot = (threadIdx.y * blockDim.x + threadIdx.x) * nn::NumVariants;
    #pragma unroll
    for (uint32_t v = 0; v < nn::NumVariants; ++v) shared[slot + v] = gradients[v];
    __syncthreads();

    if (threadIdx.y == 0 && active)
    {
        #pragma unroll
        for (uint32_t v = 0; v < nn::NumVariants; ++v)
        {
            float sum = 0.0f;
            for (uint32_t s = 0; s < numSplits; ++s)
            {
                sum += shared[(s * blockDim.x + threadIdx.x) * nn::NumVariants + v];
            }
            weightGradients[v * (inputSize + 1) + inputIdx] = sum;
        }
    }
}

__global__ void BackpropToCReLUKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ outputErrors,
    const float* __restrict__ creluInputs,
    float* __restrict__ creluErrors,
    const float* __restrict__ weights,
    uint32_t batchSize,
    uint32_t inputSize,
    float weightScale, float invWeightScale
)
{
    const uint32_t inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y;
    if (batchIdx >= batchSize || inputIdx >= inputSize) return;

    const uint32_t v = trainingVectors[batchIdx].variant;
    const uint32_t weightsOffset = v * (inputSize + 1);

    const float error = outputErrors[batchIdx];
    // Use the same fake-quantized weight as the forward pass (straight-through estimator).
    const float w = FakeQuantize(weights[weightsOffset + inputIdx], weightScale, invWeightScale);
    const float x = creluInputs[batchIdx * inputSize + inputIdx];

    creluErrors[batchIdx * inputSize + inputIdx] = error * w * SCReLUDerivative(x);
}

__global__ void FeatureTransformerGradientsKernel(
    const float* __restrict__ creluErrors,
    const TrainingEntry* __restrict__ trainingVectors,
    float* __restrict__ weightGradients,
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t accumulatorSize
)
{
    const uint32_t accumulatorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize || accumulatorIdx >= accumulatorSize) return;

    const TrainingEntry* trainingVector = trainingVectors + batchIdx;

    // Process white features
    const float whitesError = creluErrors[2 * batchIdx * accumulatorSize + accumulatorIdx];
    if (whitesError != 0.0f)
    {
        // weights gradients
        for (uint32_t i = 0; i < trainingVector->numWhiteFeatures; ++i)
        {
            const uint32_t feature = trainingVector->whiteFeatures[i];
            if (feature >= inputSize) continue;

            const uint32_t gradientIdx = feature * accumulatorSize + accumulatorIdx;
            atomicAdd(&weightGradients[gradientIdx], whitesError);
        }
    }

    // Process black features
    const float blacksError = creluErrors[2 * batchIdx * accumulatorSize + accumulatorSize + accumulatorIdx];
    if (blacksError != 0.0f)
    {
        // weight gradients
        for (uint32_t i = 0; i < trainingVector->numBlackFeatures; ++i)
        {
            const uint32_t feature = trainingVector->blackFeatures[i];
            if (feature >= inputSize) continue;

            const uint32_t gradientIdx = feature * accumulatorSize + accumulatorIdx;
            atomicAdd(&weightGradients[gradientIdx], blacksError);
        }
    }

    // bias gradient
    const uint32_t biasGradientIdx = inputSize * accumulatorSize + accumulatorIdx;
    atomicAdd(&weightGradients[biasGradientIdx], (whitesError + blacksError));
}

void CudaNeuralNetwork::Backward(CudaBatchData& batch, float learningRate, size_t iteration)
{
    const uint32_t batchSize = batch.batchSize;

    // Compute output layer error (sigmoid derivative)
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);
        SigmoidDerivativeKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.networkOutputs.Get(),
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            batchSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Compute last layer weight gradients
    {
        const dim3 blockSize(32, 8); // x: input index, y: batch-split for the reduction
        const dim3 gridSize(((2 * c_accumulatorSize + 1) + blockSize.x - 1) / blockSize.x);
        const uint32_t sharedBytes = blockSize.x * blockSize.y * c_numVariants * sizeof(float);
        LastLayerGradientsKernel<<<gridSize, blockSize, sharedBytes, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.accumulatorBuffer.Get(),
            batch.outputErrors.Get(),
            batch.lastLayerGradients.Get(),
            batchSize,
            2 * c_accumulatorSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const dim3 blockSize(256);
        const dim3 gridSize((2 * c_accumulatorSize + blockSize.x - 1) / blockSize.x, batchSize);
        BackpropToCReLUKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            batch.accumulatorBuffer.Get(),
            batch.creluErrors.Get(),
            m_lastLayerWeights->m_weights.Get(),
            batchSize,
            2 * c_accumulatorSize,
            m_lastLayerWeights->m_weightQuantScale,
            1.0f / m_lastLayerWeights->m_weightQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Compute feature transformer gradients. The buffer was cleared on the aux stream
    // (overlapping the forward pass); wait for that clear to complete before accumulating.
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_stream.Get(), m_ftGradClearedEvent, 0));

        const dim3 blockSize(32, 16);
        const dim3 gridSize(
            (c_accumulatorSize + blockSize.x - 1) / blockSize.x,
            (batchSize + blockSize.y - 1) / blockSize.y);

        FeatureTransformerGradientsKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.creluErrors.Get(),
            batch.trainingVectors.Get(),
            batch.featureTransformerGradients.Get(),
            batchSize,
            c_numNetworkInputs,
            c_accumulatorSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // The training-vectors buffer is no longer read after the FT gradient accumulation (the Adam
    // updates below don't touch it), so signal that the next batch's copy may overwrite it.
    CUDA_CHECK(cudaEventRecord(m_trainConsumedEvent, m_stream.Get()));

    // Update last layer weights
    m_lastLayerWeights->UpdateAdam(
        batch.lastLayerGradients.Get(),
        learningRate,
        m_stream.Get()
    );

    // Update feature transformer weights
    m_featureTransformerWeights->UpdateAdam(
        batch.featureTransformerGradients.Get(),
        learningRate,
        m_stream.Get()
    );

    // Signal that the FT gradient buffer is no longer needed on the main stream, so the next
    // iteration's overlapped clear (on the aux stream) may proceed.
    CUDA_CHECK(cudaEventRecord(m_ftGradConsumedEvent, m_stream.Get()));
}

void CudaNeuralNetwork::CopyTrainingBatchAsync(CudaBatchData& batch, const TrainingEntry* hostSrc, uint32_t count)
{
    // Wait until the previous batch's last reader of the training-vectors buffer
    // (FeatureTransformerGradientsKernel) has finished, then copy on the dedicated copy stream so
    // it overlaps the previous batch's Adam updates. Forward waits on m_copyDoneEvent before use.
    CUDA_CHECK(cudaStreamWaitEvent(m_copyStream.Get(), m_trainConsumedEvent, 0));
    batch.trainingVectors.CopyFromHost(hostSrc, count, m_copyStream.Get());
    CUDA_CHECK(cudaEventRecord(m_copyDoneEvent, m_copyStream.Get()));
}

} // namespace cuda
} // namespace nn
