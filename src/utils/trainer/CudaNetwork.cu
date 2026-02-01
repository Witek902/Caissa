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

__device__ __forceinline__ float CReLU(float x)
{
    return fminf(1.0f, fmaxf(0.0f, x));
}

__device__ __forceinline__ float CReLUDerivative(float x)
{
    return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
}

CudaNeuralNetwork::CudaNeuralNetwork() = default;
CudaNeuralNetwork::~CudaNeuralNetwork() = default;

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

// CUDA kernel for bias addition
__global__ void CopyBiasesKernel(
    float* __restrict__ accumulators,
    const float* __restrict__ biases,
    uint32_t batchSize,
    uint32_t accumulatorSize
)
{
    const uint32_t accumulatorIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batchIdx >= batchSize || accumulatorIdx >= accumulatorSize) return;

    accumulators[batchIdx * accumulatorSize + accumulatorIdx] = biases[accumulatorIdx];
}

// CUDA kernel for sparse binary input accumulation
__global__ void SparseBinaryInputKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ weights,
    float* __restrict__ accumulators,
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
    for (uint32_t i = 0; i < trainingVector->numWhiteFeatures; ++i)
    {
        const uint16_t feature = trainingVector->whiteFeatures[i];
        if (feature >= inputSize) continue;

        const float weight = weights[feature * accumulatorSize + accumulatorIdx];
        accumulators[2 * batchIdx * accumulatorSize + accumulatorIdx] += weight;
    }

    // Process black features
    for (uint32_t i = 0; i < trainingVector->numBlackFeatures; ++i)
    {
        const uint16_t feature = trainingVector->blackFeatures[i];
        if (feature >= inputSize) continue;

        const float weight = weights[feature * accumulatorSize + accumulatorIdx];
        accumulators[2 * batchIdx * accumulatorSize + accumulatorSize + accumulatorIdx] += weight;
    }
}

// CUDA kernel for CReLU activation
__global__ void CReLUActivationKernel(
    const float* __restrict__ inputs,
    float* __restrict__ outputs,
    uint32_t bufferSize)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bufferSize) return;
    outputs[idx] = CReLU(inputs[idx]);
}

// CUDA kernel for fully connected layer (last layer)
__global__ void FullyConnectedKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ outputs,
    uint32_t batchSize,
    uint32_t inputSize
)
{
    const uint32_t batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize) return;

    const uint32_t variant = trainingVectors[batchIdx].variant;
    const uint32_t weightsOffset = variant * (inputSize + 1);

    // Single output per batch
    float sum = weights[weightsOffset + inputSize];
    for (uint32_t i = 0; i < inputSize; ++i)
    {
        sum += inputs[batchIdx * inputSize + i] * weights[weightsOffset + i];
    }

    outputs[batchIdx] = sum;
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

    // TODO merge with kernel below
    // Copy biases to accumulators
    {
        const dim3 blockSize(32, 16);
        const dim3 gridSize(
            (c_accumulatorSize + blockSize.x - 1) / blockSize.x,
            (batchSize * 2 + blockSize.y - 1) / blockSize.y);

        const float* biases = m_featureTransformerWeights->m_weights.Get() + c_numNetworkInputs * c_accumulatorSize;

        CopyBiasesKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.accumulatorBuffer.Get(),
            biases,
            batchSize * 2,
            c_accumulatorSize
            );
        CUDA_CHECK(cudaGetLastError());
    }

    // Sparse binary input accumulation
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
            c_accumulatorSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // CReLU activation
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize * 2 * c_accumulatorSize + blockSize.x - 1) / blockSize.x);
        
        CReLUActivationKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.accumulatorBuffer.Get(),
            batch.activationBuffer.Get(),
            c_accumulatorSize * batchSize * 2
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Fully connected layer (last layer)
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);

        FullyConnectedKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.activationBuffer.Get(),
            m_lastLayerWeights->m_weights.Get(),
            batch.hiddenBuffer.Get(),
            batchSize,
            2 * c_accumulatorSize
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
    const float* __restrict__ activations,
    const float* __restrict__ outputErrors,
    float* __restrict__ weightGradients,
    uint32_t batchSize,
    uint32_t inputSize
)
{
    const uint32_t inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t variantIdx = threadIdx.y;

    const uint32_t weightsOffset = variantIdx * (inputSize + 1);

    if (inputIdx < inputSize) // Weights gradients
    {
        float gradient = 0.0f;
        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const TrainingEntry* trainingVector = trainingVectors + batchIdx;
            if (trainingVector->variant != variantIdx) continue;
            gradient += activations[batchIdx * inputSize + inputIdx] * outputErrors[batchIdx];
        }
        weightGradients[weightsOffset + inputIdx] = gradient;
    }
    else if (inputIdx == inputSize) // Bias gradient
    {
        float gradient = 0.0f;
        for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const TrainingEntry* trainingVector = trainingVectors + batchIdx;
            if (trainingVector->variant != variantIdx) continue;
            gradient += outputErrors[batchIdx];
        }
        weightGradients[weightsOffset + inputSize] = gradient;
    }
}

__global__ void BackpropToCReLUKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ outputErrors,
    const float* __restrict__ creluInputs,
    float* __restrict__ creluErrors,
    const float* __restrict__ weights,
    uint32_t batchSize,
    uint32_t inputSize
)
{ 
    const uint32_t inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y;
    if (batchIdx >= batchSize || inputIdx >= inputSize) return;

    const uint32_t v = trainingVectors[batchIdx].variant;
    const uint32_t weightsOffset = v * (inputSize + 1);

    const float error = outputErrors[batchIdx];
    const float w = weights[weightsOffset + inputIdx];
    const float x = creluInputs[batchIdx * inputSize + inputIdx];

    creluErrors[batchIdx * inputSize + inputIdx] = error * w * CReLUDerivative(x);
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
        const dim3 blockSize(32, c_numVariants);
        const dim3 gridSize(((2 * c_accumulatorSize + 1) + blockSize.x - 1) / blockSize.x);
        LastLayerGradientsKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.activationBuffer.Get(),
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
            2 * c_accumulatorSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Compute feature transformer gradients
    {
        // clear gradients buffer
        batch.featureTransformerGradients.ClearAsync(m_stream.Get());
        
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

    // Update last layer weights
    m_lastLayerWeights->UpdateAdam(
        batch.lastLayerGradients.Get(),
        learningRate,
        iteration,
        m_stream.Get()
    );

    // Update feature transformer weights
    m_featureTransformerWeights->UpdateAdam(
        batch.featureTransformerGradients.Get(),
        learningRate,
        iteration,
        m_stream.Get()
    );
}

} // namespace cuda
} // namespace nn
