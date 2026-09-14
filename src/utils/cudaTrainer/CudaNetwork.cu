#include "CudaNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace nn {
namespace cuda {

namespace {

constexpr uint32_t c_checkpointMagic = 'CCKP';
constexpr uint32_t c_checkpointVersion = 1;

struct CheckpointHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t numLayers;
    uint32_t padding;
    uint64_t numTrainingVectorsPassed;
};

// Repeated per layer, so a checkpoint written for a different architecture is rejected instead of
// being read as garbage.
struct CheckpointLayerHeader
{
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t numVariants;
    uint32_t padding;
    uint64_t adamStep;
};

} // namespace


// Number of slices the batch is split into when reducing dense weight gradients. Each slice is a
// separate block, so the reduction has enough parallelism for the small output subnet layers.
static constexpr uint32_t c_l1GradientBatchSplits = 16;
static constexpr uint32_t c_l2GradientBatchSplits = 64;
static constexpr uint32_t c_l3GradientBatchSplits = 256;

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

__device__ __forceinline__ float CReLU(float x)
{
    return fminf(1.0f, fmaxf(0.0f, x));
}

// Gate for backpropagating through CReLU, evaluated on the post-activation value.
__device__ __forceinline__ float CReLUGate(float activated)
{
    return (activated > 0.0f && activated < 1.0f) ? 1.0f : 0.0f;
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

    // Unlike the synchronization-only events above, these must keep timing enabled.
    CUDA_CHECK(cudaEventCreate(&m_iterationStartEvent));
    CUDA_CHECK(cudaEventCreate(&m_iterationEndEvent));

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
    cudaEventDestroy(m_iterationStartEvent);
    cudaEventDestroy(m_iterationEndEvent);
}

void CudaNeuralNetwork::BeginIterationTiming()
{
    CUDA_CHECK(cudaEventRecord(m_iterationStartEvent, m_stream.Get()));
}

float CudaNeuralNetwork::EndIterationTimingMs()
{
    CUDA_CHECK(cudaEventRecord(m_iterationEndEvent, m_stream.Get()));
    CUDA_CHECK(cudaEventSynchronize(m_iterationEndEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, m_iterationStartEvent, m_iterationEndEvent));
    return elapsedMs;
}

void CudaNeuralNetwork::InitRandomWeights(uint32_t seed)
{
    // scaled so that the accumulator of a ~32 feature position lands in the CReLU range
    m_featureTransformerWeights->Init(seed, 0.1f);

#if USE_FACTORIZER
    // the factorizer starts at zero, so the effective weights are just the bucket weights
    CUDA_CHECK(cudaMemset(
        m_featureTransformerWeights->m_weights.Get() + c_numNetworkInputs * c_accumulatorSize,
        0, FactorizerInputs * c_accumulatorSize * sizeof(float)));
#endif // USE_FACTORIZER

    InitRandomOutputSubnetWeights(seed);
}

void CudaNeuralNetwork::InitRandomOutputSubnetWeights(uint32_t seed)
{
    // He initialization, appropriate for the (clipped) ReLU activations of the output subnet
    m_l1Weights->Init(seed + 1, sqrtf(2.0f / (float)c_l1InputSize));
    m_l2Weights->Init(seed + 2, sqrtf(2.0f / (float)c_l1Size));
    m_l3Weights->Init(seed + 3, sqrtf(2.0f / (float)c_l2Size));
}

void CudaNeuralNetwork::Init(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    // Create CUDA weight storages based on host network
    m_featureTransformerWeights = std::make_shared<CudaWeightsStorage>(
        FeatureTransformerInputs, c_accumulatorSize, 1
    );
#if USE_FACTORIZER
    m_featureTransformerWeights->m_factorizerFirstWeight = c_numNetworkInputs * c_accumulatorSize;
    m_featureTransformerWeights->m_factorizerRange = FactorizerWeightRange;
#endif // USE_FACTORIZER

    m_l1Weights = std::make_shared<CudaWeightsStorage>(c_l1InputSize, c_l1Size, c_numVariants);
    m_l2Weights = std::make_shared<CudaWeightsStorage>(c_l1Size, c_l2Size, c_numVariants);
    m_l3Weights = std::make_shared<CudaWeightsStorage>(c_l2Size, 1, c_numVariants);

    CopyWeightsFromHost(featureTransformerWeights, l1Weights, l2Weights, l3Weights);

    // Quantization-aware training: weights/biases are fake-quantized on read in the forward/backward
    // kernels using the same scales as the final pack step.
    m_featureTransformerWeights->m_weightQuantScale = nn::InputLayerWeightQuantizationScale;
    m_featureTransformerWeights->m_biasQuantScale = nn::InputLayerBiasQuantizationScale;
    m_l1Weights->m_weightQuantScale = nn::HiddenLayerWeightQuantizationScale;
    m_l1Weights->m_biasQuantScale = nn::HiddenLayerBiasQuantizationScale;
    m_l2Weights->m_weightQuantScale = nn::HiddenLayerWeightQuantizationScale;
    m_l2Weights->m_biasQuantScale = nn::HiddenLayerBiasQuantizationScale;
    m_l3Weights->m_weightQuantScale = nn::OutputLayerWeightQuantizationScale;
    m_l3Weights->m_biasQuantScale = nn::OutputLayerBiasQuantizationScale;
}

void CudaNeuralNetwork::SetWeightDecay(float featureTransformerDecay, float outputSubnetDecay)
{
    m_featureTransformerWeights->m_weightDecay = featureTransformerDecay;
    m_l1Weights->m_weightDecay = outputSubnetDecay;
    m_l2Weights->m_weightDecay = outputSubnetDecay;
    m_l3Weights->m_weightDecay = outputSubnetDecay;
}

void CudaNeuralNetwork::SetFeatureTransformerFrozen(bool frozen)
{
    m_featureTransformerWeights->m_updateWeights = !frozen;
}

void CudaNeuralNetwork::CopyWeightsFromHost(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    m_featureTransformerWeights->CopyFromHost(*featureTransformerWeights);
    m_l1Weights->CopyFromHost(*l1Weights);
    m_l2Weights->CopyFromHost(*l2Weights);
    m_l3Weights->CopyFromHost(*l3Weights);
}

bool CudaNeuralNetwork::SaveCheckpoint(const char* path, uint64_t numTrainingVectorsPassed) const
{
    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights };

    FILE* file = fopen(path, "wb");
    if (!file)
    {
        std::cerr << "Failed to save checkpoint: cannot open " << path << std::endl;
        return false;
    }

    CheckpointHeader header{};
    header.magic = c_checkpointMagic;
    header.version = c_checkpointVersion;
    header.numLayers = (uint32_t)std::size(layers);
    header.numTrainingVectorsPassed = numTrainingVectorsPassed;

    bool ok = fwrite(&header, sizeof(header), 1, file) == 1;

    std::vector<float> weights, moment1, moment2;
    for (const CudaWeightsStoragePtr& layer : layers)
    {
        if (!ok) break;

        CheckpointLayerHeader layerHeader{};
        layerHeader.inputSize = layer->m_inputSize;
        layerHeader.outputSize = layer->m_outputSize;
        layerHeader.numVariants = layer->m_numVariants;
        layerHeader.adamStep = layer->m_adamStep;

        layer->CopyStateToHost(weights, moment1, moment2);

        ok = fwrite(&layerHeader, sizeof(layerHeader), 1, file) == 1
            && fwrite(weights.data(), sizeof(float), weights.size(), file) == weights.size()
            && fwrite(moment1.data(), sizeof(float), moment1.size(), file) == moment1.size()
            && fwrite(moment2.data(), sizeof(float), moment2.size(), file) == moment2.size();
    }

    fclose(file);

    if (!ok)
        std::cerr << "Failed to save checkpoint: cannot write " << path << std::endl;

    return ok;
}

bool CudaNeuralNetwork::LoadCheckpoint(const char* path, uint64_t& outNumTrainingVectorsPassed)
{
    const CudaWeightsStoragePtr layers[] = { m_featureTransformerWeights, m_l1Weights, m_l2Weights, m_l3Weights };

    FILE* file = fopen(path, "rb");
    if (!file)
    {
        std::cerr << "Failed to load checkpoint: cannot open " << path << std::endl;
        return false;
    }

    CheckpointHeader header{};
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        header.magic != c_checkpointMagic ||
        header.version != c_checkpointVersion ||
        header.numLayers != (uint32_t)std::size(layers))
    {
        fclose(file);
        std::cerr << "Failed to load checkpoint: " << path << " is not a compatible checkpoint" << std::endl;
        return false;
    }

    std::vector<float> weights, moment1, moment2;
    for (const CudaWeightsStoragePtr& layer : layers)
    {
        CheckpointLayerHeader layerHeader{};
        if (fread(&layerHeader, sizeof(layerHeader), 1, file) != 1 ||
            layerHeader.inputSize != layer->m_inputSize ||
            layerHeader.outputSize != layer->m_outputSize ||
            layerHeader.numVariants != layer->m_numVariants)
        {
            fclose(file);
            std::cerr << "Failed to load checkpoint: " << path << " was written for a different architecture" << std::endl;
            return false;
        }

        weights.resize(layer->m_totalWeights);
        moment1.resize(layer->m_totalWeights);
        moment2.resize(layer->m_totalWeights);

        const size_t count = layer->m_totalWeights;
        if (fread(weights.data(), sizeof(float), count, file) != count ||
            fread(moment1.data(), sizeof(float), count, file) != count ||
            fread(moment2.data(), sizeof(float), count, file) != count)
        {
            fclose(file);
            std::cerr << "Failed to load checkpoint: " << path << " is truncated" << std::endl;
            return false;
        }

        layer->CopyStateFromHost(weights, moment1, moment2);
        layer->m_adamStep = (size_t)layerHeader.adamStep;
    }

    fclose(file);
    outNumTrainingVectorsPassed = header.numTrainingVectorsPassed;
    return true;
}

void CudaNeuralNetwork::CopyWeightsToHost(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights) const
{
    m_featureTransformerWeights->CopyToHost(*featureTransformerWeights);
    m_l1Weights->CopyToHost(*l1Weights);
    m_l2Weights->CopyToHost(*l2Weights);
    m_l3Weights->CopyToHost(*l3Weights);
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

    // Sparse binary input accumulation (also initializes accumulators from biases) and the pairwise
    // activation of its output
    {
        const dim3 blockSize(32, 16);
        const dim3 gridSize(
            (c_accumulatorSize / 2 + blockSize.x - 1) / blockSize.x,
            (batchSize + blockSize.y - 1) / blockSize.y);

        SparseBinaryInputKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            m_featureTransformerWeights->m_weights.Get(),
            batch.accumulatorBuffer.Get(),
            batch.pairwiseBuffer.Get(),
            batchSize,
            FeatureTransformerInputs,
            c_accumulatorSize,
            m_featureTransformerWeights->m_weightQuantScale,
            m_featureTransformerWeights->m_biasQuantScale,
            1.0f / m_featureTransformerWeights->m_weightQuantScale,
            1.0f / m_featureTransformerWeights->m_biasQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // L1
    {
        L1ForwardKernel<<<batchSize, c_l1ForwardBlockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.pairwiseBuffer.Get(),
            m_l1Weights->m_weights.Get(),
            batch.l1Buffer.Get(),
            m_l1Weights->m_weightQuantScale,
            m_l1Weights->m_biasQuantScale,
            1.0f / m_l1Weights->m_weightQuantScale,
            1.0f / m_l1Weights->m_biasQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // L2
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize * c_l2Size + blockSize.x - 1) / blockSize.x);

        SmallDenseForwardKernel<nn::L1Size, nn::L2Size, true><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l1Buffer.Get(),
            m_l2Weights->m_weights.Get(),
            batch.l2Buffer.Get(),
            batchSize,
            m_l2Weights->m_weightQuantScale,
            m_l2Weights->m_biasQuantScale,
            1.0f / m_l2Weights->m_weightQuantScale,
            1.0f / m_l2Weights->m_biasQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // L3 (scalar output, no activation)
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);

        SmallDenseForwardKernel<nn::L2Size, 1, false><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l2Buffer.Get(),
            m_l3Weights->m_weights.Get(),
            batch.l3Buffer.Get(),
            batchSize,
            m_l3Weights->m_weightQuantScale,
            m_l3Weights->m_biasQuantScale,
            1.0f / m_l3Weights->m_weightQuantScale,
            1.0f / m_l3Weights->m_biasQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Sigmoid activation (final output)
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);

        SigmoidActivationKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.l3Buffer.Get(),
            batch.networkOutputs.Get(),
            batchSize
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

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

void CudaNeuralNetwork::Backward(CudaBatchData& batch, float learningRate)
{
    const uint32_t batchSize = batch.batchSize;

    // The dense gradient kernels accumulate atomically across batch slices
    batch.l1Gradients.ClearAsync(m_stream.Get());
    batch.l2Gradients.ClearAsync(m_stream.Get());
    batch.l3Gradients.ClearAsync(m_stream.Get());

    // Compute output layer error (sigmoid derivative)
    {
        const dim3 blockSize(c_sigmoidDerivativeBlockSize);
        const dim3 gridSize((batchSize + blockSize.x - 1) / blockSize.x);
        SigmoidDerivativeKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.networkOutputs.Get(),
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            batch.lossSum.Get(),
            batchSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // L3 gradients, then backprop to L2's pre-activation
    {
        const dim3 blockSize(c_l2Size, 1);
        const dim3 gridSize(1, c_l3GradientBatchSplits);
        DenseWeightGradientsKernel<1, nn::L2Size><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l2Buffer.Get(),
            batch.outputErrors.Get(),
            batch.l3Gradients.Get(),
            batchSize,
            c_l2Size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize * c_l2Size + blockSize.x - 1) / blockSize.x);
        BackpropToHiddenKernel<nn::L2Size, 1><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            m_l3Weights->m_weights.Get(),
            batch.l2Buffer.Get(),
            batch.l2PreErrors.Get(),
            batchSize,
            m_l3Weights->m_weightQuantScale,
            1.0f / m_l3Weights->m_weightQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // L2 gradients, then backprop to L1's pre-activation
    {
        const dim3 blockSize(c_l1Size, c_l2Size);
        const dim3 gridSize(1, c_l2GradientBatchSplits);
        DenseWeightGradientsKernel<nn::L2Size, nn::L1Size><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l1Buffer.Get(),
            batch.l2PreErrors.Get(),
            batch.l2Gradients.Get(),
            batchSize,
            c_l1Size
        );
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const dim3 blockSize(256);
        const dim3 gridSize((batchSize * c_l1Size + blockSize.x - 1) / blockSize.x);
        BackpropToHiddenKernel<nn::L1Size, nn::L2Size><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l2PreErrors.Get(),
            m_l2Weights->m_weights.Get(),
            batch.l1Buffer.Get(),
            batch.l1PreErrors.Get(),
            batchSize,
            m_l2Weights->m_weightQuantScale,
            1.0f / m_l2Weights->m_weightQuantScale
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // L1 gradients
    {
        constexpr uint32_t inputTileSize = 16;
        const dim3 blockSize(inputTileSize, c_l1Size);
        const dim3 gridSize(c_l1InputSize / inputTileSize, c_l1GradientBatchSplits);
        DenseWeightGradientsKernel<nn::L1Size, inputTileSize><<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.pairwiseBuffer.Get(),
            batch.l1PreErrors.Get(),
            batch.l1Gradients.Get(),
            batchSize,
            c_l1InputSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // The L1 -> feature transformer backprop and the FT weight gradients are only needed to update
    // the feature transformer, so both are skipped when it is frozen.
    if (m_featureTransformerWeights->m_updateWeights)
    {
        // Compute feature transformer gradients. The buffer was cleared on the aux stream
        // (overlapping the forward pass); wait for that clear to complete before accumulating.
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_stream.Get(), m_ftGradClearedEvent, 0));

            const dim3 blockSize(32, c_ftGradientBlockRows);
            const dim3 gridSize(
                (c_accumulatorSize + blockSize.x - 1) / blockSize.x,
                (batchSize + blockSize.y - 1) / blockSize.y);

            FeatureTransformerGradientsKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
                batch.trainingVectors.Get(),
                batch.l1PreErrors.Get(),
                m_l1Weights->m_weights.Get(),
                batch.accumulatorBuffer.Get(),
                batch.featureTransformerGradients.Get(),
                batchSize,
                FeatureTransformerInputs,
                m_l1Weights->m_weightQuantScale,
                1.0f / m_l1Weights->m_weightQuantScale
            );
            CUDA_CHECK(cudaGetLastError());
        }

#if USE_FACTORIZER
        {
            const dim3 blockSize(256);
            const dim3 gridSize((FactorizerInputs * c_accumulatorSize + blockSize.x - 1) / blockSize.x);
            FactorizerGradientsKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
                batch.featureTransformerGradients.Get(),
                c_accumulatorSize
            );
            CUDA_CHECK(cudaGetLastError());
        }
#endif // USE_FACTORIZER
    }

    // The training-vectors buffer is no longer read after the FT gradient accumulation (the Adam
    // updates below don't touch it), so signal that the next batch's copy may overwrite it.
    CUDA_CHECK(cudaEventRecord(m_trainConsumedEvent, m_stream.Get()));

    // Update output subnet weights
    m_l3Weights->UpdateAdam(batch.l3Gradients.Get(), learningRate, m_stream.Get());
    m_l2Weights->UpdateAdam(batch.l2Gradients.Get(), learningRate, m_stream.Get());
    m_l1Weights->UpdateAdam(batch.l1Gradients.Get(), learningRate, m_stream.Get());

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
