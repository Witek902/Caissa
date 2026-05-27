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

static constexpr uint32_t c_l1Size = nn::L1Size;
static constexpr uint32_t c_l2Size = nn::L2Size;

void CudaNeuralNetwork::Init(
    const nn::WeightsStoragePtr& featureTransformerWeights,
    const nn::WeightsStoragePtr& l1Weights,
    const nn::WeightsStoragePtr& l2Weights,
    const nn::WeightsStoragePtr& l3Weights)
{
    m_featureTransformerWeights = std::make_shared<CudaWeightsStorage>(
        c_numNetworkInputs, c_accumulatorSize, 1);
    m_featureTransformerWeights->Init(c_numNetworkInputs);

    m_l1Weights = std::make_shared<CudaWeightsStorage>(
        2 * c_accumulatorSize, c_l1Size, c_numVariants);
    m_l1Weights->Init(2 * c_accumulatorSize);

    m_l2Weights = std::make_shared<CudaWeightsStorage>(
        c_l1Size, c_l2Size, c_numVariants);
    m_l2Weights->Init(c_l1Size);

    m_l3Weights = std::make_shared<CudaWeightsStorage>(
        c_l2Size, 1, c_numVariants);
    m_l3Weights->Init(c_l2Size);

    m_errorAccumulator.Allocate(1);

    CopyWeightsFromHost(featureTransformerWeights, l1Weights, l2Weights, l3Weights);
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

// Hidden layer forward: dense float matmul with per-variant weights + CReLU on output.
// applyInputCReLU: true for L1 (FT accumulator is raw), false for L2/L3 (already activated).
template<bool applyInputCReLU, bool applyOutputCReLU>
__global__ void HiddenLayerForwardKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ inputs,       // [batchSize * inputSize]
    const float* __restrict__ weights,      // [numVariants * (inputSize*outputSize + outputSize)]
    float* __restrict__ outputs,            // [batchSize * outputSize]
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t outputSize)
{
    const uint32_t batchIdx  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t outputIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIdx >= batchSize || outputIdx >= outputSize) return;

    const uint32_t variant = trainingVectors[batchIdx].variant;
    // Layout: [variant][input * outputSize + output] then bias at [inputSize * outputSize + output]
    const uint32_t variantStride = inputSize * outputSize + outputSize;
    const float* w = weights + variant * variantStride;

    float sum = w[inputSize * outputSize + outputIdx]; // bias
    for (uint32_t j = 0; j < inputSize; ++j)
    {
        const float in = inputs[batchIdx * inputSize + j];
        sum += (applyInputCReLU ? CReLU(in) : in) * w[j * outputSize + outputIdx];
    }

    outputs[batchIdx * outputSize + outputIdx] = applyOutputCReLU ? CReLU(sum) : sum;
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

// Reduce sum of squared (output - target) over the batch and atomically add to a single
// device accumulator. Block-level reduction in shared memory keeps atomic traffic to one
// add per block. blockDim.x must be a power of two.
__global__ void ComputeErrorKernel(
    const float* __restrict__ networkOutputs,
    const TrainingEntry* __restrict__ trainingVectors,
    float* __restrict__ errorAccumulator,
    uint32_t batchSize
)
{
    extern __shared__ float sdata[];
    const uint32_t tid = threadIdx.x;
    const uint32_t idx = blockIdx.x * blockDim.x + tid;

    float v = 0.0f;
    if (idx < batchSize)
    {
        const float diff = networkOutputs[idx] - trainingVectors[idx].targetOutput;
        v = diff * diff;
    }
    sdata[tid] = v;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(errorAccumulator, sdata[0]);
}

void CudaNeuralNetwork::Forward(CudaBatchData& batch)
{
    const uint32_t batchSize = batch.batchSize;

    // Stage 1: FT — copy biases then accumulate sparse inputs
    {
        const dim3 blockSize(32, 16);
        const dim3 gridSize(
            (c_accumulatorSize + blockSize.x - 1) / blockSize.x,
            (batchSize * 2 + blockSize.y - 1) / blockSize.y);

        CopyBiasesKernel<<<gridSize, blockSize, 0, m_stream.Get()>>>(
            batch.accumulatorBuffer.Get(),
            m_featureTransformerWeights->m_weights.Get() + c_numNetworkInputs * c_accumulatorSize,
            batchSize * 2,
            c_accumulatorSize);
        CUDA_CHECK(cudaGetLastError());
    }
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
            c_accumulatorSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 2: L1 — CReLU(FT_accum) → L1 (applyInputCReLU=true, applyOutputCReLU=true)
    {
        const dim3 block(32, 16);
        const dim3 grid(
            (batchSize + block.x - 1) / block.x,
            (c_l1Size + block.y - 1) / block.y);
        HiddenLayerForwardKernel<true, true><<<grid, block, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.accumulatorBuffer.Get(),
            m_l1Weights->m_weights.Get(),
            batch.l1Buffer.Get(),
            batchSize,
            2 * c_accumulatorSize,
            c_l1Size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 3: L2 — L1_out (already CReLU'd) → L2 (applyInputCReLU=false, applyOutputCReLU=true)
    {
        const dim3 block(32, 16);
        const dim3 grid(
            (batchSize + block.x - 1) / block.x,
            (c_l2Size + block.y - 1) / block.y);
        HiddenLayerForwardKernel<false, true><<<grid, block, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l1Buffer.Get(),
            m_l2Weights->m_weights.Get(),
            batch.l2Buffer.Get(),
            batchSize,
            c_l1Size,
            c_l2Size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Stage 4: L3 — L2_out (already CReLU'd) → scalar (applyInputCReLU=false, applyOutputCReLU=false)
    {
        const dim3 block(256);
        const dim3 grid((batchSize + block.x - 1) / block.x);
        HiddenLayerForwardKernel<false, false><<<grid, block, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l2Buffer.Get(),
            m_l3Weights->m_weights.Get(),
            batch.l3Buffer.Get(),
            batchSize,
            c_l2Size,
            1u);
        CUDA_CHECK(cudaGetLastError());
    }

    // Sigmoid activation → final network output
    {
        const dim3 block(256);
        const dim3 grid((batchSize + block.x - 1) / block.x);
        SigmoidActivationKernel<<<grid, block, 0, m_stream.Get()>>>(
            batch.l3Buffer.Get(),
            batch.networkOutputs.Get(),
            batchSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Accumulate float-net squared error for this batch (sum, not yet averaged)
    {
        const uint32_t block = 256;
        const dim3 grid((batchSize + block - 1) / block);
        ComputeErrorKernel<<<grid, block, block * sizeof(float), m_stream.Get()>>>(
            batch.networkOutputs.Get(),
            batch.trainingVectors.Get(),
            m_errorAccumulator.Get(),
            batchSize);
        CUDA_CHECK(cudaGetLastError());
    }
}

void CudaNeuralNetwork::ResetError()
{
    m_errorAccumulator.ClearAsync(m_stream.Get());
}

double CudaNeuralNetwork::GetAccumulatedSquaredError() const
{
    float hostError = 0.0f;
    m_stream.Synchronize();
    m_errorAccumulator.CopyToHost(&hostError, 1);
    return (double)hostError;
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

// Per-variant dense weight gradients: dL/dW[in,out] = Σ_{b: variant(b)==v} actIn[b,in] * outGrad[b,out]
// One thread per (input, output). Each thread scans the batch ONCE, accumulating into a small
// register array indexed by the position's variant. This avoids re-scanning the whole batch per
// variant (and re-reading trainingVectors[b].variant 8x). nn::NumVariants accumulators live in
// registers; the inner unrolled compare keeps the array indices compile-time so it never spills.
// applyInputCReLU: when the layer's inputs are stored raw (pre-CReLU), apply CReLU here so the
//   weight gradient uses the actual activated input. Needed for L1 (raw FT accumulator); L2/L3
//   receive already-activated buffers (false).
template<bool applyInputCReLU>
__global__ void DenseWeightGradientsKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ activatedInputs, // [batchSize * inputSize]
    const float* __restrict__ outputGrads,     // [batchSize * outputSize]
    float* __restrict__ weightGradients,       // [(inputSize*outputSize + outputSize) * numVariants]
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t outputSize)
{
    const uint32_t in  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t out = blockIdx.y * blockDim.y + threadIdx.y;
    if (in >= inputSize || out >= outputSize) return;

    float g[nn::NumVariants];
    #pragma unroll
    for (uint32_t v = 0; v < nn::NumVariants; ++v) g[v] = 0.0f;

    for (uint32_t b = 0; b < batchSize; ++b)
    {
        const uint32_t v = trainingVectors[b].variant;
        float inVal = activatedInputs[b * inputSize + in];
        if (applyInputCReLU) inVal = CReLU(inVal);
        const float contrib = inVal * outputGrads[b * outputSize + out];
        #pragma unroll
        for (uint32_t vv = 0; vv < nn::NumVariants; ++vv)
            if (v == vv) g[vv] += contrib;
    }

    const uint32_t variantStride = inputSize * outputSize + outputSize;
    #pragma unroll
    for (uint32_t v = 0; v < nn::NumVariants; ++v)
        weightGradients[v * variantStride + in * outputSize + out] = g[v];
}

// Per-variant dense bias gradients: dL/dBias[out] = Σ_{b: variant(b)==v} outGrad[b,out]
__global__ void DenseBiasGradientsKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ outputGrads,     // [batchSize * outputSize]
    float* __restrict__ weightGradients,       // [(inputSize*outputSize + outputSize) * numVariants]
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t outputSize)
{
    const uint32_t out        = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t variantIdx = blockIdx.y;
    if (out >= outputSize) return;

    float g = 0.0f;
    for (uint32_t b = 0; b < batchSize; ++b)
    {
        if (trainingVectors[b].variant != variantIdx) continue;
        g += outputGrads[b * outputSize + out];
    }

    const uint32_t variantStride = inputSize * outputSize + outputSize;
    weightGradients[variantIdx * variantStride + inputSize * outputSize + out] = g;
}

// Launch helper: weight + bias gradient kernels for one dense layer.
template<bool applyInputCReLU>
static void LaunchDenseGradients(
    cudaStream_t stream,
    const TrainingEntry* trainingVectors,
    const float* activatedInputs,
    const float* outputGrads,
    float* weightGradients,
    uint32_t batchSize, uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
{
    // Weight kernel accumulates all variants per (input, output) thread — no variant grid dim.
    const dim3 wBlock(32, 8);
    const dim3 wGrid(
        (inputSize  + wBlock.x - 1) / wBlock.x,
        (outputSize + wBlock.y - 1) / wBlock.y);
    DenseWeightGradientsKernel<applyInputCReLU><<<wGrid, wBlock, 0, stream>>>(
        trainingVectors, activatedInputs, outputGrads, weightGradients,
        batchSize, inputSize, outputSize);
    CUDA_CHECK(cudaGetLastError());

    const dim3 bBlock(64);
    const dim3 bGrid((outputSize + bBlock.x - 1) / bBlock.x, numVariants);
    DenseBiasGradientsKernel<<<bGrid, bBlock, 0, stream>>>(
        trainingVectors, outputGrads, weightGradients,
        batchSize, inputSize, outputSize);
    CUDA_CHECK(cudaGetLastError());
}

// Backpropagates error through a dense layer to its inputs.
// outputGrads: dL/d(layer_output), applied through weights to produce inputErrors.
// If applyInputCReLUGate: gate by CReLU'(activatedInputs) (use post-CReLU value to detect gate).
template<bool applyInputCReLUGate>
__global__ void BackpropToHiddenKernel(
    const TrainingEntry* __restrict__ trainingVectors,
    const float* __restrict__ outputGrads,      // [batchSize * outputSize]
    const float* __restrict__ weights,          // [numVariants * (inputSize*outputSize + outputSize)]
    const float* __restrict__ activatedInputs,  // [batchSize * inputSize] (post-CReLU)
    float* __restrict__ inputErrors,            // [batchSize * inputSize]
    uint32_t batchSize,
    uint32_t inputSize,
    uint32_t outputSize)
{
    const uint32_t inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t batchIdx = blockIdx.y;
    if (batchIdx >= batchSize || inputIdx >= inputSize) return;

    const uint32_t v = trainingVectors[batchIdx].variant;
    const uint32_t variantStride = inputSize * outputSize + outputSize;
    const float* w = weights + v * variantStride + inputIdx * outputSize;

    float err = 0.0f;
    for (uint32_t o = 0; o < outputSize; ++o)
        err += outputGrads[batchIdx * outputSize + o] * w[o];

    if (applyInputCReLUGate)
    {
        // CReLU gate: derivative is 1 iff output is in (0,1), i.e. post-CReLU value ∈ (0,1)
        const float act = activatedInputs[batchIdx * inputSize + inputIdx];
        err *= (act > 0.0f && act < 1.0f) ? 1.0f : 0.0f;
    }

    inputErrors[batchIdx * inputSize + inputIdx] = err;
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

    // Step 1: dL/d(l3_input) = sigmoid_derivative
    {
        const dim3 block(256);
        const dim3 grid((batchSize + block.x - 1) / block.x);
        SigmoidDerivativeKernel<<<grid, block, 0, m_stream.Get()>>>(
            batch.networkOutputs.Get(),
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            batchSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: L3 weight/bias gradients and backprop to l2Buffer
    // outputErrors is [batchSize * 1] (single output); l2Buffer is already CReLU-activated
    LaunchDenseGradients<false>(
        m_stream.Get(),
        batch.trainingVectors.Get(),
        batch.l2Buffer.Get(),
        batch.outputErrors.Get(),
        batch.l3Gradients.Get(),
        batchSize, c_l2Size, 1u, c_numVariants);

    // Backprop L3 output error to L2 pre-activation (gate by L2's output CReLU)
    {
        const dim3 block(256);
        const dim3 grid((c_l2Size + block.x - 1) / block.x, batchSize);
        BackpropToHiddenKernel<true><<<grid, block, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.outputErrors.Get(),
            m_l3Weights->m_weights.Get(),
            batch.l2Buffer.Get(),
            batch.l2PreErrors.Get(),
            batchSize,
            c_l2Size,
            1u);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: L2 weight/bias gradients and backprop to l1Buffer
    // l1Buffer is already CReLU-activated
    LaunchDenseGradients<false>(
        m_stream.Get(),
        batch.trainingVectors.Get(),
        batch.l1Buffer.Get(),
        batch.l2PreErrors.Get(),
        batch.l2Gradients.Get(),
        batchSize, c_l1Size, c_l2Size, c_numVariants);

    // Backprop L2 pre-activation error to L1 pre-activation (gate by L1's output CReLU)
    {
        const dim3 block(256);
        const dim3 grid((c_l1Size + block.x - 1) / block.x, batchSize);
        BackpropToHiddenKernel<true><<<grid, block, 0, m_stream.Get()>>>(
            batch.trainingVectors.Get(),
            batch.l2PreErrors.Get(),
            m_l2Weights->m_weights.Get(),
            batch.l1Buffer.Get(),
            batch.l1PreErrors.Get(),
            batchSize,
            c_l1Size,
            c_l2Size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: L1 weight/bias gradients and backprop to FT accumulator.
    // accumulatorBuffer holds the RAW (pre-CReLU) FT accumulator, so apply CReLU
    // to the input here (<true>) — the L1 linear input is CReLU(accumulator).
    LaunchDenseGradients<true>(
        m_stream.Get(),
        batch.trainingVectors.Get(),
        batch.accumulatorBuffer.Get(),
        batch.l1PreErrors.Get(),
        batch.l1Gradients.Get(),
        batchSize, 2 * c_accumulatorSize, c_l1Size, c_numVariants);

    // The L1->FT backprop and the FT weight gradients are only needed to update the feature
    // transformer. When it is frozen, both are dead work and are skipped (the L1->FT backprop's
    // only output, creluErrors, is consumed solely by the FT gradient kernel below).
    if (m_featureTransformerWeights->m_updateWeights)
    {
        // Backprop from L1 to FT accumulator with CReLU gate (FT output is raw, gated by CReLU)
        {
            const dim3 block(256);
            const dim3 grid((2 * c_accumulatorSize + block.x - 1) / block.x, batchSize);
            BackpropToHiddenKernel<true><<<grid, block, 0, m_stream.Get()>>>(
                batch.trainingVectors.Get(),
                batch.l1PreErrors.Get(),
                m_l1Weights->m_weights.Get(),
                batch.accumulatorBuffer.Get(),
                batch.creluErrors.Get(),
                batchSize,
                2 * c_accumulatorSize,
                c_l1Size);
            CUDA_CHECK(cudaGetLastError());
        }

        // Step 5: FT weight gradients
        {
            batch.featureTransformerGradients.ClearAsync(m_stream.Get());
            const dim3 block(32, 16);
            const dim3 grid(
                (c_accumulatorSize + block.x - 1) / block.x,
                (batchSize + block.y - 1) / block.y);
            FeatureTransformerGradientsKernel<<<grid, block, 0, m_stream.Get()>>>(
                batch.creluErrors.Get(),
                batch.trainingVectors.Get(),
                batch.featureTransformerGradients.Get(),
                batchSize,
                c_numNetworkInputs,
                c_accumulatorSize);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Adam updates (UpdateAdam is a no-op for storages with m_updateWeights == false)
    m_l3Weights->UpdateAdam(batch.l3Gradients.Get(), learningRate, iteration, m_stream.Get());
    m_l2Weights->UpdateAdam(batch.l2Gradients.Get(), learningRate, iteration, m_stream.Get());
    m_l1Weights->UpdateAdam(batch.l1Gradients.Get(), learningRate, iteration, m_stream.Get());
    m_featureTransformerWeights->UpdateAdam(batch.featureTransformerGradients.Get(), learningRate, iteration, m_stream.Get());
}

} // namespace cuda
} // namespace nn
