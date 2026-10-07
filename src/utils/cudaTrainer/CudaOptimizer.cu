#include "CudaKernels.hpp"

namespace nn {
namespace cuda {

constexpr float c_epsilon = 1.0e-8f;

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// Per-variant layout is [inputSize*outputSize weights][outputSize biases]; the last outputSize
// entries of each variant block are biases.
__device__ __host__ __forceinline__ bool IsBiasIndex(uint32_t idx, uint32_t inputSize, uint32_t outputSize)
{
    const uint32_t perVariant = (inputSize + 1) * outputSize;
    return (idx % perVariant) >= inputSize * outputSize;
}

// CUDA kernel for AdamW weight updates (Adam with decoupled weight decay)
__global__ void AdamUpdateKernel(
    float* weights,
    float* moment1,
    float* moment2,
    const float* gradients,
    uint32_t inputSize,
    uint32_t outputSize,
    uint32_t numWeights,
    float learningRate,
    float weightDecay,
    float maxWeightRange,
    float maxBiasRange,
    uint32_t factorizerFirstWeight,
    float maxFactorizerRange,
    float biasCorrection1, // 1 / (1 - beta1^t), precomputed on the host
    float biasCorrection2  // 1 / (1 - beta2^t), precomputed on the host
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWeights) return;

    const bool isBias = IsBiasIndex(idx, inputSize, outputSize);
    const float maxWeightValue = isBias ? maxBiasRange : (idx >= factorizerFirstWeight ? maxFactorizerRange : maxWeightRange);

    const float grad = static_cast<float>(gradients[idx]);

    // Update biased first moment estimate
    const float m1 = moment1[idx] = c_beta1 * moment1[idx] + (1.0f - c_beta1) * grad;

    // Update biased second raw moment estimate
    const float m2 = moment2[idx] = c_beta2 * moment2[idx] + (1.0f - c_beta2) * grad * grad;

    // Bias-corrected moment estimates (the correction factors are step-only, precomputed host-side)
    const float m_hat = m1 * biasCorrection1;
    const float v_hat = m2 * biasCorrection2;

    const float oldWeight = weights[idx];

    // Compute the update step
    float delta = learningRate * m_hat / (sqrtf(v_hat) + c_epsilon);

    // Apply decoupled weight decay only to weights
    if (!isBias)
    {
        delta += learningRate * weightDecay * oldWeight;
    }

    weights[idx] = clamp(oldWeight - delta, -maxWeightValue, maxWeightValue);
}

void LaunchAdamUpdate(const AdamUpdateParams& p, cudaStream_t stream)
{
    const dim3 blockSize(256);
    const dim3 gridSize((p.numWeights + blockSize.x - 1) / blockSize.x);

    AdamUpdateKernel<<<gridSize, blockSize, 0, stream>>>(
        p.weights,
        p.moment1,
        p.moment2,
        p.gradients,
        p.inputSize,
        p.outputSize,
        p.numWeights,
        p.learningRate,
        p.weightDecay,
        p.maxWeightRange,
        p.maxBiasRange,
        p.factorizerFirstWeight,
        p.maxFactorizerRange,
        p.biasCorrection1,
        p.biasCorrection2
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace nn
