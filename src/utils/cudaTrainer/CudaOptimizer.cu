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
__global__ void AdamUpdateKernel(const AdamUpdateParams p)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.numWeights) return;

    const bool isBias = IsBiasIndex(idx, p.inputSize, p.outputSize);
    const float maxWeightValue = isBias ? p.maxBiasRange : (idx >= p.factorizerFirstWeight ? p.maxFactorizerRange : p.maxWeightRange);

    const float grad = p.gradients[idx];

    // Update biased first moment estimate
    const float m1 = p.moment1[idx] = p.beta1 * p.moment1[idx] + (1.0f - p.beta1) * grad;

    // Update biased second raw moment estimate
    const float m2 = p.moment2[idx] = p.beta2 * p.moment2[idx] + (1.0f - p.beta2) * grad * grad;

    // Bias-corrected moment estimates (the correction factors are step-only, precomputed host-side)
    const float m_hat = m1 * p.biasCorrection1;
    const float v_hat = m2 * p.biasCorrection2;

    const float oldWeight = p.weights[idx];

    // Compute the update step
    float delta = p.learningRate * m_hat / (sqrtf(v_hat) + c_epsilon);

    // Apply decoupled weight decay only to weights
    if (!isBias)
    {
        delta += p.learningRate * p.weightDecay * oldWeight;
    }

    p.weights[idx] = clamp(oldWeight - delta, -maxWeightValue, maxWeightValue);
}

void LaunchAdamUpdate(const AdamUpdateParams& p, cudaStream_t stream)
{
    const dim3 blockSize(256);
    const dim3 gridSize((p.numWeights + blockSize.x - 1) / blockSize.x);

    AdamUpdateKernel<<<gridSize, blockSize, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace nn
