#include "CudaWeightsStorage.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <cstdlib>

namespace nn {
namespace cuda {

CudaWeightsStorage::CudaWeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : m_inputSize(inputSize)
    , m_outputSize(outputSize)
    , m_numVariants(numVariants)
{
    m_totalWeights = (inputSize + 1) * outputSize * numVariants; // +1 for biases
    AllocateBuffers();
}

CudaWeightsStorage::~CudaWeightsStorage()
{
    // Buffers are automatically freed by CudaBuffer destructors
}

void CudaWeightsStorage::AllocateBuffers()
{
    m_weights.Allocate(m_totalWeights);
    m_moment1.Allocate(m_totalWeights);
    m_moment2.Allocate(m_totalWeights);
}

void CudaWeightsStorage::Init(uint32_t numActiveInputs, float bias)
{
    std::vector<float> hostWeights(m_totalWeights, 0.0f);
    std::vector<float> hostMask(m_totalWeights, 1.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const uint32_t weightsPerVariant = (m_inputSize + 1) * m_outputSize;

    for (uint32_t variant = 0; variant < m_numVariants; ++variant)
    {
        const uint32_t variantOffset = variant * weightsPerVariant;

        // Initialize weights (excluding biases)
        for (uint32_t i = 0; i < m_inputSize * m_outputSize; ++i)
        {
            hostWeights[variantOffset + i] = dist(gen) * 0.1f;
        }

        // Initialize biases
        for (uint32_t i = 0; i < m_outputSize; ++i)
        {
            hostWeights[variantOffset + m_inputSize * m_outputSize + i] = bias;
        }
    }

    m_weights.CopyFromHost(hostWeights.data(), hostWeights.size());
}

void CudaWeightsStorage::CopyFromHost(const nn::WeightsStorage& hostWeights)
{
    std::vector<float> hostWeightsData;
    hostWeightsData.reserve(m_totalWeights);

    for (const auto& variant : hostWeights.m_variants)
    {
        hostWeightsData.insert(hostWeightsData.end(), variant.m_weights.begin(), variant.m_weights.end());
    }

    // Note: hostWeights.m_weightsMask is not applied in CUDA path; all weights are updated.
    if (hostWeightsData.size() != m_totalWeights)
    {
        std::cerr << "CudaWeightsStorage::CopyFromHost size mismatch: host " << hostWeightsData.size() << " vs " << m_totalWeights << std::endl;
        std::exit(1);
    }

    m_weights.CopyFromHost(hostWeightsData.data(), hostWeightsData.size());

    m_weightsRange = hostWeights.m_weightsRange;
    m_biasRange = hostWeights.m_biasRange;
}

void CudaWeightsStorage::CopyToHost(nn::WeightsStorage& hostWeights) const
{
    std::vector<float> hostWeightsData(m_totalWeights);

    m_weights.CopyToHost(hostWeightsData.data(), hostWeightsData.size());

    const uint32_t weightsPerVariant = (m_inputSize + 1) * m_outputSize;

    for (uint32_t variant = 0; variant < m_numVariants; ++variant)
    {
        const uint32_t offset = variant * weightsPerVariant;
        auto& hostVariant = hostWeights.m_variants[variant];

        hostVariant.m_weights.assign(
            hostWeightsData.begin() + offset,
            hostWeightsData.begin() + offset + weightsPerVariant
        );
    }
}

// Adam parameters
constexpr float c_beta1 = 0.9f;
constexpr float c_beta2 = 0.999f;
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
    float biasCorrection1, // 1 / (1 - beta1^t), precomputed on the host
    float biasCorrection2  // 1 / (1 - beta2^t), precomputed on the host
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWeights) return;

    const bool isBias = IsBiasIndex(idx, inputSize, outputSize);
    const float maxWeightValue = isBias ? maxBiasRange : maxWeightRange;

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

void CudaWeightsStorage::UpdateAdam(const float* gradients, float learningRate, cudaStream_t stream)
{
    if (!m_updateWeights)
        return;

    const size_t step = m_adamStep++;

    // Bias-correction factors depend only on the step, so compute them once here (in double
    // precision) instead of calling powf for every weight inside the kernel.
    const float biasCorrection1 = (float)(1.0 / (1.0 - std::pow(c_beta1, (double)(step + 1))));
    const float biasCorrection2 = (float)(1.0 / (1.0 - std::pow(c_beta2, (double)(step + 1))));

    const dim3 blockSize(256);
    const dim3 gridSize((m_totalWeights + blockSize.x - 1) / blockSize.x);

    AdamUpdateKernel<<<gridSize, blockSize, 0, stream>>>(
        m_weights.Get(),
        m_moment1.Get(),
        m_moment2.Get(),
        gradients,
        m_inputSize,
        m_outputSize,
        m_totalWeights,
        learningRate,
        m_weightDecay,
        m_weightsRange,
        m_biasRange,
        biasCorrection1,
        biasCorrection2
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace nn
