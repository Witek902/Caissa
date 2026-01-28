#include "CudaWeightsStorage.hpp"
#include <random>
#include <cmath>

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
    m_weightsMask.Allocate(m_totalWeights);
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
    m_weightsMask.CopyFromHost(hostMask.data(), hostMask.size());
}

void CudaWeightsStorage::CopyFromHost(const nn::WeightsStorage& hostWeights)
{
    std::vector<float> hostWeightsData;
    std::vector<float> hostMaskData;

    hostWeightsData.reserve(m_totalWeights);
    hostMaskData.reserve(m_totalWeights);

    for (const auto& variant : hostWeights.m_variants)
    {
        hostWeightsData.insert(hostWeightsData.end(), variant.m_weights.begin(), variant.m_weights.end());
    }

    hostMaskData.assign(hostWeights.m_weightsMask.begin(), hostWeights.m_weightsMask.end());

    m_weights.CopyFromHost(hostWeightsData.data(), hostWeightsData.size());
    m_weightsMask.CopyFromHost(hostMaskData.data(), hostMaskData.size());
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

// CUDA kernel for Adam weight updates
__global__ void AdamUpdateKernel(
    float* weights,
    float* moment1,
    float* moment2,
    const double* gradients,
    uint32_t numWeights,
    float learningRate,
    float weightDecay,
    size_t iteration
)
{
    // Adam parameters
    const float c_beta1 = 0.9f;
    const float c_beta2 = 0.999f;
    const float c_epsilon = 1.0e-10f;

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWeights) return;

    const float grad = static_cast<float>(gradients[idx]);

    // Update biased first moment estimate
    const float m1 = moment1[idx] = c_beta1 * moment1[idx] + (1.0f - c_beta1) * grad;

    // Update biased second raw moment estimate
    const float m2 = moment2[idx] = c_beta2 * moment2[idx] + (1.0f - c_beta2) * grad * grad;

    // Compute bias-corrected first moment estimate
    const float m_hat = m1 / (1.0f - powf(c_beta1, iteration + 1));

    // Compute bias-corrected second raw moment estimate
    const float v_hat = m2 / (1.0f - powf(c_beta2, iteration + 1));

    // Compute weight delta
    const float delta = learningRate * (m_hat / (sqrtf(v_hat) + c_epsilon) + weights[idx] * weightDecay);

    // Update weights
    weights[idx] -= delta;
}

void CudaWeightsStorage::UpdateAdam(const double* gradients, float learningRate, float weightDecay, size_t iteration, cudaStream_t stream)
{
    if (!m_updateWeights) return;

    const dim3 blockSize(256);
    const dim3 gridSize((m_totalWeights + blockSize.x - 1) / blockSize.x);

    AdamUpdateKernel<<<gridSize, blockSize, 0, stream>>>(
        m_weights.Get(),
        m_moment1.Get(),
        m_moment2.Get(),
        gradients,
        m_totalWeights,
        learningRate,
        weightDecay,
        iteration
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace nn
