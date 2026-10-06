#include "CudaWeightsStorage.hpp"
#include "CudaKernels.hpp"

#include <cmath>
#include <iostream>
#include <random>

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

void CudaWeightsStorage::Init(uint32_t seed, float stdev, float bias)
{
    std::vector<float> hostWeights(m_totalWeights, 0.0f);

    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);

    const uint32_t weightsPerVariant = (m_inputSize + 1) * m_outputSize;

    // Initialize weights (excluding biases)
    for (uint32_t i = 0; i < m_inputSize * m_outputSize; ++i)
    {
        const float weightValue = dist(gen);
        for (uint32_t variant = 0; variant < m_numVariants; ++variant)
        {
            hostWeights[variant * weightsPerVariant + i] = weightValue;
        }
    }

    // Initialize biases
    for (uint32_t variant = 0; variant < m_numVariants; ++variant)
    {
        const uint32_t variantOffset = variant * weightsPerVariant;
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

void CudaWeightsStorage::CopyStateToHost(std::vector<float>& outWeights, std::vector<float>& outMoment1, std::vector<float>& outMoment2) const
{
    outWeights.resize(m_totalWeights);
    outMoment1.resize(m_totalWeights);
    outMoment2.resize(m_totalWeights);

    m_weights.CopyToHost(outWeights.data(), m_totalWeights);
    m_moment1.CopyToHost(outMoment1.data(), m_totalWeights);
    m_moment2.CopyToHost(outMoment2.data(), m_totalWeights);
}

void CudaWeightsStorage::CopyStateFromHost(const std::vector<float>& weights, const std::vector<float>& moment1, const std::vector<float>& moment2)
{
    m_weights.CopyFromHost(weights.data(), m_totalWeights);
    m_moment1.CopyFromHost(moment1.data(), m_totalWeights);
    m_moment2.CopyFromHost(moment2.data(), m_totalWeights);
}

void CudaWeightsStorage::UpdateAdam(const float* gradients, float learningRate, cudaStream_t stream)
{
    if (!m_updateWeights)
        return;

    const size_t step = m_adamStep++;

    // Bias-correction factors depend only on the step, so compute them once here (in double
    // precision) instead of calling powf for every weight inside the kernel.
    AdamUpdateParams params;
    params.weights = m_weights.Get();
    params.moment1 = m_moment1.Get();
    params.moment2 = m_moment2.Get();
    params.gradients = gradients;
    params.inputSize = m_inputSize;
    params.outputSize = m_outputSize;
    params.numWeights = m_totalWeights;
    params.learningRate = learningRate;
    params.weightDecay = m_weightDecay;
    params.maxWeightRange = m_weightsRange;
    params.maxBiasRange = m_biasRange;
    params.factorizerFirstWeight = m_factorizerFirstWeight;
    params.maxFactorizerRange = m_factorizerRange;
    params.beta1 = m_beta1;
    params.beta2 = m_beta2;
    params.biasCorrection1 = (float)(1.0 / (1.0 - std::pow((double)m_beta1, (double)(step + 1))));
    params.biasCorrection2 = (float)(1.0 / (1.0 - std::pow((double)m_beta2, (double)(step + 1))));

    LaunchAdamUpdate(params, stream);
}

} // namespace cuda
} // namespace nn
