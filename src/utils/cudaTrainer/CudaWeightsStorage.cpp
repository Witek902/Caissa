#include "CudaWeightsStorage.hpp"
#include "CudaKernels.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace nn {
namespace cuda {

CudaWeightsStorage::CudaWeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : m_inputSize(inputSize)
    , m_outputSize(outputSize)
    , m_numVariants(numVariants)
{
    m_totalWeights = (inputSize + 1) * outputSize * numVariants; // +1 for biases
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

    m_weights.CopyFromHost(hostWeights.data(), 0, hostWeights.size());
}

void CudaWeightsStorage::CopyFromHost(const nn::WeightsStorage& hostWeights)
{
    const uint32_t weightsPerVariant = (m_inputSize + 1) * m_outputSize;
    TRAINER_CHECK(hostWeights.m_variants.size() == m_numVariants);
    for (uint32_t variant = 0; variant < m_numVariants; ++variant)
    {
        const nn::Values& variantWeights = hostWeights.m_variants[variant].m_weights;
        TRAINER_CHECK(variantWeights.size() == weightsPerVariant);
        m_weights.CopyFromHost(variantWeights.data(), (size_t)variant * weightsPerVariant, weightsPerVariant);
    }

    m_weightsRange = hostWeights.m_weightsRange;
    m_biasRange = hostWeights.m_biasRange;
}

void CudaWeightsStorage::CopyToHost(nn::WeightsStorage& hostWeights) const
{
    // a frozen layer does not change on the device, the host keeps the copy made before freezing
    if (!m_updateWeights)
        return;

    const uint32_t weightsPerVariant = (m_inputSize + 1) * m_outputSize;

    for (uint32_t variant = 0; variant < m_numVariants; ++variant)
    {
        auto& hostVariant = hostWeights.m_variants[variant];
        hostVariant.m_weights.resize(weightsPerVariant);
        m_weights.CopyToHost(hostVariant.m_weights.data(), variant * weightsPerVariant, weightsPerVariant);
    }
}

void CudaWeightsStorage::CopyStateToHost(std::vector<float>& outWeights, std::vector<float>& outMoment1, std::vector<float>& outMoment2) const
{
    outWeights.resize(m_totalWeights);
    outMoment1.resize(m_totalWeights);
    outMoment2.resize(m_totalWeights);

    m_weights.CopyToHost(outWeights.data(), 0, m_totalWeights);
    m_moment1.CopyToHost(outMoment1.data(), 0, m_totalWeights);
    m_moment2.CopyToHost(outMoment2.data(), 0, m_totalWeights);
}

void CudaWeightsStorage::CopyStateFromHost(const std::vector<float>& weights, const std::vector<float>& moment1, const std::vector<float>& moment2)
{
    m_weights.CopyFromHost(weights.data(), 0, m_totalWeights);
    m_moment1.CopyFromHost(moment1.data(), 0, m_totalWeights);
    m_moment2.CopyFromHost(moment2.data(), 0, m_totalWeights);
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
