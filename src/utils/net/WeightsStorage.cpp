#include "WeightsStorage.hpp"

#include <cmath>
#include <limits>
#include <algorithm>

namespace nn {

WeightsStorage::WeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : m_inputSize(inputSize)
    , m_outputSize(outputSize)
{
    const uint32_t numWeights = (inputSize + 1) * outputSize;

    m_variants.resize(numVariants);
    for (Variant& variant : m_variants)
        variant.m_weights.resize(numWeights, 0.0f);
}

WeightsStorage::Stats WeightsStorage::ComputeStats() const
{
    float minWeight = std::numeric_limits<float>::max();
    float maxWeight = -std::numeric_limits<float>::max();
    float minBias = std::numeric_limits<float>::max();
    float maxBias = -std::numeric_limits<float>::max();
    float weightAvg = 0.0f;
    float biasAvg = 0.0f;

    for (const auto& variant : m_variants)
    {
        for (uint32_t i = 0; i < m_outputSize; i++)
        {
            const float bias = variant.m_weights[m_inputSize * m_outputSize + i];
            minBias = std::min(minBias, bias);
            maxBias = std::max(maxBias, bias);
            biasAvg += bias;

            for (uint32_t j = 0; j < m_inputSize; j++)
            {
                const float weight = variant.m_weights[j * m_outputSize + i];
                minWeight = std::min(minWeight, weight);
                maxWeight = std::max(maxWeight, weight);
                weightAvg += weight;
            }
        }
    }

    weightAvg /= m_inputSize * m_outputSize * m_variants.size();
    biasAvg /= m_outputSize * m_variants.size();

    // calculate standard deviation
    float weightStdDev = 0.0f;
    float biasStdDev = 0.0f;
    for (const auto& variant : m_variants)
    {
        for (uint32_t i = 0; i < m_outputSize; i++)
        {
            const float bias = variant.m_weights[m_inputSize * m_outputSize + i];
            biasStdDev += (bias - biasAvg) * (bias - biasAvg);

            for (uint32_t j = 0; j < m_inputSize; j++)
            {
                const float weight = variant.m_weights[j * m_outputSize + i];
                weightStdDev += (weight - weightAvg) * (weight - weightAvg);
            }
        }
    }

    weightStdDev = sqrtf(weightStdDev / (m_inputSize * m_outputSize * m_variants.size()));
    biasStdDev = sqrtf(biasStdDev / (m_outputSize * m_variants.size()));

    return { minWeight, maxWeight, weightAvg, weightStdDev, minBias, maxBias, biasAvg, biasStdDev };
}

} // namespace nn
