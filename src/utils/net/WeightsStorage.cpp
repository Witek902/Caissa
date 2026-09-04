#include "WeightsStorage.hpp"
#include "../minitrace/minitrace.h"

#include <algorithm>
#include <random>

namespace nn {

WeightsStorage::WeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : m_inputSize(inputSize)
    , m_outputSize(outputSize)
{
    const uint32_t numWeights = (inputSize + 1) * outputSize;
    m_weightsMask.resize(numWeights, 1.0f);

    m_variants.resize(numVariants);
    for (Variant& variant : m_variants)
    {
        variant.m_weights.resize(numWeights, 0.0f);
        variant.m_gradientMoment1.resize(numWeights, 0.0f);
        variant.m_gradientMoment2.resize(numWeights, 0.0f);
    }
}

void WeightsStorage::Init(uint32_t numActiveNeurons, float bias)
{
    ASSERT(!m_variants.empty());

    std::fill(m_weightsMask.begin(), m_weightsMask.end(), 1.0f);

    // init first variant
    {
        Variant& variant = m_variants[0];

        memset(variant.m_gradientMoment1.data(), 0, sizeof(float) * variant.m_gradientMoment1.size());
        memset(variant.m_gradientMoment2.data(), 0, sizeof(float) * variant.m_gradientMoment2.size());

        std::random_device rd;
        std::mt19937 gen(rd());

        // Xavier weights initialization
        std::normal_distribution<float> weightDistr(0.0f, 2.0f / (float)numActiveNeurons);

        for (uint32_t j = 0; j < m_outputSize; ++j)
        {
            for (uint32_t i = 0; i < m_inputSize; ++i)
            {
                variant.m_weights[m_outputSize * i + j] = weightDistr(gen);
            }
        }

        for (size_t j = 0; j < m_outputSize; j++)
        {
            variant.m_weights[m_outputSize * m_inputSize + j] = bias;
        }
    }

    // copy first variant weights to remaining
    for (size_t i = 1; i < m_variants.size(); i++)
    {
        Variant& variant = m_variants[i];
        memset(variant.m_gradientMoment1.data(), 0, sizeof(float) * variant.m_gradientMoment1.size());
        memset(variant.m_gradientMoment2.data(), 0, sizeof(float) * variant.m_gradientMoment2.size());
        memcpy(variant.m_weights.data(), m_variants[0].m_weights.data(), sizeof(float) * variant.m_weights.size());
    }
}

void WeightsStorage::PrintStats() const
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

    std::cout
        << "weight range: [" << minWeight << " ... " << maxWeight
        << "], bias range: [" << minBias << " ... " << maxBias
        << "], weight avg: " << weightAvg << ", bias avg: " << biasAvg
        << ", weight std dev: " << weightStdDev << ", bias std dev: " << biasStdDev
        << '\n';
}

} // namespace nn
