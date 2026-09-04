#pragma once

#include "../Common.hpp"

#include "../../backend/Memory.hpp"

#include <vector>
#include <memory>

namespace nn {

using Values = std::vector<float, AlignmentAllocator<float, 32>>;

struct Gradients;

struct WeightsStorage
{
public:
    WeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants);

    void Init(uint32_t numActiveInputs, float bias = 0.0f);

    void PrintStats() const;

    uint32_t m_inputSize = 0;
    uint32_t m_outputSize = 0;
    bool m_isSparse = false;

    bool m_updateWeights = true;
    Values m_weightsMask;

    float m_weightsRange = 10.0f;
    float m_biasRange = 10.0f;

    struct Variant
    {
        Values m_weights;

        // used for learning
        Values m_gradientMoment1;
        Values m_gradientMoment2;
    };

    std::vector<Variant> m_variants;
};

using WeightsStoragePtr = std::shared_ptr<WeightsStorage>;

} // namespace nn
