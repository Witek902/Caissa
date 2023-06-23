#pragma once

#include "Common.hpp"

namespace nn {

// Sparse gradients for WeightsStorage
struct Gradients
{
    uint32_t            m_numInputs = 0;
    uint32_t            m_numOutputs = 0;
    bool                m_isSparse = false;

    struct Variant
    {
        Values              m_values;
        std::vector<bool>   m_dirty;
    };
    std::vector<Variant> m_variants;

    void Init(uint32_t numInputs, uint32_t numOutputs, uint32_t numVariants, bool isSparse);
    void Clear();
    void Accumulate(Gradients& rhs);
};

} // namespace nn
