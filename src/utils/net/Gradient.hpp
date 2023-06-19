#pragma once

#include "Common.hpp"

namespace nn {

// Sparse gradients
struct Gradients
{
    uint32_t            m_numInputs = 0;
    uint32_t            m_numOutputs = 0;
    bool                m_isSparse = false;
    Values              m_values;
    std::vector<bool>   m_dirty;

    void Init(uint32_t numInputs, uint32_t numOutputsm, bool isSparse);
    void Clear();
    void Accumulate(Gradients& rhs);
};

} // namespace nn
