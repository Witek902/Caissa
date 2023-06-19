#include "Gradient.hpp"

#include <random>

namespace nn {

void Gradients::Init(uint32_t numInputs, uint32_t numOutputs, bool isSparse)
{
    m_numInputs = numInputs;
    m_numOutputs = numOutputs;
    m_isSparse = isSparse;
    m_values.resize((numInputs + 1) * numOutputs, 0.0f);
    m_dirty.resize(numInputs + 1, false);
}

void Gradients::Clear()
{
    if (m_isSparse)
    {
        // clear only dirty gradients
        for (size_t i = 0; i <= m_numInputs; ++i)
        {
            if (m_dirty[i])
            {
                std::fill(m_values.begin() + i * m_numOutputs,
                    m_values.begin() + (i + 1) * m_numOutputs,
                    0.0f);
            }
        }

#ifndef CONFIGURATION_FINAL
        for (size_t i = 0; i < m_values.size(); ++i)
        {
            ASSERT(m_values[i] == 0.0f);
        }
#endif // CONFIGURATION_FINAL

        std::fill(m_dirty.begin(), m_dirty.end(), false);
    }
    else
    {
        std::fill(m_values.begin(), m_values.end(), 0.0f);
    }
}

void Gradients::Accumulate(Gradients& rhs)
{
    ASSERT(rhs.m_numInputs == m_numInputs);
    ASSERT(rhs.m_numOutputs == m_numOutputs);
    ASSERT(rhs.m_values.size() == m_values.size());
    ASSERT(rhs.m_isSparse == m_isSparse);

    if (m_isSparse)
    {
        for (size_t i = 0; i <= m_numInputs; ++i)
        {
            if (!rhs.m_dirty[i])
                continue;

            m_dirty[i] = true;
            rhs.m_dirty[i] = false;

            size_t j = i * m_numOutputs;
            const size_t j_max = (i + 1) * m_numOutputs;

#ifdef USE_AVX
            float* values = m_values.data();
            float* rhsValues = rhs.m_values.data();
            for (; j + 8 <= j_max; j += 8)
            {
                _mm256_store_ps(values + j,
                    _mm256_add_ps(_mm256_load_ps(values + j), _mm256_load_ps(rhsValues + j)));
                _mm256_store_ps(rhsValues + j, _mm256_setzero_ps());
            }
#endif // USE_AVX

            for (; j < j_max; ++j)
            {
                m_values[j] += rhs.m_values[j];
                rhs.m_values[j] = 0.0f;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < m_values.size(); ++i)
        {
            m_values[i] += rhs.m_values[i];
            rhs.m_values[i] = 0.0f;
        }
    }
}

} // namespace nn
