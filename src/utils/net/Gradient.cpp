#include "Gradient.hpp"

#include <random>

namespace nn {

void Gradients::Init(uint32_t numInputs, uint32_t numOutputs, uint32_t numVariants, bool isSparse)
{
    m_numInputs = numInputs;
    m_numOutputs = numOutputs;
    m_isSparse = isSparse;

    m_variants.resize(numVariants);
    for (Variant& variant : m_variants)
    {
        variant.m_values.resize((numInputs + 1) * numOutputs, 0.0f);
        variant.m_dirty.resize(numInputs + 1, false);
    }
}

void Gradients::Clear()
{
    if (m_isSparse)
    {
        for (Variant& variant : m_variants)
        {
            // clear only dirty gradients
            for (size_t i = 0; i <= m_numInputs; ++i)
            {
                if (variant.m_dirty[i])
                {
                    std::fill(
                        variant.m_values.begin() + i * m_numOutputs,
                        variant.m_values.begin() + (i + 1) * m_numOutputs,
                        0.0f);
                }
            }

#ifndef CONFIGURATION_FINAL
            for (size_t i = 0; i < variant.m_values.size(); ++i)
            {
                ASSERT(variant.m_values[i] == 0.0f);
            }
#endif // CONFIGURATION_FINAL

            std::fill(variant.m_dirty.begin(), variant.m_dirty.end(), false);
        }
    }
    else
    {
        for (Variant& variant : m_variants)
        {
            std::fill(variant.m_values.begin(), variant.m_values.end(), 0.0f);
        }
    }
}

void Gradients::Accumulate(Gradients& rhs, uint32_t inputIndex)
{
    ASSERT(inputIndex <= m_numInputs);
    ASSERT(rhs.m_numInputs == m_numInputs);
    ASSERT(rhs.m_numOutputs == m_numOutputs);
    ASSERT(rhs.m_variants.size() == m_variants.size());
    ASSERT(rhs.m_isSparse == m_isSparse);

    for (size_t variantIndex = 0; variantIndex < m_variants.size(); ++variantIndex)
    {
        Variant& variant = m_variants[variantIndex];
        Variant& rhsVariant = rhs.m_variants[variantIndex];

        ASSERT(rhsVariant.m_values.size() == variant.m_values.size());

        if (m_isSparse && !rhsVariant.m_dirty[inputIndex])
            continue;

        // NOTE: not updating dirty flags here, because it's not thread-safe
        // It will be done later in Accumulate_UpdateDirtyFlags
        //variant.m_dirty[inputIndex] = true;
        //rhsVariant.m_dirty[inputIndex] = false;

        size_t j = inputIndex * m_numOutputs;
        const size_t j_max = (inputIndex + 1) * m_numOutputs;

#ifdef USE_AVX
        float* values = variant.m_values.data();
        float* rhsValues = rhsVariant.m_values.data();
        for (; j + 8 <= j_max; j += 8)
        {
            _mm256_store_ps(values + j,
                _mm256_add_ps(_mm256_load_ps(values + j), _mm256_load_ps(rhsValues + j)));
            _mm256_store_ps(rhsValues + j, _mm256_setzero_ps());
        }
#endif // USE_AVX

        for (; j < j_max; ++j)
        {
            variant.m_values[j] += rhsVariant.m_values[j];
            rhsVariant.m_values[j] = 0.0f;
        }
    }
}

void Gradients::Accumulate_UpdateDirtyFlags(Gradients& rhs, uint32_t inputIndex)
{
    ASSERT(inputIndex <= m_numInputs);
    ASSERT(rhs.m_numInputs == m_numInputs);
    ASSERT(rhs.m_numOutputs == m_numOutputs);
    ASSERT(rhs.m_variants.size() == m_variants.size());
    ASSERT(rhs.m_isSparse == m_isSparse);

    for (size_t variantIndex = 0; variantIndex < m_variants.size(); ++variantIndex)
    {
        Variant& variant = m_variants[variantIndex];
        Variant& rhsVariant = rhs.m_variants[variantIndex];

        ASSERT(rhsVariant.m_values.size() == variant.m_values.size());

        if (m_isSparse)
        {
            if (rhsVariant.m_dirty[inputIndex])
            {
                variant.m_dirty[inputIndex] = true;
                rhsVariant.m_dirty[inputIndex] = false;
            }
        }
    }
}

} // namespace nn
