#include "SparseInputNode.hpp"
#include "WeightsStorage.hpp"
#include "Gradient.hpp"

namespace nn {

void SparseInputNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(!m_weightsStorage->m_variants.empty());
    const size_t variantIndex = std::min<size_t>(ctx.variant, m_weightsStorage->m_variants.size() - 1);
    const Values& weights = m_weightsStorage->m_variants[variantIndex].m_weights;

    ASSERT(ctx.outputs.size() == m_numOutputs);

    // apply biases
    std::copy(
        weights.data() + m_numOutputs * m_numInputs,
        weights.data() + m_numOutputs * (m_numInputs + 1),
        ctx.outputs.data());

    // accumulate active feature weights
    for (const ActiveFeature& feature : context.sparseInputs)
    {
        ASSERT(feature.index < m_numInputs);
        ASSERT(!std::isnan(feature.value));

        size_t i = 0;

#ifdef USE_AVX
        const __m256 vInputValue = _mm256_set1_ps(feature.value);
        const float* weightsPtr = weights.data() + feature.index * m_numOutputs;
        float* valuesPtr = ctx.outputs.data();
        for (; i + 8 <= m_numOutputs; i += 8)
        {
            _mm256_store_ps(valuesPtr + i,
                            _mm256_fmadd_ps(vInputValue,
                                            _mm256_load_ps(weightsPtr + i),
                                            _mm256_load_ps(valuesPtr + i)));
        }
#endif // USE_AVX

        for (; i < m_numOutputs; i++)
        {
            ctx.outputs[i] += weights[feature.index * m_numOutputs + i] * feature.value;
        }
    }
}

void SparseInputNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const
{
    const Context& context = static_cast<const Context&>(ctx);

    ASSERT(!m_weightsStorage->m_variants.empty());
    const size_t variantIndex = std::min<size_t>(ctx.variant, m_weightsStorage->m_variants.size() - 1);
    Gradients::Variant& gradientsVariant = gradients.m_variants[variantIndex];

    ASSERT(gradients.m_isSparse);

    // update gradient of active features
    for (const ActiveFeature& feature : context.sparseInputs)
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradientsVariant.m_values.data() + feature.index * m_numOutputs;
        const __m256 vInputValue = _mm256_set1_ps(feature.value);
        for (; i + 8 <= m_numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_fmadd_ps(vInputValue, _mm256_load_ps(error.data() + i), _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < m_numOutputs; i++)
        {
            gradientsVariant.m_values[feature.index * m_numOutputs + i] += feature.value * error[i];
        }
        gradientsVariant.m_dirty[feature.index] = true;
    }

    // add bias gradient
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradientsVariant.m_values.data() + m_numInputs * m_numOutputs;
        for (; i + 8 <= m_numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_add_ps(_mm256_load_ps(error.data() + i),
                    _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < m_numOutputs; i++)
        {
            gradientsVariant.m_values[m_numInputs * m_numOutputs + i] += error[i];
        }
        gradientsVariant.m_dirty[m_numInputs] = true;
    }
}

} // namespace nn
