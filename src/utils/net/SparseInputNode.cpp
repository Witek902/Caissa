#include "SparseInputNode.hpp"
#include "WeightsStorage.hpp"

namespace nn {

void SparseInputNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);
    const Values& weights = m_weightsStorage->m_weights;

    ASSERT(ctx.outputs.size() == numOutputs);

    // apply biases
    std::copy(
        weights.data() + numOutputs * numInputs,
        weights.data() + numOutputs * (numInputs + 1),
        ctx.outputs.data());

    // accumulate active feature weights
    for (const ActiveFeature& feature : context.sparseInputs)
    {
        ASSERT(feature.index < numInputs);
        ASSERT(!std::isnan(feature.value));

        size_t i = 0;

#ifdef USE_AVX
        const __m256 vInputValue = _mm256_set1_ps(feature.value);
        const float* weightsPtr = weights.data() + feature.index * numOutputs;
        float* valuesPtr = ctx.outputs.data();
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(valuesPtr + i,
                            _mm256_fmadd_ps(vInputValue,
                                            _mm256_load_ps(weightsPtr + i),
                                            _mm256_load_ps(valuesPtr + i)));
        }
#endif // USE_AVX

        for (; i < numOutputs; i++)
        {
            ctx.outputs[i] += weights[feature.index * numOutputs + i] * feature.value;
        }
    }
}

void SparseInputNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const
{
    const Context& context = static_cast<const Context&>(ctx);

    // update gradient of active features
    for (const ActiveFeature& feature : context.sparseInputs)
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.m_values.data() + feature.index * numOutputs;
        const __m256 vInputValue = _mm256_set1_ps(feature.value);
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_fmadd_ps(vInputValue, _mm256_load_ps(error.data() + i), _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            gradients.m_values[feature.index * numOutputs + i] += feature.value * error[i];
        }
        gradients.m_dirty[feature.index] = true;
    }

    // add bias gradient
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.m_values.data() + numInputs * numOutputs;
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_add_ps(_mm256_load_ps(error.data() + i),
                    _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            gradients.m_values[numInputs * numOutputs + i] += error[i];
        }
        gradients.m_dirty[numInputs] = true;
    }
}

} // namespace nn
