#include "SparseBinaryInputNode.hpp"
#include "WeightsStorage.hpp"
#include "Gradient.hpp"

namespace nn {

SparseBinaryInputNode::SparseBinaryInputNode(uint32_t inputSize, uint32_t outputSize, const nn::WeightsStoragePtr& weights)
    : ITrainableNode(nullptr, weights, inputSize, outputSize)
{
}

void SparseBinaryInputNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);
    const Values& weights = m_weightsStorage->m_weights;

    ASSERT(ctx.outputs.size() == m_numOutputs);

    // apply biases
    std::copy(
        weights.data() + m_numOutputs * m_numInputs,
        weights.data() + m_numOutputs * (m_numInputs + 1),
        ctx.outputs.data());

    // accumulate active feature weights
    for (const IndexType featureIdx : context.sparseInputs)
    {
        size_t i = 0;

#ifdef USE_AVX
        const float* weightsPtr = weights.data() + featureIdx * m_numOutputs;
        float* valuesPtr = ctx.outputs.data();
        for (; i + 8 <= m_numOutputs; i += 8)
        {
            _mm256_store_ps(valuesPtr + i,
                            _mm256_add_ps(_mm256_load_ps(valuesPtr + i),
                                          _mm256_load_ps(weightsPtr + i)));
        }
#endif // USE_AVX

        for (; i < m_numOutputs; i++)
        {
            ctx.outputs[i] += weights[featureIdx * m_numOutputs + i];
        }
    }
}

void SparseBinaryInputNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const
{
    const Context& context = static_cast<const Context&>(ctx);

    ASSERT(gradients.m_isSparse);

    // update gradient of active features
    for (const IndexType j : context.sparseInputs)
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.m_values.data() + j * m_numOutputs;
        for (; i + 8 <= m_numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_add_ps(_mm256_load_ps(error.data() + i), _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < m_numOutputs; i++)
        {
            // not multiplying by input value, because it's equal to 1.0
            gradients.m_values[j * m_numOutputs + i] += error[i];
        }
        gradients.m_dirty[j] = true;
    }

    // add bias gradient
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.m_values.data() + m_numInputs * m_numOutputs;
        for (; i + 8 <= m_numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_add_ps(_mm256_load_ps(error.data() + i),
                    _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < m_numOutputs; i++)
        {
            gradients.m_values[m_numInputs * m_numOutputs + i] += error[i];
        }
        gradients.m_dirty[m_numInputs] = true;
    }
}

} // namespace nn
