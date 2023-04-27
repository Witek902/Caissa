#include "SparseBinaryInputNode.hpp"
#include "WeightsStorage.hpp"

namespace nn {

SparseBinaryInputNode::SparseBinaryInputNode(uint32_t inputSize, uint32_t outputSize, const nn::WeightsStoragePtr& weights)
    : ITrainableNode(nullptr, weights, inputSize, outputSize)
{
}

void SparseBinaryInputNode::Run(INodeContext& ctx) const
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
    for (const IndexType featureIdx : context.sparseInputs)
    {
        size_t i = 0;

#ifdef USE_AVX
        const float* weightsPtr = weights.data() + featureIdx * numOutputs;
        float* valuesPtr = ctx.outputs.data();
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(valuesPtr + i,
                            _mm256_add_ps(_mm256_load_ps(valuesPtr + i),
                                          _mm256_load_ps(weightsPtr + i)));
        }
#endif // USE_AVX

        for (; i < numOutputs; i++)
        {
            ctx.outputs[i] += weights[featureIdx * numOutputs + i];
        }
    }
}

void SparseBinaryInputNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const
{
    const Context& context = static_cast<const Context&>(ctx);

    // update gradient of active features
    for (const IndexType j : context.sparseInputs)
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.m_values.data() + j * numOutputs;
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                _mm256_add_ps(_mm256_load_ps(error.data() + i), _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            // not multiplying by input value, because it's equal to 1.0
            gradients.m_values[j * numOutputs + i] += error[i];
        }
        gradients.m_dirty[j] = true;
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
