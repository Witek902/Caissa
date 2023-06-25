#include "SparseBinaryInputNode.hpp"
#include "WeightsStorage.hpp"
#include "Gradient.hpp"

namespace nn {

static constexpr uint32_t c_NumRegisters = 8;

SparseBinaryInputNode::SparseBinaryInputNode(uint32_t inputSize, uint32_t outputSize, const nn::WeightsStoragePtr& weights)
    : ITrainableNode(nullptr, weights, inputSize, outputSize)
{
}

void SparseBinaryInputNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(!m_weightsStorage->m_variants.empty());
    const size_t variantIndex = std::min<size_t>(ctx.variant, m_weightsStorage->m_variants.size() - 1);
    const Values& weights = m_weightsStorage->m_variants[variantIndex].m_weights;

    ASSERT(ctx.outputs.size() == m_numOutputs);
    ASSERT(m_numOutputs % (c_NumRegisters * 8) == 0);

    const float* biasesPtr = weights.data() + m_numOutputs * m_numInputs;
    float* valuesPtr = ctx.outputs.data();

#ifdef USE_AVX

    // split processing into tiles of 8 AVX registers
    const uint32_t numTiles = m_numOutputs / (c_NumRegisters * 8u);

    __m256 regs[c_NumRegisters];

    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * (c_NumRegisters * 8u);

        // load biases
        for (uint32_t i = 0; i < c_NumRegisters; ++i)
            regs[i] = _mm256_load_ps(biasesPtr + chunkBase + i * 8u);

        // accumulate active feature weights
        for (const IndexType featureIdx : context.sparseInputs)
        {
            const float* weightsPtr = weights.data() + featureIdx * m_numOutputs;
            for (uint32_t i = 0; i < c_NumRegisters; ++i)
                regs[i] = _mm256_add_ps(regs[i], _mm256_load_ps(weightsPtr + chunkBase + i * 8u));
        }

        // store results
        for (uint32_t i = 0; i < c_NumRegisters; ++i)
            _mm256_store_ps(valuesPtr + chunkBase + i * 8u, regs[i]);
    }

#else

    // apply biases
    std::copy(
        weights.data() + m_numOutputs * m_numInputs,
        weights.data() + m_numOutputs * (m_numInputs + 1),
        ctx.outputs.data());

    // accumulate active feature weights
    for (const IndexType featureIdx : context.sparseInputs)
    {
        for (size_t i = 0; i < m_numOutputs; i++)
        {
            ctx.outputs[i] += weights[featureIdx * m_numOutputs + i];
        }
    }

#endif // USE_AVX

}

void SparseBinaryInputNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const
{
    const Context& context = static_cast<const Context&>(ctx);

    ASSERT(!m_weightsStorage->m_variants.empty());
    const size_t variantIndex = std::min<size_t>(ctx.variant, m_weightsStorage->m_variants.size() - 1);

    Gradients::Variant& gradientsVariant = gradients.m_variants[variantIndex];

    ASSERT(gradients.m_isSparse);

#ifdef USE_AVX

    // split processing into tiles of 8 AVX registers
    const uint32_t numTiles = m_numOutputs / (c_NumRegisters * 8u);

    __m256 regs[c_NumRegisters];

    for (uint32_t tile = 0; tile < numTiles; ++tile)
    {
        const uint32_t chunkBase = tile * (c_NumRegisters * 8u);

        // load error into registers
        for (uint32_t i = 0; i < c_NumRegisters; ++i)
            regs[i] = _mm256_load_ps(error.data() + chunkBase + i * 8u);

        // accumulate error to active feature's gradients
        for (const IndexType featureIdx : context.sparseInputs)
        {
            float* gradientPtr = gradientsVariant.m_values.data() + featureIdx * m_numOutputs;
            for (uint32_t i = 0; i < c_NumRegisters; ++i)
            {
                _mm256_store_ps(gradientPtr + chunkBase + i * 8u,
                    _mm256_add_ps(_mm256_load_ps(gradientPtr + chunkBase + i * 8u), regs[i]));
            }
        }
    }

    // mark gradients as dirty
    for (const IndexType featureIdx : context.sparseInputs)
    {
        gradientsVariant.m_dirty[featureIdx] = true;
    }

#else

    // update gradient of active features
    for (const IndexType j : context.sparseInputs)
    {
        for (size_t i = 0; i < m_numOutputs; i++)
        {
            // not multiplying by input value, because it's equal to 1.0
            gradientsVariant.m_values[j * m_numOutputs + i] += error[i];
        }
        gradientsVariant.m_dirty[j] = true;
    }
#endif // USE_AVX


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
