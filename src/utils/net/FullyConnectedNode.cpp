#include "FullyConnectedNode.hpp"
#include "WeightsStorage.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef USE_SSE
#include <immintrin.h>
#endif // USE_SSE

static constexpr float c_activationEpsilon = 1.0e-10f;

namespace nn {

#ifdef USE_AVX

INLINE static float m256_hadd(__m256 x)
{
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    const __m128 loQuad = _mm256_castps256_ps128(x);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    const __m128 loDual = sumQuad;
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    const __m128 lo = sumDual;
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

#endif // USE_AVX

FullyConnectedNode::FullyConnectedNode(const NodePtr& previousNode, uint32_t inputSize, uint32_t outputSize, const nn::WeightsStoragePtr& weights)
    : ITrainableNode(previousNode, weights, inputSize, outputSize)
{
}

void FullyConnectedNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);
    const Values& weights = m_weightsStorage->m_weights;

    ASSERT(ctx.outputs.size() == numOutputs);
    ASSERT(ctx.inputs.size() == numInputs);

    // apply biases
    std::copy(
        weights.data() + numOutputs * numInputs,
        weights.data() + numOutputs * (numInputs + 1),
        ctx.outputs.data());

    // TODO dedicated node with single output
    if (numOutputs == 1)
    {
        size_t i = 0;
#ifdef USE_AVX
        const float* weightsPtr = weights.data();
        __m256 sum = _mm256_setzero_ps();
        for (; i + 8 <= numInputs; i += 8)
        {
            const __m256 inputs = _mm256_loadu_ps(context.inputs.data() + i);
            sum = _mm256_fmadd_ps(_mm256_load_ps(weightsPtr + i),
                                  inputs,
                                  sum);
        }
        ctx.outputs[0] += m256_hadd(sum);
#endif // USE_AVX

        for (; i < numInputs; i++)
        {
            ctx.outputs[0] += weights[i] * context.inputs[i];
        }
    }
    else
    {
        // accumulate weights
        for (uint32_t j = 0; j < numInputs; j++)
        {
            const float inputValue = context.inputs[j];
            if (std::abs(inputValue) > c_activationEpsilon)
            {
                uint32_t i = 0;
#ifdef USE_AVX
                const float* weightsPtr = weights.data() + j * numOutputs;
                float* valuesPtr = ctx.outputs.data();
                const __m256 vInputValue = _mm256_set1_ps(inputValue);
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
                    ctx.outputs[i] += weights[j * numOutputs + i] * inputValue;
                }
            }
        }
    }
}

void FullyConnectedNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const
{
    const Context& context = static_cast<const Context&>(ctx);
    const Values& weights = m_weightsStorage->m_weights;

    ASSERT(ctx.outputs.size() == GetNumOutputs());
    ASSERT(ctx.inputs.size() == GetNumInputs());
    ASSERT(ctx.inputError.size() == GetNumInputs());

    std::fill(ctx.inputError.begin(), ctx.inputError.end(), 0.0f);

    if (numOutputs > 1)
    {
        for (size_t i = 0; i < numOutputs; i++)
        {
            const float activationError = error[i];
            if (std::abs(activationError) > c_activationEpsilon)
            {
                for (size_t j = 0; j < numInputs; j++)
                {
                    ctx.inputError[j] += weights[j * numOutputs + i] * activationError;
                }
            }
        }

        for (size_t j = 0; j < numInputs; j++)
        {
            // compute weights gradient
            const float inputValue = context.inputs[j];
            if (std::abs(inputValue) > c_activationEpsilon)
            {
                size_t i = 0;
#ifdef USE_AVX
                float* gradientPtr = gradients.m_values.data() + j * numOutputs;
                for (; i + 8 <= numOutputs; i += 8)
                {
                    _mm256_store_ps(gradientPtr + i,
                        _mm256_fmadd_ps(_mm256_set1_ps(inputValue),
                            _mm256_load_ps(error.data() + i),
                            _mm256_load_ps(gradientPtr + i)));
                }
#endif // USE_AVX
                for (; i < numOutputs; i++)
                {
                    gradients.m_values[j * numOutputs + i] += inputValue * error[i];
                }
                gradients.m_dirty[j] = true;
            }
        }
    }
    else // numOutputs == 1
    {
        const float activationError = error[0];
        if (std::abs(activationError) > c_activationEpsilon)
        {
            size_t j = 0;
#ifdef USE_AVX
            float* gradientPtr = gradients.m_values.data();
            float* inputGradientPtr = ctx.inputError.data();
            const float* inputPtr = context.inputs.data();
            const float* weightsPtr = weights.data();
            const __m256 activationErrorV = _mm256_set1_ps(activationError);
            for (; j + 8 <= numInputs; j += 8)
            {
                // compute input gradient
                _mm256_store_ps(inputGradientPtr + j,
                    _mm256_mul_ps(
                        _mm256_load_ps(weightsPtr + j),
                        activationErrorV));

                // compute weights gradient
                _mm256_store_ps(gradientPtr + j,
                    _mm256_fmadd_ps(activationErrorV,
                        _mm256_load_ps(inputPtr + j),
                        _mm256_load_ps(gradientPtr + j)));
            }
#endif // USE_AVX
            for (; j < numInputs; j++)
            {
                // compute input gradient
                ctx.inputError[j] += weights[j] * activationError;
                // compute weights gradient
                gradients.m_values[j] += context.inputs[j] * activationError;

                // gradients.m_dirty[j] = true; // dense layers gradients are always accumulated
            }
        }
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
