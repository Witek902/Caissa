#include "NeuralNetworkLayer.hpp"
#include "minitrace/minitrace.h"

#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>

static constexpr float c_activationEpsilon = 1.0e-10f;

namespace nn {

void LayerRunContext::Init(const Layer& layer)
{
    linearValue.resize(layer.numOutputs);
    output.resize(layer.numOutputs);
    inputGradient.resize(layer.numInputs);
}

Layer::Layer(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : numInputs(inputSize)
    , numOutputs(outputSize)
{
    ASSERT(numOutputs <= MaxLayerOutputs);
    ASSERT(numVariants > 0);
    activationFunc = ActivationFunction::ClippedReLu;

    variants.resize(numVariants);
    for (Variant& variant : variants)
    {
        const uint32_t numWeights = (inputSize + 1) * outputSize;
        variant.weights.resize(numWeights);
        variant.gradientMean.resize(numWeights, 0.0f);
        variant.gradientMoment.resize(numWeights, 0.0f);
    }
}

void Layer::InitWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (Variant& variant : variants)
    {
        memset(variant.gradientMean.data(), 0, sizeof(float) * variant.gradientMean.size());
        memset(variant.gradientMoment.data(), 0, sizeof(float) * variant.gradientMoment.size());

        size_t offs = 0;

        if (activationFunc == ActivationFunction::Sigmoid)
        {
            const float r = sqrtf(6.0f / numInputs);
            std::uniform_real_distribution<float> weightDistr(-r, r);
            for (; offs < numOutputs * numInputs; offs++)
            {
                variant.weights[offs] = weightDistr(rd);
            }

            for (size_t j = 0; j < numOutputs; j++)
            {
                variant.weights[offs + j] = 0.0f;
            }
        }
        else
        {
            // He initialization
            const float r = sqrtf(6.0f / (float)(numInputs + numOutputs));
            std::uniform_real_distribution<float> weightDistr(-r, r);
            for (; offs < numOutputs * numInputs; offs++)
            {
                variant.weights[offs] = weightDistr(rd);
            }

            for (size_t j = 0; j < numOutputs; j++)
            {
                variant.weights[offs + j] = 0.01f;
            }
        }
    }
}

INLINE static float ApplyActivationFunction(float x, ActivationFunction func)
{
    switch (func)
    {
    case ActivationFunction::ClippedReLu:   return ClippedReLu(x);
    case ActivationFunction::Sigmoid:       return Sigmoid(x);
    case ActivationFunction::ATan:          return InvTan(x);
    }
    return x;
}

INLINE static float GetActivationFunctionDerivative(float x, ActivationFunction func)
{
    switch (func)
    {
    case ActivationFunction::ClippedReLu:   return ClippedReLuDerivative(x);
    case ActivationFunction::Sigmoid:       return SigmoidDerivative(x);
    case ActivationFunction::ATan:          return InvTanDerivative(x);
    }
    return 1.0f;
}

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

void Layer::Run(const float* values, LayerRunContext& ctx) const
{
    ctx.inputs.resize(numInputs);
    ctx.inputMode = InputMode::Full;

    // TODO VARIANT
    const Variant& variant = variants.front();
    const Values& weights = variant.weights;

    // apply biases
    memcpy(ctx.linearValue.data(), weights.data() + numOutputs * numInputs, sizeof(float) * numOutputs);

    if (numOutputs == 1)
    {
        size_t i = 0;
#ifdef USE_AVX
        const float* weightsPtr = weights.data();
        __m256 sum = _mm256_setzero_ps();
        for (; i + 8 <= numInputs; i += 8)
        {
            const __m256 inputs = _mm256_loadu_ps(values + i);
            _mm256_store_ps(ctx.inputs.data() + i, inputs);

            sum = _mm256_fmadd_ps(_mm256_load_ps(weightsPtr + i),
                                  inputs,
                                  sum);
        }
        ctx.linearValue[0] += m256_hadd(sum);
#endif // USE_AVX

        for (; i < numInputs; i++)
        {
            const float inputValue = values[i];
            ctx.inputs[i] = inputValue;
            ctx.linearValue[0] += weights[i] * inputValue;
        }
    }
    else
    {
        // accumulate weights
        for (uint32_t j = 0; j < numInputs; j++)
        {
            const float inputValue = values[j];
            ctx.inputs[j] = inputValue;

            if (std::abs(inputValue) > c_activationEpsilon)
            {
                uint32_t i = 0;

#ifdef USE_AVX
                const float* weightsPtr = weights.data() + j * numOutputs;
                float* valuesPtr = ctx.linearValue.data();
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
                    ctx.linearValue[i] += weights[j * numOutputs + i] * ctx.inputs[j];
                }
            }
        }
    }

    ctx.ComputeOutput(activationFunc);
}

void Layer::Run(uint32_t numFeatures, const uint16_t* featureIndices, LayerRunContext& ctx) const
{
	const Variant& variant = variants.front();
	const Values& weights = variant.weights;

    ctx.sparseBinaryInputs.resize(numFeatures);
    ctx.inputMode = InputMode::SparseBinary;

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        const uint16_t idx = featureIndices[i];
        ASSERT(idx < numInputs);
        ctx.sparseBinaryInputs[i] = idx;
    }

	// apply biases
	memcpy(ctx.linearValue.data(), weights.data() + numOutputs * numInputs, sizeof(float) * numOutputs);

    // accumulate active feature weights
    for (uint32_t j = 0; j < numFeatures; ++j)
    {
        const uint32_t idx = featureIndices[j];

        size_t i = 0;

#ifdef USE_AVX
        const float* weightsPtr = weights.data() + idx * numOutputs;
        float* valuesPtr = ctx.linearValue.data();
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(valuesPtr + i,
                            _mm256_add_ps(_mm256_load_ps(valuesPtr + i),
                                          _mm256_load_ps(weightsPtr + i)));
        }
#endif // USE_AVX

        for (; i < numOutputs; i++)
        {
            ctx.linearValue[i] += weights[idx * numOutputs + i];
        }
    }

    ctx.ComputeOutput(activationFunc);
}

void Layer::Run(uint32_t numFeatures, const ActiveFeature* features, LayerRunContext& ctx) const
{
	const Variant& variant = variants.front();
	const Values& weights = variant.weights;

    ctx.sparseInputs.resize(numFeatures);
    ctx.inputMode = InputMode::Sparse;

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        const ActiveFeature& feature = features[i];
        ASSERT(feature.index < numInputs);
        ASSERT(!std::isnan(feature.value));
        ctx.sparseInputs[i] = feature;
    }

	// apply biases
	memcpy(ctx.linearValue.data(), weights.data() + numOutputs * numInputs, sizeof(float) * numOutputs);

    // accumulate active feature weights
    for (uint32_t j = 0; j < numFeatures; ++j)
    {
        const uint32_t idx = features[j].index;

        size_t i = 0;

#ifdef USE_AVX
        const __m256 vInputValue = _mm256_set1_ps(features[j].value);
        const float* weightsPtr = weights.data() + idx * numOutputs;
        float* valuesPtr = ctx.linearValue.data();
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
            ctx.linearValue[i] += weights[idx * numOutputs + i] * features[j].value;
        }
    }

    ctx.ComputeOutput(activationFunc);
}

void LayerRunContext::ComputeOutput(ActivationFunction activationFunc)
{
    const size_t numOutputs = output.size();

#ifndef CONFIGURATION_FINAL
    for (size_t i = 0; i < numOutputs; i++)
    {
        const float x = linearValue[i];
        ASSERT(!std::isnan(x));
        ASSERT(fabsf(x) < 10000.0f);
    }
#endif // CONFIGURATION_FINAL

    size_t i = 0;
#ifdef USE_AVX
    if (activationFunc == ActivationFunction::ClippedReLu)
    {
        float* outputsPtr = output.data();
        const float* valuesPtr = linearValue.data();
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(outputsPtr + i, ClippedReLu(_mm256_load_ps(valuesPtr + i)));
        }
    }
#endif // USE_AVX
    for (; i < numOutputs; i++)
    {
        output[i] = ApplyActivationFunction(linearValue[i], activationFunc);
    }
}

void Layer::Backpropagate(const Values& error, LayerRunContext& ctx, Gradients& gradients) const
{
	const Variant& variant = variants.front();
	const Values& weights = variant.weights;

    ASSERT(ctx.output.size() == error.size());
    ASSERT(ctx.output.size() <= MaxLayerOutputs);
    alignas(CACHELINE_SIZE) float activationErrors[MaxLayerOutputs];

    // precompute error gradients
    {
        size_t i = 0;
#ifdef USE_AVX
        if (activationFunc == ActivationFunction::ClippedReLu)
        {
            const float* errorsPtr = error.data();
            const float* valuesPtr = ctx.linearValue.data();
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(activationErrors + i,
                                ClippedReLuDerivative(_mm256_load_ps(valuesPtr + i), _mm256_load_ps(errorsPtr + i)));
            }
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            activationErrors[i] = error[i] * GetActivationFunctionDerivative(ctx.linearValue[i], activationFunc);
        }
    }

    if (ctx.inputMode == InputMode::SparseBinary)
    {
        // 'SparseBinary' mode is used only for first layer, so don't compute nextError (there's no more layers before it to backpropagate)

        // update gradient of active features
        for (const uint16_t& j : ctx.sparseBinaryInputs)
        {
            size_t i = 0;
#ifdef USE_AVX
            float* gradientPtr = gradients.m_values.data() + j * numOutputs;
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(gradientPtr + i,
                                _mm256_add_ps(_mm256_load_ps(activationErrors + i), _mm256_load_ps(gradientPtr + i)));
            }
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                // not multiplying by input value, because it's equal to 1.0
                gradients.m_values[j * numOutputs + i] += activationErrors[i];
            }
            gradients.m_dirty[j] = true;
        }
    }
    else if (ctx.inputMode == InputMode::Sparse)
    {
        // 'Sparse' mode is used only for first layer, so don't compute nextError (there's no more layers before it to backpropagate)
        // TODO use Sparse mode for next layers?

        // update gradient of active features
        for (const ActiveFeature& feature : ctx.sparseInputs)
        {
            size_t i = 0;
#ifdef USE_AVX
            float* gradientPtr = gradients.m_values.data() + feature.index * numOutputs;
            const __m256 vInputValue = _mm256_set1_ps(feature.value);
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(gradientPtr + i,
                                _mm256_fmadd_ps(vInputValue, _mm256_load_ps(activationErrors + i), _mm256_load_ps(gradientPtr + i)));
            }
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                gradients.m_values[feature.index * numOutputs + i] += feature.value * activationErrors[i];
            }
            gradients.m_dirty[feature.index] = true;
        }
    }
    else if (ctx.inputMode == InputMode::Full)
    {
        // for later layers, use exact input values and compute input error (for back propagation)
        {
            for (size_t j = 0; j < numInputs; j++)
            {
                ctx.inputGradient[j] = 0.0f;
            }

            for (size_t i = 0; i < numOutputs; i++)
            {
                const float activationError = activationErrors[i];
                if (std::abs(activationError) > c_activationEpsilon)
                {
                    for (size_t j = 0; j < numInputs; j++)
                    {
                        ctx.inputGradient[j] += weights[j * numOutputs + i] * activationError;
                    }
                }
            }
        }

        for (size_t j = 0; j < numInputs; j++)
        {
            // compute weights gradient
            const float inputValue = ctx.inputs[j];
            if (std::abs(inputValue) > c_activationEpsilon)
            {
                size_t i = 0;
#ifdef USE_AVX
                float* gradientPtr = gradients.m_values.data() + j * numOutputs;
                for (; i + 8 <= numOutputs; i += 8)
                {
                    _mm256_store_ps(gradientPtr + i,
                                    _mm256_fmadd_ps(_mm256_set1_ps(inputValue),
                                                    _mm256_load_ps(activationErrors + i),
                                                    _mm256_load_ps(gradientPtr + i)));
                }
#endif // USE_AVX
                for (; i < numOutputs; i++)
                {
                    gradients.m_values[j * numOutputs + i] += inputValue * activationErrors[i];
                }
                gradients.m_dirty[j] = true;
            }
        }
    }
    else
    {
        DEBUG_BREAK();
    }

    // compute biases gradient
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.m_values.data() + numInputs * numOutputs;
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                            _mm256_add_ps(_mm256_load_ps(activationErrors + i),
                                          _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            gradients.m_values[numInputs * numOutputs + i] += activationErrors[i];
        }
        gradients.m_dirty[numInputs] = true;
    }
}

void Layer::UpdateWeights(float learningRate, const Gradients& gradients, const float gradientScale, const float weightsRange, const float biasRange, const float weightDecay)
{
    MTR_SCOPE("Layer::UpdateWeights", "UpdateWeights");

    ASSERT(gradients.m_values.size() == (numInputs + 1) * numOutputs);

    const float cRho = 0.95f;
    const float cEpsilon = 1.0e-8f;

#ifdef USE_AVX
    const __m256 cOneMinusRhoVec = _mm256_set1_ps(1.0f - cRho);
    const __m256 cRhoVec = _mm256_set1_ps(cRho);
    const __m256 cEpsilonVec = _mm256_set1_ps(cEpsilon);
    const __m256 gradientScaleVec = _mm256_set1_ps(gradientScale);
#endif

    for (Variant& variant : variants)
    {
        for (size_t j = 0; j <= numInputs; j++)
        {
            const float maxWeightValue = j < numInputs ? weightsRange : biasRange;

            size_t i = 0;

#ifdef USE_AVX
            const __m256 minValueV = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_set1_ps(maxWeightValue));
            const __m256 maxValueV = _mm256_set1_ps(maxWeightValue);
            for (; i + 8 <= numOutputs; i += 8)
            {
                float* mPtr = variant.gradientMean.data() + j * numOutputs + i;
                float* vPtr = variant.gradientMoment.data() + j * numOutputs + i;
                float* wPtr = variant.weights.data() + j * numOutputs + i;
                const float* gPtr = gradients.m_values.data() + j * numOutputs + i;

                __m256 g = _mm256_mul_ps(gradientScaleVec, _mm256_load_ps(gPtr));
                __m256 v = _mm256_load_ps(vPtr);
                __m256 m = _mm256_load_ps(mPtr);
                __m256 w = _mm256_load_ps(wPtr);

                // weight decay
                g = _mm256_fmadd_ps(w, _mm256_set1_ps(weightDecay), g);

                // ADADELTA algorithm
                m = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(g, g), _mm256_mul_ps(cRhoVec, m));
                const __m256 delta = _mm256_mul_ps(g, _mm256_sqrt_ps(_mm256_div_ps(_mm256_add_ps(v, cEpsilonVec), _mm256_add_ps(m, cEpsilonVec))));
                v = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(delta, delta), _mm256_mul_ps(cRhoVec, v));
                w = _mm256_fnmadd_ps(delta, _mm256_set1_ps(learningRate), w);

                // clamping
                w = _mm256_min_ps(w, maxValueV);
                w = _mm256_max_ps(w, minValueV);

                _mm256_store_ps(vPtr, v);
                _mm256_store_ps(mPtr, m);
                _mm256_store_ps(wPtr, w);
            }
#endif // USE_AVX

            for (; i < numOutputs; ++i)
            {
				float& m = variant.gradientMean[j * numOutputs + i];
				float& v = variant.gradientMoment[j * numOutputs + i];
				float& w = variant.weights[j * numOutputs + i];
                float g = gradientScale * gradients.m_values[j * numOutputs + i];

                ASSERT(!std::isnan(g));
                ASSERT(v >= 0.0f);
                ASSERT(m >= 0.0f);

                // weight decay
                g += w * weightDecay;

                // ADADELTA algorithm
                m = cRho * m + (1.0f - cRho) * g * g;
                ASSERT(!std::isnan(m));

                const float delta = g * sqrtf((v + cEpsilon) / (m + cEpsilon));
                v = cRho * v + (1.0f - cRho) * delta * delta;
                ASSERT(!std::isnan(v));

                w -= learningRate * delta;
                ASSERT(!std::isnan(w));

                // clamping
                w = std::clamp(w, -maxWeightValue, maxWeightValue);
            }
        }
    }
}

void Gradients::Init(uint32_t numInputs, uint32_t numOutputs)
{
	m_numInputs = numInputs;
	m_numOutputs = numOutputs;
    m_values.resize((numInputs + 1) * numOutputs, 0.0f);
    m_dirty.resize(numInputs + 1, false);
}

void Gradients::Clear()
{
	for (size_t i = 0; i <= m_numInputs; ++i)
	{
		if (m_dirty[i])
		{
			std::fill(m_values.begin() + i * m_numOutputs,
                      m_values.begin() + (i + 1) * m_numOutputs,
					  0.0f);
		}
	}

	for (size_t i = 0; i < m_values.size(); ++i)
	{
		ASSERT(m_values[i] == 0.0f);
	}

	std::fill(m_dirty.begin(), m_dirty.end(), false);
}

void Gradients::Accumulate(Gradients& rhs)
{
    ASSERT(rhs.m_numInputs == m_numInputs);
    ASSERT(rhs.m_numOutputs == m_numOutputs);

	for (size_t i = 0; i <= m_numInputs; ++i)
	{
		if (rhs.m_dirty[i])
		{
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
}

} // namespace nn
