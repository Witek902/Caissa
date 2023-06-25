#include "WeightsStorage.hpp"
#include "Gradient.hpp"
#include "../HaltonSequence.hpp"
#include "../minitrace/minitrace.h"

#include <random>

namespace nn {

WeightsStorage::WeightsStorage(uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : m_inputSize(inputSize)
    , m_outputSize(outputSize)
{
    const uint32_t numWeights = (inputSize + 1) * outputSize;
    m_weightsMask.resize(numWeights, 1.0f);

    m_variants.resize(numVariants);
    for (Variant& variant : m_variants)
    {
        variant.m_weights.resize(numWeights, 0.0f);
        variant.m_gradientMoment1.resize(numWeights, 0.0f);
        variant.m_gradientMoment2.resize(numWeights, 0.0f);
    }
}

void WeightsStorage::Init(uint32_t numActiveNeurons, float bias)
{
    ASSERT(!m_variants.empty());

    std::fill(m_weightsMask.begin(), m_weightsMask.end(), 1.0f);

    // init first variant
    {
        Variant& variant = m_variants[0];

        memset(variant.m_gradientMoment1.data(), 0, sizeof(float) * variant.m_gradientMoment1.size());
        memset(variant.m_gradientMoment2.data(), 0, sizeof(float) * variant.m_gradientMoment2.size());

        const float scale = sqrtf(2.0f / (float)numActiveNeurons);

        //std::random_device rd;
        //std::mt19937 gen(rd());

        // Xavier weights initialization
        //std::normal_distribution<float> weightDistr(0.0f, sqrtf(2.0f / (float)numActiveNeurons));

        HaltonSequence haltonSequence;
        haltonSequence.Initialize(m_inputSize);

        for (uint32_t j = 0; j < m_outputSize; ++j)
        {
            for (uint32_t i = 0; i < m_inputSize; ++i)
            {
                const float u = static_cast<float>(haltonSequence.GetDouble(i));
                variant.m_weights[m_outputSize * i + j] = (u - 0.5f) * scale;
            }
            haltonSequence.NextSample();
        }

        for (size_t j = 0; j < m_outputSize; j++)
        {
            variant.m_weights[m_outputSize * m_inputSize + j] = bias;
        }
    }

    // copy first variant weights to remaining
    for (size_t i = 1; i < m_variants.size(); i++)
    {
        Variant& variant = m_variants[i];
        memset(variant.m_gradientMoment1.data(), 0, sizeof(float) * variant.m_gradientMoment1.size());
        memset(variant.m_gradientMoment2.data(), 0, sizeof(float) * variant.m_gradientMoment2.size());
        memcpy(variant.m_weights.data(), m_variants[0].m_weights.data(), sizeof(float) * variant.m_weights.size());
    }
}

void WeightsStorage::Update_Adadelta(const Gradients& gradients, const WeightsUpdateOptions& options)
{
    MTR_SCOPE("WeightsStorage::Update_Adadelta", "Update_Adadelta");

    ASSERT(gradients.m_numInputs == m_inputSize);
    ASSERT(gradients.m_numOutputs == m_outputSize);
    ASSERT(gradients.m_variants.size() == m_variants.size());

    for (size_t variantIndex = 0; variantIndex < m_variants.size(); ++variantIndex)
    {
        Variant& variant = m_variants[variantIndex];
        const Gradients::Variant& gradientsVariant = gradients.m_variants[variantIndex];

        ASSERT(gradientsVariant.m_values.size() == (m_inputSize + 1) * m_outputSize);

        const float cRho = 0.95f;
        const float cEpsilon = 1.0e-8f;

#ifdef USE_AVX
        const __m256 cOneMinusRhoVec = _mm256_set1_ps(1.0f - cRho);
        const __m256 cRhoVec = _mm256_set1_ps(cRho);
        const __m256 cEpsilonVec = _mm256_set1_ps(cEpsilon);
        const __m256 gradientScaleVec = _mm256_set1_ps(options.gradientScale);
#endif

        // TODO parallel for
        for (size_t j = 0; j <= m_inputSize; j++)
        {
            const float maxWeightValue = j < m_inputSize ? m_weightsRange : m_biasRange;

            size_t i = 0;

#ifdef USE_AVX
            const __m256 minValueV = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_set1_ps(maxWeightValue));
            const __m256 maxValueV = _mm256_set1_ps(maxWeightValue);
            for (; i + 8 <= m_outputSize; i += 8)
            {
                float* mPtr = variant.m_gradientMoment1.data() + j * m_outputSize + i;
                float* vPtr = variant.m_gradientMoment2.data() + j * m_outputSize + i;
                float* wPtr = variant.m_weights.data() + j * m_outputSize + i;
                float* wMaskPtr = m_weightsMask.data() + j * m_outputSize + i;
                const float* gPtr = gradientsVariant.m_values.data() + j * m_outputSize + i;

                __m256 g = _mm256_mul_ps(gradientScaleVec, _mm256_load_ps(gPtr));
                __m256 v = _mm256_load_ps(vPtr);
                __m256 m = _mm256_load_ps(mPtr);
                __m256 w = _mm256_load_ps(wPtr);
                const __m256 wMask = _mm256_load_ps(wMaskPtr);

                // weight decay
                g = _mm256_fmadd_ps(w, _mm256_set1_ps(options.weightDecay), g);

                // ADADELTA algorithm
                m = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(g, g), _mm256_mul_ps(cRhoVec, m));
                __m256 delta = _mm256_mul_ps(g, _mm256_sqrt_ps(_mm256_div_ps(_mm256_add_ps(v, cEpsilonVec), _mm256_add_ps(m, cEpsilonVec))));
                v = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(delta, delta), _mm256_mul_ps(cRhoVec, v));
                delta = _mm256_mul_ps(wMask, delta);
                w = _mm256_fnmadd_ps(delta, _mm256_set1_ps(options.learningRate), w);

                // clamping
                w = _mm256_min_ps(w, maxValueV);
                w = _mm256_max_ps(w, minValueV);

                _mm256_store_ps(vPtr, v);
                _mm256_store_ps(mPtr, m);
                _mm256_store_ps(wPtr, w);
            }
#endif // USE_AVX

            for (; i < m_outputSize; ++i)
            {
                float& m = variant.m_gradientMoment1[j * m_outputSize + i];
                float& v = variant.m_gradientMoment2[j * m_outputSize + i];
                float& w = variant.m_weights[j * m_outputSize + i];
                const float& wMask = m_weightsMask[j * m_outputSize + i];
                float g = options.gradientScale * gradientsVariant.m_values[j * m_outputSize + i];

                ASSERT(!std::isnan(g));
                ASSERT(v >= 0.0f);
                ASSERT(m >= 0.0f);

                // weight decay
                g += w * options.weightDecay;

                // ADADELTA algorithm
                m = cRho * m + (1.0f - cRho) * g * g;
                ASSERT(!std::isnan(m));

                const float delta = g * sqrtf((v + cEpsilon) / (m + cEpsilon));
                v = cRho * v + (1.0f - cRho) * delta * delta;
                ASSERT(!std::isnan(v));

                w -= wMask * options.learningRate * delta;
                ASSERT(!std::isnan(w));

                // clamping
                w = std::clamp(w, -maxWeightValue, maxWeightValue);
            }
        }
    }
}

void WeightsStorage::Update_Adam(const Gradients& gradients, const WeightsUpdateOptions& options)
{
    MTR_SCOPE("WeightsStorage::Weights_Adam", "Update_Adam");

    ASSERT(gradients.m_numInputs == m_inputSize);
    ASSERT(gradients.m_numOutputs == m_outputSize);
    ASSERT(gradients.m_variants.size() == m_variants.size());

    for (size_t variantIndex = 0; variantIndex < m_variants.size(); ++variantIndex)
    {
        Variant& variant = m_variants[variantIndex];
        const Gradients::Variant& gradientsVariant = gradients.m_variants[variantIndex];

        ASSERT(gradientsVariant.m_values.size() == (m_inputSize + 1) * m_outputSize);

        const float cBeta1 = 0.9f;
        const float cBeta2 = 0.999f;
        const float cEpsilon = 1.0e-8f;

        const float cIter = (float)(options.iteration + 1);
        const float cBeta1Mult = 1.0f / (1.0f - powf(cBeta1, cIter));
        const float cBeta2Mult = 1.0f / (1.0f - powf(cBeta2, cIter));

#ifdef USE_AVX
        const __m256 cOneMinusBeta1Vec = _mm256_set1_ps(1.0f - cBeta1);
        const __m256 cBeta1Vec = _mm256_set1_ps(cBeta1);
        const __m256 cOneMinusBeta2Vec = _mm256_set1_ps(1.0f - cBeta2);
        const __m256 cBeta2Vec = _mm256_set1_ps(cBeta2);
        const __m256 cEpsilonVec = _mm256_set1_ps(cEpsilon);
        const __m256 gradientScaleVec = _mm256_set1_ps(options.gradientScale);
#endif

        // TODO parallel for
        for (size_t j = 0; j <= m_inputSize; j++)
        {
            const float maxWeightValue = j < m_inputSize ? m_weightsRange : m_biasRange;

            size_t i = 0;

#ifdef USE_AVX
            const __m256 minValueV = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_set1_ps(maxWeightValue));
            const __m256 maxValueV = _mm256_set1_ps(maxWeightValue);
            for (; i + 8 <= m_outputSize; i += 8)
            {
                float* mPtr = variant.m_gradientMoment1.data() + j * m_outputSize + i;
                float* vPtr = variant.m_gradientMoment2.data() + j * m_outputSize + i;
                float* wPtr = variant.m_weights.data() + j * m_outputSize + i;
                float* wMaskPtr = m_weightsMask.data() + j * m_outputSize + i;
                const float* gPtr = gradientsVariant.m_values.data() + j * m_outputSize + i;

                __m256 g = _mm256_mul_ps(gradientScaleVec, _mm256_load_ps(gPtr));
                __m256 v = _mm256_load_ps(vPtr);
                __m256 m = _mm256_load_ps(mPtr);
                __m256 w = _mm256_load_ps(wPtr);
                const __m256 wMask = _mm256_load_ps(wMaskPtr);

                // update biased first moment estimate
                m = _mm256_fmadd_ps(cOneMinusBeta1Vec, g, _mm256_mul_ps(cBeta1Vec, m));

                // update biased second moment estimate
                v = _mm256_fmadd_ps(cOneMinusBeta2Vec, _mm256_mul_ps(g, g), _mm256_mul_ps(cBeta2Vec, v));

                // compute bias-corrected moment estimates
                const __m256 m_hat = _mm256_mul_ps(m, _mm256_set1_ps(cBeta1Mult));
                const __m256 v_hat = _mm256_mul_ps(v, _mm256_set1_ps(cBeta2Mult));

                // compute final weight change
                __m256 delta = _mm256_div_ps(m_hat, _mm256_add_ps(cEpsilonVec, _mm256_sqrt_ps(v_hat)));
                delta = _mm256_fmadd_ps(w, _mm256_set1_ps(options.weightDecay), delta); // weight decay
                delta = _mm256_mul_ps(wMask, delta);
                w = _mm256_fnmadd_ps(delta, _mm256_set1_ps(options.learningRate), w);

                // clamping
                w = _mm256_min_ps(w, maxValueV);
                w = _mm256_max_ps(w, minValueV);

                _mm256_store_ps(vPtr, v);
                _mm256_store_ps(mPtr, m);
                _mm256_store_ps(wPtr, w);
            }
#endif // USE_AVX

            for (; i < m_outputSize; ++i)
            {
                float& m = variant.m_gradientMoment1[j * m_outputSize + i];
                float& v = variant.m_gradientMoment2[j * m_outputSize + i];
                float& w = variant.m_weights[j * m_outputSize + i];
                const float wMask = m_weightsMask[j * m_outputSize + i];
                float g = options.gradientScale * gradientsVariant.m_values[j * m_outputSize + i];

                ASSERT(!std::isnan(g));
                ASSERT(v >= 0.0f);

                // update biased first moment estimate
                m = cBeta1 * m + (1.0f - cBeta1) * g;
                ASSERT(!std::isnan(m));

                // update biased second moment estimate
                v = cBeta2 * v + (1.0f - cBeta2) * g * g;
                ASSERT(!std::isnan(v));

                // compute bias-corrected moment estimates
                const float m_hat = m * cBeta1Mult;
                const float v_hat = v * cBeta2Mult;

                // compute final weight change
                const float delta = options.learningRate * (m_hat / (cEpsilon + sqrtf(v_hat)) + w * options.weightDecay);
                ASSERT(!std::isnan(delta));

                w -= wMask * delta;
                ASSERT(!std::isnan(w));

                // clamping
                w = std::clamp(w, -maxWeightValue, maxWeightValue);
            }
        }
    }
}

} // namespace nn
