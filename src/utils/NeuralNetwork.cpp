#include "NeuralNetwork.hpp"
#include "ThreadPool.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>

using namespace threadpool;

namespace nn {

void TrainingVector::CombineSparseInputs()
{
    std::sort(sparseInputs.begin(), sparseInputs.end(),
              [](const ActiveFeature& a, const ActiveFeature& b) { return a.index < b.index; });

    // TODO this is O(N^2)
    for (size_t i = 0; ; )
    {
        if (i + 1 >= sparseInputs.size()) break;

        if (sparseInputs[i].index == sparseInputs[i + 1].index)
        {
            // combine with next and erase
            sparseInputs[i].value += sparseInputs[i + 1].value;
            sparseInputs.erase(sparseInputs.begin() + i + 1);
        }
        else if (std::abs(sparseInputs[i].value) < 1.0e-7f)
        {
            // remove zero inputs
            sparseInputs.erase(sparseInputs.begin() + i);
        }
        else
        {
            i++;
        }
    }
}

void TrainingVector::Validate() const
{
    if (inputMode == InputMode::Full)
    {
    }
    else if (inputMode == InputMode::SparseBinary)
    {
        std::vector<uint16_t> sortedInputs = sparseBinaryInputs;
        std::sort(sortedInputs.begin(), sortedInputs.end());
        ASSERT(std::adjacent_find(sortedInputs.begin(), sortedInputs.end()) == sortedInputs.end());
    }
    else if (inputMode == InputMode::Sparse)
    {

    }
    else
    {
        DEBUG_BREAK();
    }
}

void LayerRunContext::Init(const Layer& layer)
{
    linearValue.resize(layer.numOutputs);
    output.resize(layer.numOutputs);
    inputGradient.resize(layer.numInputs);
}

void NeuralNetworkRunContext::Init(const NeuralNetwork& network)
{
    layers.resize(network.layers.size());

    for (size_t i = 0; i < network.layers.size(); ++i)
    {
        layers[i].Init(network.layers[i]);
    }

    tempValues.resize(network.GetOutputSize());
}

Layer::Layer(uint32_t inputSize, uint32_t outputSize)
    : numInputs(inputSize)
    , numOutputs(outputSize)
{
    activationFunction = ActivationFunction::ClippedReLu;

    weights.resize((inputSize + 1) * outputSize);
    gradientMean.resize(weights.size(), 0.0f);
    gradientMoment.resize(weights.size(), 0.0f);
}

void Layer::InitWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 0; i < weights.size(); i++)
    {
        gradientMean[i] = 0.0f;
        gradientMoment[i] = 0.0f;
    }

    size_t offs = 0;

    if (activationFunction == ActivationFunction::Sigmoid)
    {
        const float r = sqrtf(6.0f / numInputs);
        std::uniform_real_distribution<float> weightDistr(-r, r);
        for (; offs < numOutputs * numInputs; offs++)
        {
            weights[offs] = weightDistr(rd);
        }

        for (size_t j = 0; j < numOutputs; j++)
        {
            weights[offs + j] = 0.0f;
        }
    }
    else
    {
        // He initialization
        const float r = sqrtf(6.0f / (float)(numInputs + numOutputs));
        std::uniform_real_distribution<float> weightDistr(-r, r);
        for (; offs < numOutputs * numInputs; offs++)
        {
            weights[offs] = weightDistr(rd);
        }

        for (size_t j = 0; j < numOutputs; j++)
        {
            weights[offs + j] = 0.01f;
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

void Layer::Run(const Values& in, LayerRunContext& ctx) const
{
    ASSERT(in.size() == numInputs);

    ctx.inputs = in;
    ctx.inputMode = InputMode::Full;

    // apply biases
    memcpy(ctx.linearValue.data(), weights.data() + numOutputs * numInputs, sizeof(float) * numOutputs);

    if (numOutputs == 1)
    {
        size_t i = 0;
#ifdef USE_AVX
        const float* weightsPtr = weights.data();
        const float* inputsPtr = in.data();
        __m256 sum = _mm256_setzero_ps();
        for (; i + 8 <= numInputs; i += 8)
        {
            sum = _mm256_fmadd_ps(_mm256_load_ps(weightsPtr + i),
                                  _mm256_load_ps(inputsPtr + i),
                                  sum);
        }
        ctx.linearValue[0] += m256_hadd(sum);
#endif // USE_AVX

        for (; i < numInputs; i++)
        {
            ctx.linearValue[0] += weights[i] * ctx.inputs[i];
        }
    }
    else
    {
        // accumulate weights
        for (uint32_t j = 0; j < numInputs; j++)
        {
            const float inputValue = in[j];

            if (std::abs(inputValue) > 1.0e-7f)
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

    ctx.ComputeOutput(activationFunction);
}

void Layer::Run(uint32_t numFeatures, const uint16_t* featureIndices, LayerRunContext& ctx) const
{
    ctx.sparseBinaryInputs.resize(numFeatures);
    ctx.inputMode = InputMode::SparseBinary;

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        const uint16_t idx = featureIndices[i];
        ASSERT(idx < numInputs);
        ctx.sparseBinaryInputs[i] = idx;
    }

    // apply biases
    for (uint32_t i = 0; i < numOutputs; i++)
    {
        ctx.linearValue[i] = weights[numOutputs * numInputs + i];
    }

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

    ctx.ComputeOutput(activationFunction);
}

void Layer::Run(uint32_t numFeatures, const ActiveFeature* features, LayerRunContext& ctx) const
{
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
    for (uint32_t i = 0; i < numOutputs; i++)
    {
        ctx.linearValue[i] = weights[numOutputs * numInputs + i];
    }

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

    ctx.ComputeOutput(activationFunction);
}

void LayerRunContext::ComputeOutput(ActivationFunction activationFunction)
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
    if (activationFunction == ActivationFunction::ClippedReLu)
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
        output[i] = ApplyActivationFunction(linearValue[i], activationFunction);
    }
}

void Layer::Backpropagate(const Values& error, LayerRunContext& ctx, Gradients& gradients) const
{
    ASSERT(ctx.output.size() == error.size());
    ASSERT(ctx.output.size() <= PackedNeuralNetwork::MaxNeuronsInFirstLayer);
    alignas(CACHELINE_SIZE) float activationErrors[PackedNeuralNetwork::MaxNeuronsInFirstLayer];

    // precompute error gradients
    {
        size_t i = 0;
#ifdef USE_AVX
        if (activationFunction == ActivationFunction::ClippedReLu)
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
            activationErrors[i] = error[i] * GetActivationFunctionDerivative(ctx.linearValue[i], activationFunction);
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
            float* gradientPtr = gradients.values.data() + j * numOutputs;
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(gradientPtr + i,
                                _mm256_add_ps(_mm256_load_ps(activationErrors + i), _mm256_load_ps(gradientPtr + i)));
            }
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                // not multiplying by input value, because it's equal to 1.0
                gradients.values[j * numOutputs + i] += activationErrors[i];
            }
            gradients.dirty[j] = true;
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
            float* gradientPtr = gradients.values.data() + feature.index * numOutputs;
            const __m256 vInputValue = _mm256_set1_ps(feature.value);
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(gradientPtr + i,
                                _mm256_fmadd_ps(vInputValue, _mm256_load_ps(activationErrors + i), _mm256_load_ps(gradientPtr + i)));
            }
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                gradients.values[feature.index * numOutputs + i] += feature.value * activationErrors[i];
            }
            gradients.dirty[feature.index] = true;
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
                if (std::abs(activationError) > 1.0e-7f)
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
            if (std::abs(inputValue) > 1.0e-7f)
            {
                size_t i = 0;
#ifdef USE_AVX
                float* gradientPtr = gradients.values.data() + j * numOutputs;
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
                    gradients.values[j * numOutputs + i] += inputValue * activationErrors[i];
                }
                gradients.dirty[j] = true;
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
        float* gradientPtr = gradients.values.data() + numInputs * numOutputs;
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                            _mm256_add_ps(_mm256_load_ps(activationErrors + i),
                                          _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            gradients.values[numInputs * numOutputs + i] += activationErrors[i];
        }
    }
}

bool NeuralNetwork::Save(const char* filePath) const
{
    FILE* file = fopen(filePath, "wb");

    const uint32_t numLayers = (uint32_t)layers.size();
    if (1 != fwrite(&numLayers, sizeof(uint32_t), 1, file))
    {
        fclose(file);
        return false;
    }

    if (!layers.empty())
    {
        const uint32_t numLayerInputs = layers.front().numInputs;
        if (1 != fwrite(&numLayerInputs, sizeof(uint32_t), 1, file))
        {
            fclose(file);
            return false;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const uint32_t numLayerOutputs = layers[i].numOutputs;
        if (1 != fwrite(&numLayerOutputs, sizeof(uint32_t), 1, file))
        {
            fclose(file);
            return false;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const uint32_t numWeights = (uint32_t)layers[i].weights.size();
        if (numWeights != fwrite(layers[i].weights.data(), sizeof(float), numWeights, file))
        {
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

bool NeuralNetwork::Load(const char* filePath)
{
    FILE* file = fopen(filePath, "rb");

    if (!file)
    {
        std::cout << "Failed to load neural network: " << filePath << std::endl;
        return false;
    }

    uint32_t numLayers = 0;
    if (1 != fread(&numLayers, sizeof(uint32_t), 1, file))
    {
        fclose(file);
        return false;
    }

    if (numLayers == 0 || numLayers > 10)
    {
        std::cout << "Failed to load neural network. Invalid number of layers" << std::endl;
        fclose(file);
        return false;
    }

    uint32_t numInputs = 0;
    if (1 != fread(&numInputs, sizeof(uint32_t), 1, file))
    {
        fclose(file);
        return false;
    }

    if (numInputs == 0 || numInputs > 10000)
    {
        std::cout << "Failed to load neural network. Invalid number of first layer inputs" << std::endl;
        fclose(file);
        return false;
    }

    layers.clear();
    layers.reserve(numLayers);
    uint32_t prevLayerSize = numInputs;

    for (uint32_t i = 0; i < numLayers; i++)
    {
        uint32_t numLayerOutputs = 0;
        if (1 != fread(&numLayerOutputs, sizeof(uint32_t), 1, file))
        {
            fclose(file);
            return false;
        }

        if (numLayerOutputs == 0 || numInputs > 10000)
        {
            std::cout << "Failed to load neural network. Invalid number of layer outputs" << std::endl;
            return false;
        }

        layers.push_back(Layer(prevLayerSize, numLayerOutputs));
        layers[i].InitWeights();
        prevLayerSize = numLayerOutputs;
    }

    layers.back().activationFunction = ActivationFunction::Sigmoid;

    // read weights
    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const uint32_t numWeights = (uint32_t)layers[i].weights.size();
        if (numWeights != fread(layers[i].weights.data(), sizeof(float), numWeights, file))
        {
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

void NeuralNetwork::Init(uint32_t inputSize, const std::vector<uint32_t>& layersSizes, ActivationFunction outputLayerActivationFunc)
{
    layers.reserve(layersSizes.size());
    uint32_t prevLayerSize = inputSize;

    for (size_t i = 0; i < layersSizes.size(); i++)
    {
        layers.push_back(Layer(prevLayerSize, layersSizes[i]));
        prevLayerSize = layersSizes[i];
    }

    layers.back().activationFunction = outputLayerActivationFunc;

    for (size_t i = 0; i < layersSizes.size(); i++)
    {
        layers[i].InitWeights();
    }
}

const Values& NeuralNetwork::Run(const Values& input, NeuralNetworkRunContext& ctx) const
{
    ASSERT(layers.size() == ctx.layers.size());

    layers.front().Run(input, ctx.layers.front());

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Values& prevOutput = ctx.layers[i - 1].output;
        layers[i].Run(prevOutput, ctx.layers[i]);
    }

    return ctx.layers.back().output;
}

const Values& NeuralNetwork::Run(uint32_t numFeatures, const uint16_t* features, NeuralNetworkRunContext& ctx) const
{
    ASSERT(layers.size() == ctx.layers.size());

    layers.front().Run(numFeatures, features, ctx.layers.front());

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Values& prevOutput = ctx.layers[i - 1].output;
        layers[i].Run(prevOutput, ctx.layers[i]);
    }

    return ctx.layers.back().output;
}

const Values& NeuralNetwork::Run(uint32_t numFeatures, const ActiveFeature* features, NeuralNetworkRunContext& ctx) const
{
    ASSERT(layers.size() == ctx.layers.size());

    layers.front().Run(numFeatures, features, ctx.layers.front());

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Values& prevOutput = ctx.layers[i - 1].output;
        layers[i].Run(prevOutput, ctx.layers[i]);
    }

    return ctx.layers.back().output;
}

void Layer::UpdateWeights(float learningRate, const Gradients& gradients, const float gradientScale, const float weightsRange, const float biasRange, const float weightDecay)
{
    ASSERT(gradients.values.size() == (numInputs + 1) * numOutputs);

    const float cRho = 0.95f;
    const float cEpsilon = 1.0e-7f;

#ifdef USE_AVX
    const __m256 cOneMinusRhoVec = _mm256_set1_ps(1.0f - cRho);
    const __m256 cRhoVec = _mm256_set1_ps(cRho);
    const __m256 cEpsilonVec = _mm256_set1_ps(cEpsilon);
    const __m256 gradientScaleVec = _mm256_set1_ps(gradientScale);
    const __m256 signMaskVec = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
#endif

    for (size_t j = 0; j <= numInputs; j++)
    {
        const float maxWeightValue = j < numInputs ? weightsRange : biasRange;

        size_t i = 0;

#ifdef USE_AVX
        const __m256 minValueV = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_set1_ps(maxWeightValue));
        const __m256 maxValueV = _mm256_set1_ps(maxWeightValue);
        for (; i + 8 <= numOutputs; i += 8)
        {
            float* mPtr = gradientMean.data() + j * numOutputs + i;
            float* vPtr = gradientMoment.data() + j * numOutputs + i;
            float* wPtr = weights.data() + j * numOutputs + i;
            const float* gPtr = gradients.values.data() + j * numOutputs + i;

            __m256 g = _mm256_mul_ps(gradientScaleVec, _mm256_load_ps(gPtr));
            __m256 v = _mm256_load_ps(vPtr);
            __m256 m = _mm256_load_ps(mPtr);
            __m256 w = _mm256_load_ps(wPtr);

            // weight decay
            g = _mm256_fmadd_ps(w, _mm256_set1_ps(weightDecay), g);

            // ADADELTA algorithm
            m = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(g, g), _mm256_mul_ps(cRhoVec, m));
            __m256 delta = _mm256_mul_ps(g, _mm256_sqrt_ps(_mm256_div_ps(_mm256_add_ps(v, cEpsilonVec), _mm256_add_ps(m, cEpsilonVec))));
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
            float& m = gradientMean[j * numOutputs + i];
            float& v = gradientMoment[j * numOutputs + i];
            float& w = weights[j * numOutputs + i];
            float g = gradientScale * gradients.values[j * numOutputs + i];

            ASSERT(!std::isnan(g));
            ASSERT(v >= 0.0f);
            ASSERT(m >= 0.0f);

            // weight decay
            g += w * weightDecay;

            // ADADELTA algorithm
            m = cRho * m + (1.0f - cRho) * g * g;
            float delta = g * sqrtf((v + cEpsilon) / (m + cEpsilon));
            v = cRho * v + (1.0f - cRho) * delta * delta;
            w -= learningRate * delta;

            // clamping
            w = std::clamp(w, -maxWeightValue, maxWeightValue);

            ASSERT(!std::isnan(m));
            ASSERT(!std::isnan(v));
            ASSERT(!std::isnan(w));
        }
    }
}

NeuralNetworkTrainer::NeuralNetworkTrainer()
{
    m_perThreadData.resize(ThreadPool::GetInstance().GetNumThreads());
}

#pragma optimize("",off)

void NeuralNetworkTrainer::Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder)
{
    for (PerThreadData& threadData : m_perThreadData)
    {
        threadData.runContext.Init(network);
        threadData.gradients.resize(network.GetLayersNumber());
        for (size_t i = 0; i < network.layers.size(); ++i)
        {
            threadData.gradients[i].values.resize(network.layers[i].weights.size(), 0.0f);
            threadData.gradients[i].dirty.resize(network.layers[i].numInputs, false);
        }
    }

    const size_t numBatches = (trainingSet.size() + params.batchSize - 1) / params.batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        // clear accumulated gradients
        taskBuilder->ParallelFor("ClearGradients", (uint32_t)m_perThreadData.size(),
                                 [this, &network](const TaskContext&, uint32_t threadIdx)
        {
            PerThreadData& threadData = m_perThreadData[threadIdx];

            /*
            // at the first layer, clear only dirty gradients (most of them are zero)
            {
                Gradients& layerZeroGradients = threadData.gradients.front();
                const Layer& layerZero = network.layers.front();

                for (size_t i = 0; i < layerZero.numInputs; ++i)
                {
                    if (layerZeroGradients.dirty[i])
                    {
                        std::fill(layerZeroGradients.values.begin() + i * layerZero.numOutputs,
                                  layerZeroGradients.values.begin() + (i + 1) * layerZero.numOutputs,
                                  0.0f);
                    }
                }
                std::fill(layerZeroGradients.dirty.begin(), layerZeroGradients.dirty.end(), false);

                // clear gradients of the bias (it's always changing)
                std::fill(layerZeroGradients.values.begin() + layerZero.numInputs * layerZero.numOutputs,
                          layerZeroGradients.values.begin() + (layerZero.numInputs + 1) * layerZero.numOutputs,
                          0.0f);

                for (size_t i = 0; i < layerZeroGradients.values.size(); ++i)
                {
                    ASSERT(layerZeroGradients.values[i] == 0.0f);
                }
            }
            */

            // reset accumulated gradients for remaining layers
            for (size_t i = 0 /*1*/; i < network.layers.size(); ++i)
            {
                std::fill(threadData.gradients[i].values.begin(), threadData.gradients[i].values.end(), 0.0f);
            }
        });

        taskBuilder->Fence();

        taskBuilder->ParallelFor("Backpropagate", (uint32_t)params.batchSize,
                                 [this, &network, &trainingSet, batchIdx, params](const TaskContext& taskCtx, uint32_t indexInBatch)
        {
            PerThreadData& perThreadData = m_perThreadData[taskCtx.threadId];
            NeuralNetworkRunContext& ctx = perThreadData.runContext;

            const size_t vecIndex = batchIdx * params.batchSize + indexInBatch;
            if (vecIndex >= trainingSet.size()) return;

            const TrainingVector& vec = trainingSet[vecIndex];

            switch (vec.inputMode)
            {
            case InputMode::Full:
                ctx.tempValues = network.Run(vec.inputs, ctx);
                break;
            case InputMode::Sparse:
                ctx.tempValues = network.Run((uint32_t)vec.sparseInputs.size(), vec.sparseInputs.data(), ctx);
                break;
            case InputMode::SparseBinary:
                ctx.tempValues = network.Run((uint32_t)vec.sparseBinaryInputs.size(), vec.sparseBinaryInputs.data(), ctx);
                break;
            default:
                DEBUG_BREAK();
            }

            // train last layers
            {
                // compute gradient (error derivative)
                if (vec.outputMode == OutputMode::Single)
                {
                    ASSERT(ctx.tempValues.size() == 1);
                    ctx.tempValues[0] = 2.0f * (ctx.tempValues[0] - vec.singleOutput);
                }
                else
                {
                    for (size_t i = 0; i < ctx.tempValues.size(); i++)
                    {
                        // compute gradient (error derivative)
                        ctx.tempValues[i] = 2.0f * (ctx.tempValues[i] - vec.outputs[i]);
                    }
                }

                network.layers.back().Backpropagate(ctx.tempValues, ctx.layers.back(), perThreadData.gradients.back());
            }

            // train hidden layers
            if (network.layers.size() > 1)
            {
                for (size_t i = network.layers.size() - 1; i-- > 0; )
                {
                    network.layers[i].Backpropagate(ctx.layers[i + 1].inputGradient, ctx.layers[i], perThreadData.gradients[i]);
                }
            }
        });

        taskBuilder->Fence();

        taskBuilder->Task("UpdateWeights",
                          [this, &network, params](const TaskContext&)
        {
            const float gradientScale = 1.0f / (float)params.batchSize;
            for (size_t layerIdx = 0; layerIdx < network.layers.size(); ++layerIdx)
            {
                float weightQuantizationScale = 0.0f, biasQuantizationScale = 0.0f;
                float weightRange = 0.0f, biasRange = 0.0f;
                float weightDecay = 0.0f;

                if (layerIdx == 0) // input layer
                {
                    weightQuantizationScale = InputLayerWeightQuantizationScale;
                    biasQuantizationScale = InputLayerBiasQuantizationScale;
                    weightRange = std::numeric_limits<FirstLayerWeightType>::max();
                    biasRange = std::numeric_limits<FirstLayerBiasType>::max();
                    weightDecay = 2.0e-7f;
                }
                else if (layerIdx + 1 == network.layers.size()) // output layer
                {
                    weightQuantizationScale = OutputLayerWeightQuantizationScale;
                    biasQuantizationScale = OutputLayerBiasQuantizationScale;
                    weightRange = (float)std::numeric_limits<LastLayerWeightType>::max();
                    biasRange = (float)std::numeric_limits<LastLayerBiasType>::max();
                    weightDecay = 1.0e-6f;
                }
                else // hidden layer
                {
                    weightQuantizationScale = HiddenLayerWeightQuantizationScale;
                    biasQuantizationScale = HiddenLayerBiasQuantizationScale;
                    weightRange = (float)std::numeric_limits<HiddenLayerWeightType>::max();
                    biasRange = (float)std::numeric_limits<HiddenLayerBiasType>::max();
                    weightDecay = 1.0e-6f;
                }

                // accumulate gradients from all per-thread gradients
                for (size_t threadIdx = 1; threadIdx < m_perThreadData.size(); ++threadIdx)
                {
                    Values& targetGradients = m_perThreadData[0].gradients[layerIdx].values;
                    const Values& srcGradients = m_perThreadData[threadIdx].gradients[layerIdx].values;

                    const size_t numGradients = targetGradients.size();
                    ASSERT(srcGradients.size() == numGradients);

                    for (size_t i = 0; i < numGradients; ++i)
                    {
                        targetGradients[i] += srcGradients[i];
                    }
                }

                network.layers[layerIdx].UpdateWeights(
                    params.learningRate, m_perThreadData.front().gradients[layerIdx], gradientScale,\
                    params.clampWeights ? (weightRange / weightQuantizationScale) : 10000.0f,
                    params.clampWeights ? (biasRange / biasQuantizationScale) : 10000.0f,
                    weightDecay);
            }
        });
    }
}

template<typename WeightType, typename BiasType>
static void PackLayerWeights(const Layer& layer, WeightType* outWeights, BiasType* outBiases, float weightScale, float biasScale, bool transpose)
{
    // weights
    for (uint32_t j = 0; j < layer.numInputs; j++)
    {
        uint32_t i = 0;
#ifdef USE_AVX2
        const float* weightsPtr = layer.weights.data() + j * layer.numOutputs;
        for (; i + 8 < layer.numOutputs; i += 8)
        {
            const __m256i quantizedWeights =
                _mm256_cvtps_epi32(_mm256_round_ps(
                    _mm256_mul_ps(_mm256_load_ps(weightsPtr + i), _mm256_set1_ps(weightScale)),
                    _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));

            if (transpose)
            {
                outWeights[layer.numOutputs * j + (i + 0)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 0);
                outWeights[layer.numOutputs * j + (i + 1)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 1);
                outWeights[layer.numOutputs * j + (i + 2)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 2);
                outWeights[layer.numOutputs * j + (i + 3)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 3);
                outWeights[layer.numOutputs * j + (i + 4)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 4);
                outWeights[layer.numOutputs * j + (i + 5)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 5);
                outWeights[layer.numOutputs * j + (i + 6)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 6);
                outWeights[layer.numOutputs * j + (i + 7)] = (WeightType)_mm256_extract_epi32(quantizedWeights, 7);
            }
            else
            {
                outWeights[layer.numInputs * (i + 0) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 0);
                outWeights[layer.numInputs * (i + 1) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 1);
                outWeights[layer.numInputs * (i + 2) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 2);
                outWeights[layer.numInputs * (i + 3) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 3);
                outWeights[layer.numInputs * (i + 4) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 4);
                outWeights[layer.numInputs * (i + 5) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 5);
                outWeights[layer.numInputs * (i + 6) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 6);
                outWeights[layer.numInputs * (i + 7) + j] = (WeightType)_mm256_extract_epi32(quantizedWeights, 7);
            }
        }
#endif // USE_AVX2
        for (; i < layer.numOutputs; i++)
        {
            const float weight = layer.weights[j * layer.numOutputs + i];
            const int32_t quantizedWeight = (int32_t)std::round(weight * weightScale);
            ASSERT(quantizedWeight <= std::numeric_limits<WeightType>::max());
            ASSERT(quantizedWeight >= std::numeric_limits<WeightType>::min());

            if (transpose)
            {
                outWeights[layer.numOutputs * j + i] = (WeightType)quantizedWeight;
            }
            else
            {
                outWeights[layer.numInputs * i + j] = (WeightType)quantizedWeight;
            }
        }
    }

    // biases
    for (uint32_t i = 0; i < layer.numOutputs; i++)
    {
        const float bias = layer.weights[layer.numInputs * layer.numOutputs + i];
        const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
        ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
        ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());
        outBiases[i] = (BiasType)quantizedBias;
    }
}

bool NeuralNetwork::ToPackedNetwork(PackedNeuralNetwork& outNetwork) const
{
    ASSERT(layers.size() == 4);
    ASSERT(layers[0].numOutputs <= FirstLayerMaxSize);
    ASSERT(layers[1].numInputs <= FirstLayerMaxSize);
    ASSERT(layers[3].numOutputs == 1);

    if (!outNetwork.Resize(layers[0].numInputs,
                           layers[1].numInputs,
                           layers[2].numInputs,
                           layers[3].numInputs))
    {
        return false;
    }

    PackLayerWeights(layers[0], (FirstLayerWeightType*)outNetwork.GetAccumulatorWeights(), (FirstLayerBiasType*)outNetwork.GetAccumulatorBiases(), InputLayerWeightQuantizationScale, InputLayerBiasQuantizationScale, true);
    PackLayerWeights(layers[1], (HiddenLayerWeightType*)outNetwork.GetLayer1Weights(), (HiddenLayerBiasType*)outNetwork.GetLayer1Biases(), HiddenLayerWeightQuantizationScale, HiddenLayerBiasQuantizationScale, false);
    PackLayerWeights(layers[2], (HiddenLayerWeightType*)outNetwork.GetLayer2Weights(), (HiddenLayerBiasType*)outNetwork.GetLayer2Biases(), HiddenLayerWeightQuantizationScale, HiddenLayerBiasQuantizationScale, false);
    PackLayerWeights(layers[3], (LastLayerWeightType*)outNetwork.GetLayer3Weights(), (LastLayerBiasType*)outNetwork.GetLayer3Biases(), OutputLayerWeightQuantizationScale, OutputLayerBiasQuantizationScale, false);

    return true;
}

void NeuralNetwork::PrintStats() const
{
    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex)
    {
        const Layer& layer = layers[layerIndex];

        float minWeight = std::numeric_limits<float>::max();
        float maxWeight = -std::numeric_limits<float>::max();
        float minBias = std::numeric_limits<float>::max();
        float maxBias = -std::numeric_limits<float>::max();

        for (uint32_t i = 0; i < layer.numOutputs; i++)
        {
            float bias = layer.weights[layer.numInputs * layer.numOutputs + i];
            minBias = std::min(minBias, bias);
            maxBias = std::max(maxBias, bias);

            for (uint32_t j = 0; j < layer.numInputs; j++)
            {
                const float weight = layer.weights[j * layer.numOutputs + i];

                minWeight = std::min(minWeight, weight);
                maxWeight = std::max(maxWeight, weight);
            }
        }

        std::cout
            << "Layer #" << layerIndex
            << ": weight range: [" << minWeight << " ... " << maxWeight
            << "], bias range: [" << minBias << " ... " << maxBias
            << "]" << std::endl;
    }
}

} // namespace nn
