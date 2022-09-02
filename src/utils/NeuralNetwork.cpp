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

namespace nn {

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
        // Xavier initialization
        std::normal_distribution<float> weightDistr(0.0f, sqrtf(2.0f / (float)(numInputs + numOutputs)));
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

    ctx.input = in;
    ctx.useActiveFeaturesList = false;

    // apply biases
    memcpy(ctx.linearValue.data(), weights.data() + numOutputs * numInputs, sizeof(float) * numOutputs);

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
                ctx.linearValue[i] += weights[j * numOutputs + i] * ctx.input[j];
            }
        }
    }

    ctx.ComputeOutput(activationFunction);
}

void Layer::Run(const uint16_t* featureIndices, uint32_t numFeatures, LayerRunContext& ctx) const
{
    ctx.activeFeatures.resize(numFeatures);
    ctx.useActiveFeaturesList = true;

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        const uint16_t idx = featureIndices[i];
        ASSERT(idx < numInputs);
        ctx.activeFeatures[i] = idx;
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
    alignas(CACHELINE_SIZE) float activationGradients[PackedNeuralNetwork::MaxNeuronsInFirstLayer];

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
                _mm256_store_ps(activationGradients + i,
                                ClippedReLuDerivative(_mm256_load_ps(valuesPtr + i), _mm256_load_ps(errorsPtr + i)));
            }
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            activationGradients[i] = error[i] * GetActivationFunctionDerivative(ctx.linearValue[i], activationFunction);
        }
    }

    if (ctx.useActiveFeaturesList)
    {
        // for first layer, use active feature indices and don't compute nextError (there's no more layers before to backpropagate)

        // update gradient of active features
        for (const uint16_t j : ctx.activeFeatures)
        {
            size_t i = 0;
#ifdef USE_AVX
            float* gradientPtr = gradients.values.data() + j * numOutputs;
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(gradientPtr + i,
                                _mm256_add_ps(_mm256_load_ps(activationGradients + i), _mm256_load_ps(gradientPtr + i)));
            }
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                // not multiplying by input value, because it's equal to 1.0 in case of first layer
                gradients.values[j * numOutputs + i] += activationGradients[i];
            }
            gradients.dirty[j] = true;
        }
    }
    else
    {
        // for later layers, use exact input values and compute nextError (for back propagation)

        // compute input values gradient
        for (size_t j = 0; j < numInputs; j++)
        {
            float errorSum = 0.0f;
            {
                size_t i = 0;
#ifdef USE_AVX
                const float* weightsPtr = weights.data() + j * numOutputs;
                __m256 sum = _mm256_setzero_ps();
                for (; i + 8 <= numOutputs; i += 8)
                {
                    sum = _mm256_fmadd_ps(_mm256_load_ps(weightsPtr + i),
                                          _mm256_load_ps(activationGradients + i),
                                          sum);
                }
                errorSum = m256_hadd(sum);
#endif // USE_AVX

                for (; i < numOutputs; i++)
                {
                    errorSum += weights[j * numOutputs + i] * activationGradients[i];
                }
            }
            ctx.inputGradient[j] = errorSum;
        }

        // compute weights gradient
        for (size_t j = 0; j < numInputs; j++)
        {
            const float inputValue = ctx.input[j];
            if (std::abs(inputValue) > 1.0e-7f)
            {
                size_t i = 0;
#ifdef USE_AVX
                float* gradientPtr = gradients.values.data() + j * numOutputs;
                const __m256 vInputValue = _mm256_set1_ps(inputValue);
                for (; i + 8 <= numOutputs; i += 8)
                {
                    _mm256_store_ps(gradientPtr + i,
                                    _mm256_fmadd_ps(vInputValue,
                                                    _mm256_load_ps(activationGradients + i),
                                                    _mm256_load_ps(gradientPtr + i)));
                }
#endif // USE_AVX
                for (; i < numOutputs; i++)
                {
                    gradients.values[j * numOutputs + i] += ctx.input[j] * activationGradients[i];
                }
                gradients.dirty[j] = true;
            }
        }
    }

    // compute biases gradient
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradients.values.data() + numInputs * numOutputs;
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                            _mm256_add_ps(_mm256_load_ps(activationGradients + i), _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            gradients.values[numInputs * numOutputs + i] += activationGradients[i];
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

const Values& NeuralNetwork::Run(const uint16_t* featureIndices, uint32_t numFeatures, NeuralNetworkRunContext& ctx) const
{
    ASSERT(layers.size() == ctx.layers.size());

    layers.front().Run(featureIndices, numFeatures, ctx.layers.front());

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Values& prevOutput = ctx.layers[i - 1].output;
        layers[i].Run(prevOutput, ctx.layers[i]);
    }

    return ctx.layers.back().output;
}

void Layer::UpdateWeights_SGD(float learningRate, const Gradients& gradients)
{
    const size_t numAllWeights = (numInputs + 1) * numOutputs;
    ASSERT(gradients.values.size() == numAllWeights);

    const float cBeta = 0.9f;
    const float cAlpha = 0.01f * learningRate;

    size_t i = 0;

    for (; i < numAllWeights; ++i)
    {
        float& v = gradientMoment[i];
        float& w = weights[i];
        const float g = gradients.values[i];

        ASSERT(!std::isnan(g));

        v = cBeta * v + cAlpha * g;
        w -= v;

        ASSERT(!std::isnan(v));
        ASSERT(!std::isnan(w));
    }
}

void Layer::UpdateWeights_AdaDelta(float learningRate, const Gradients& gradients, const float gradientScale)
{
    const size_t numAllWeights = (numInputs + 1) * numOutputs;
    ASSERT(gradients.values.size() == numAllWeights);

    const float cDecay = 0.5e-6f;
    const float cRho = 0.95f;
    const float cEpsilon = 1.0e-7f;

    size_t i = 0;

#ifdef USE_AVX
    const __m256 cDecayVec = _mm256_set1_ps(cDecay);
    const __m256 cOneMinusRhoVec = _mm256_set1_ps(1.0f - cRho);
    const __m256 cRhoVec = _mm256_set1_ps(cRho);
    const __m256 cEpsilonVec = _mm256_set1_ps(cEpsilon);
    const __m256 gradientScaleVec = _mm256_set1_ps(gradientScale);

    for (; i + 8 <= numAllWeights; i += 8)
    {
        float* mPtr = gradientMean.data() + i;
        float* vPtr = gradientMoment.data() + i;
        float* wPtr = weights.data() + i;
        const float* gPtr = gradients.values.data() + i;

        const __m256 g = _mm256_mul_ps(gradientScaleVec, _mm256_load_ps(gPtr));
        __m256 v = _mm256_load_ps(vPtr);
        __m256 m = _mm256_load_ps(mPtr);

        // ADADELTA algorithm
        __m256 w = _mm256_load_ps(wPtr);
        m = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(g, g), _mm256_mul_ps(cRhoVec, m));
        __m256 delta = _mm256_mul_ps(g, _mm256_sqrt_ps(_mm256_div_ps(_mm256_add_ps(v, cEpsilonVec), _mm256_add_ps(m, cEpsilonVec))));
        delta = _mm256_fmadd_ps(cDecayVec, w, delta);
        v = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(delta, delta), _mm256_mul_ps(cRhoVec, v));
        w = _mm256_fnmadd_ps(delta, _mm256_set1_ps(learningRate), w);

        _mm256_store_ps(vPtr, v);
        _mm256_store_ps(mPtr, m);
        _mm256_store_ps(wPtr, w);
    }
#endif // USE_AVX

    for (; i < numAllWeights; ++i)
    {
        float& m = gradientMean[i];
        float& v = gradientMoment[i];
        float& w = weights[i];
        const float g = gradientScale * gradients.values[i];

        ASSERT(!std::isnan(g));
        ASSERT(v >= 0.0f);
        ASSERT(m >= 0.0f);

        // ADADELTA algorithm
        m = cRho * m + (1.0f - cRho) * g * g;
        float delta = g * sqrtf((v + cEpsilon) / (m + cEpsilon));
        v = cRho * v + (1.0f - cRho) * delta * delta;
        w -= (delta * learningRate + w * cDecay);

        ASSERT(!std::isnan(m));
        ASSERT(!std::isnan(v));
        ASSERT(!std::isnan(w));
    }
}

void NeuralNetwork::ClampLayerWeights(size_t layerIndex, float weightRange, float biasRange, float weightQuantizationScale, float biasQuantizationScale)
{
    biasRange *= 0.98f;
    weightRange *= 0.98f;

    Layer& layer = layers[layerIndex];

    for (uint32_t j = 0; j < layer.numInputs; j++)
    {
        const bool isBiasWeight = (j == layer.numInputs);

        for (uint32_t i = 0; i < layer.numOutputs; i++)
        {
            float& w = layer.weights[j * layer.numOutputs + i];

            if (isBiasWeight)
            {
                w = std::clamp(w * biasQuantizationScale, -biasRange, biasRange) / biasQuantizationScale;
            }
            else
            {
                w = std::clamp(w * weightQuantizationScale, -weightRange, weightRange) / weightQuantizationScale;
            }
        }
    }
}

void NeuralNetworkTrainer::Train(NeuralNetwork& network, const TrainingSet& trainingSet, size_t batchSize, float learningRate, bool clampWeights)
{
    perThreadRunContext.resize(1);
    perThreadRunContext[0].Init(network);

    gradients.resize(network.GetLayersNumber());
    for (size_t i = 0; i < network.layers.size(); ++i)
    {
        gradients[i].values.resize(network.layers[i].weights.size(), 0.0f);
        gradients[i].dirty.resize(network.layers[i].numInputs, false);
    }

    size_t numBatches = (trainingSet.size() + batchSize - 1) / batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        // at the first layer, clear only dirty gradients (most of them are zero)
        {
            Gradients& layerZeroGradients = gradients[0];
            const Layer& layerZero = network.layers[0];

            for (size_t i = 0; i < layerZero.numInputs; ++i)
            {
                if (layerZeroGradients.dirty[i])
                {
                    std::fill(layerZeroGradients.values.begin() + i * layerZero.numOutputs,
                              layerZeroGradients.values.begin() + (i + 1) * layerZero.numOutputs,
                              0.0f);
                    layerZeroGradients.dirty[i] = false;
                }
            }

            // clear griadients of the bias (it's always changing)
            std::fill(layerZeroGradients.values.begin() + layerZero.numInputs * layerZero.numOutputs,
                      layerZeroGradients.values.begin() + (layerZero.numInputs + 1) * layerZero.numOutputs,
                      0.0f);

            for (size_t i = 0; i < layerZeroGradients.values.size(); ++i)
            {
                ASSERT(layerZeroGradients.values[i] == 0.0f);
            }
        }

        // reset accumulated gradients
        for (size_t i = 1; i < network.layers.size(); ++i)
        {
            std::fill(gradients[i].values.begin(), gradients[i].values.end(), 0.0f);
        }

        for (uint32_t indexInBatch = 0; indexInBatch < batchSize; ++indexInBatch)
        {
            NeuralNetworkRunContext& ctx = perThreadRunContext.front();

            const size_t vecIndex = batchIdx * batchSize + indexInBatch;
            if (vecIndex < trainingSet.size())
            {
                const TrainingVector& vec = trainingSet[vecIndex];

                if (!vec.inputs.empty())
                {
                    ctx.tempValues = network.Run(vec.inputs, ctx);
                }
                else
                {
                    ctx.tempValues = network.Run(vec.features.data(), (uint32_t)vec.features.size(), ctx);
                }

                // train last layers
                {
                    for (size_t i = 0; i < ctx.tempValues.size(); i++)
                    {
                        // gradient of RMS loss function
                        ctx.tempValues[i] = ctx.tempValues[i] - vec.output[i];

                        // gradient of cross-entropy loss function
                        //const float target = vec.output[i];
                        //const float output = ctx.tempValues[i];
                        //ctx.tempValues[i] = (output - target) / std::clamp(1.0e-4f + output * (1.0f - output), -10.0f, 10.0f);
                    }

                    network.layers.back().Backpropagate(ctx.tempValues, ctx.layers.back(), gradients.back());
                }

                // train hidden layers
                if (network.layers.size() > 1)
                {
                    for (size_t i = network.layers.size() - 1; i-- > 0; )
                    {
                        network.layers[i].Backpropagate(ctx.layers[i + 1].inputGradient, ctx.layers[i], gradients[i]);
                    }
                }
            }
        }
        
        const float gradientScale = 1.0f / (float)batchSize;
        for (size_t i = 0; i < network.layers.size(); ++i)
        {
            network.layers[i].UpdateWeights_AdaDelta(learningRate, gradients[i], gradientScale);
        }
    }

    if (clampWeights)
    {
        network.ClampWeights();
    }
}

void NeuralNetwork::ClampWeights()
{
    for (size_t i = layers.size(); i-- > 0; )
    {
        float weightQuantizationScale = 0.0f, biasQuantizationScale = 0.0f;
        float weightRange = 0.0f, biasRange = 0.0f;

        if (i == 0) // input layer
        {
            weightQuantizationScale = InputLayerWeightQuantizationScale;
            biasQuantizationScale = InputLayerBiasQuantizationScale;
            weightRange = std::numeric_limits<FirstLayerWeightType>::max();
            biasRange = std::numeric_limits<FirstLayerBiasType>::max();
        }
        else if (i + 1 == layers.size()) // output layer
        {
            weightQuantizationScale = OutputLayerWeightQuantizationScale;
            biasQuantizationScale = OutputLayerBiasQuantizationScale;
            weightRange = (float)std::numeric_limits<LastLayerWeightType>::max();
            biasRange = (float)std::numeric_limits<LastLayerBiasType>::max();
        }
        else // hidden layer
        {
            weightQuantizationScale = HiddenLayerWeightQuantizationScale;
            biasQuantizationScale = HiddenLayerBiasQuantizationScale;
            weightRange = (float)std::numeric_limits<HiddenLayerWeightType>::max();
            biasRange = (float)std::numeric_limits<HiddenLayerBiasType>::max();
        }

        ClampLayerWeights(i, weightRange, biasRange, weightQuantizationScale, biasQuantizationScale);
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
