#include "NeuralNetwork.hpp"
#include "PackedNeuralNetwork.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>

namespace nn {

Layer::Layer(size_t inputSize, size_t outputSize)
{
    activationFunction = ActivationFunction::ClippedReLu;

    linearValue.resize(outputSize);
    output.resize(outputSize);
    input.resize(inputSize);
    weights.resize((inputSize + 1) * outputSize);

    nextError.resize(input.size() + 1);
    gradient.resize(weights.size());
    m.resize(weights.size(), 0.0f);
    v.resize(weights.size(), 0.0f);
}

void Layer::InitWeights()
{
    const float scale = 2.0f / sqrtf((float)input.size());

    for (size_t i = 0; i < weights.size(); i++)
    {
        m[i] = 0.0f;
        v[i] = 0.0f;
    }

    size_t offs = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        for (size_t j = 0; j < input.size(); j++)
        {
            weights[++offs] = ((float)(rand() % RAND_MAX) / (float)(RAND_MAX)-0.5f) * scale;
        }
    }

    for (size_t j = 0; j < output.size(); j++)
    {
        weights[offs + j] = 0.0f;
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

void Layer::Run(const Values& in)
{
    ASSERT(in.size() == input.size());

    const size_t numInputs = input.size();
    const size_t numOutputs = output.size();
    ASSERT(numInputs % 8 == 0);

    input = in;

    // apply biases
    memcpy(linearValue.data(), weights.data() + numOutputs * numInputs, sizeof(float) * numOutputs);

    // accumulate weights
    for (size_t j = 0; j < numInputs; j++)
    {
        if (input[j] > 0.0f)
        {
            size_t i = 0;

#ifdef USE_AVX
            const float* weightsPtr = weights.data() + j * numOutputs;
            float* valuesPtr = linearValue.data();
            const __m256 inputValue = _mm256_set1_ps(input[j]);
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(valuesPtr + i,
                                _mm256_fmadd_ps(inputValue,
                                                _mm256_load_ps(weightsPtr + i),
                                                _mm256_load_ps(valuesPtr + i)));
            }
#endif // USE_AVX

            for (; i < numOutputs; i++)
            {
                linearValue[i] += weights[j * numOutputs + i] * input[j];
            }
        }
    }

    ComputeOutput();
}

void Layer::Run(const uint16_t* featureIndices, uint32_t numFeatures)
{
    activeFeatures.resize(numFeatures);

    const size_t numInputs = input.size();
    const size_t numOutputs = output.size();

    memset(input.data(), 0, numInputs * sizeof(float));

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        const uint16_t idx = featureIndices[i];
        ASSERT(idx < input.size());
        input[idx] = 1.0f;
        activeFeatures[i] = idx;
    }

    // apply biases
    for (size_t i = 0; i < numOutputs; i++)
    {
        linearValue[i] = weights[numOutputs * numInputs + i];
    }

    // accumulate active feature weights
    for (uint32_t j = 0; j < numFeatures; ++j)
    {
        const uint32_t idx = featureIndices[j];

        size_t i = 0;

#ifdef USE_AVX
        const float* weightsPtr = weights.data() + idx * numOutputs;
        float* valuesPtr = linearValue.data();
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(valuesPtr + i,
                            _mm256_add_ps(_mm256_load_ps(valuesPtr + i),
                                          _mm256_load_ps(weightsPtr + i)));
        }
#endif // USE_AVX

        for (; i < numOutputs; i++)
        {
            linearValue[i] += weights[idx * numOutputs + i];
        }
    }

    ComputeOutput();
}

void Layer::ComputeOutput()
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

void Layer::Backpropagate(uint32_t layerIndex, const Values& error)
{
    const size_t numInputs = input.size();
    const size_t numOutputs = output.size();

    ASSERT(output.size() <= PackedNeuralNetwork::MaxNeuronsInFirstLayer);
    alignas(CACHELINE_SIZE) float errorGradients[PackedNeuralNetwork::MaxNeuronsInFirstLayer];

    // precompute error gradients
    {
        size_t i = 0;
#ifdef USE_AVX
        if (activationFunction == ActivationFunction::ClippedReLu)
        {
            const float* errorsPtr = error.data();
            const float* valuesPtr = linearValue.data();
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(errorGradients + i,
                                ClippedReLuDerivative(_mm256_load_ps(valuesPtr + i), _mm256_load_ps(errorsPtr + i)));
            }
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            errorGradients[i] = error[i] * GetActivationFunctionDerivative(linearValue[i], activationFunction);
        }
    }

    if (layerIndex == 0)
    {
        // for first layer, use active feature indices and don't compute nextError (there's no more layers before to backpropagate)

        // update gradient of active features
        for (const uint16_t j : activeFeatures)
        {
            size_t i = 0;
#ifdef USE_AVX
            float* gradientPtr = gradient.data() + j * numOutputs;
            for (; i + 8 <= numOutputs; i += 8)
            {
                _mm256_store_ps(gradientPtr + i,
                                _mm256_add_ps(_mm256_load_ps(errorGradients + i), _mm256_load_ps(gradientPtr + i)));
            }
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                // not multiplying by input value, because it's equal to 1.0 in case of first layer
                gradient[j * numOutputs + i] += errorGradients[i];
            }
        }
    }
    else
    {
        // for later layers, use exact input values and compute nextError (for back propagation)

        // weight error propagation
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
                                          _mm256_load_ps(errorGradients + i),
                                          sum);
                }
                errorSum = m256_hadd(sum);
#endif // USE_AVX

                for (; i < numOutputs; i++)
                {
                    errorSum += weights[j * numOutputs + i] * errorGradients[i];
                }
            }
            nextError[j] = errorSum;

            if (input[j] > 0.0f)
            {
                size_t i = 0;
#ifdef USE_AVX
                float* gradientPtr = gradient.data() + j * numOutputs;
                const __m256 inputValue = _mm256_set1_ps(input[j]);
                for (; i + 8 <= numOutputs; i += 8)
                {
                    _mm256_store_ps(gradientPtr + i,
                                    _mm256_fmadd_ps(inputValue,
                                                    _mm256_load_ps(errorGradients + i),
                                                    _mm256_load_ps(gradientPtr + i)));
                }
#endif // USE_AVX
                for (; i < numOutputs; i++)
                {
                    gradient[j * numOutputs + i] += input[j] * errorGradients[i];
                }
            }
        }

        // bias error propagation
        {
            float errorSum = 0.0f;
            size_t i = 0;
#ifdef USE_AVX
            const float* weightsPtr = weights.data() + numInputs * numOutputs;
            __m256 sum = _mm256_setzero_ps();
            for (; i + 8 <= numOutputs; i += 8)
            {
                sum = _mm256_fmadd_ps(_mm256_load_ps(weightsPtr + i),
                                      _mm256_load_ps(errorGradients + i),
                                      sum);
            }
            errorSum = m256_hadd(sum);
#endif // USE_AVX
            for (; i < numOutputs; i++)
            {
                errorSum += weights[numInputs * numOutputs + i] * errorGradients[i];
            }

            nextError[numInputs] = errorSum;
        }
    }

    // update gradient for bias
    {
        size_t i = 0;
#ifdef USE_AVX
        float* gradientPtr = gradient.data() + numInputs * numOutputs;
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(gradientPtr + i,
                            _mm256_add_ps(_mm256_load_ps(errorGradients + i), _mm256_load_ps(gradientPtr + i)));
        }
#endif // USE_AVX
        for (; i < numOutputs; i++)
        {
            gradient[numInputs * numOutputs + i] += errorGradients[i];
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
        const uint32_t numLayerInputs = (uint32_t)layers.front().input.size();
        if (1 != fwrite(&numLayerInputs, sizeof(uint32_t), 1, file))
        {
            fclose(file);
            return false;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const uint32_t numLayerOutputs = (uint32_t)layers[i].output.size();
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

    uint32_t numLayers = (uint32_t)layers.size();
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
    tempError.resize(prevLayerSize);

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

void NeuralNetwork::Init(size_t inputSize, const std::vector<size_t>& layersSizes, ActivationFunction outputLayerActivationFunc)
{
    layers.reserve(layersSizes.size());
    size_t prevLayerSize = inputSize;

    for (size_t i = 0; i < layersSizes.size(); i++)
    {
        layers.push_back(Layer(prevLayerSize, layersSizes[i]));
        layers[i].InitWeights();
        prevLayerSize = layersSizes[i];
    }

    layers.back().activationFunction = outputLayerActivationFunc;
    tempError.resize(layersSizes.back());
}

const Values& NeuralNetwork::Run(const Values& input)
{
    layers.front().Run(input);

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Values& prevOutput = layers[i - 1].output;
        layers[i].Run(prevOutput);
    }

    return layers.back().output;
}

const Values& NeuralNetwork::Run(const uint16_t* featureIndices, uint32_t numFeatures)
{
    layers.front().Run(featureIndices, numFeatures);

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Values& prevOutput = layers[i - 1].output;
        layers[i].Run(prevOutput);
    }

    return layers.back().output;
}

void NeuralNetwork::UpdateLayerWeights(Layer& layer, float learningRate) const
{
    const size_t numInputs = layer.input.size();
    const size_t numOutputs = layer.output.size();
    const size_t numAllWeights = (numInputs + 1) * numOutputs;

    const float cRho = 0.95f;
    const float cEpsilon = 1.0e-7f;

    size_t i = 0;

#ifdef USE_AVX
    const __m256 cOneMinusRhoVec = _mm256_set1_ps(1.0f - cRho);
    const __m256 cRhoVec = _mm256_set1_ps(cRho);
    const __m256 cEpsilonVec = _mm256_set1_ps(cEpsilon);

    for (; i + 8 <= numAllWeights; i += 8)
    {
        float* mPtr = layer.m.data() + i;
        float* vPtr = layer.v.data() + i;
        float* wPtr = layer.weights.data() + i;
        const float* gPtr = layer.gradient.data() + i;

        __m256 v = _mm256_load_ps(vPtr);
        __m256 m = _mm256_load_ps(mPtr);
        __m256 w = _mm256_load_ps(wPtr);
        const __m256 g = _mm256_load_ps(gPtr);

        // ADADELTA algorithm
        m = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(g, g), _mm256_mul_ps(cRhoVec, m));
        __m256 delta = _mm256_mul_ps(g, _mm256_sqrt_ps(_mm256_div_ps(_mm256_add_ps(v, cEpsilonVec), _mm256_add_ps(m, cEpsilonVec))));
        v = _mm256_fmadd_ps(cOneMinusRhoVec, _mm256_mul_ps(delta, delta), _mm256_mul_ps(cRhoVec, v));
        w = _mm256_fnmadd_ps(delta, _mm256_set1_ps(learningRate), w);

        _mm256_store_ps(vPtr, v);
        _mm256_store_ps(mPtr, m);
        _mm256_store_ps(wPtr, w);
    }
#endif // USE_AVX

    for (; i < numAllWeights; ++i)
    {
        float& m = layer.m[i];
        float& v = layer.v[i];
        float& w = layer.weights[i];
        const float g = layer.gradient[i];

        // ADADELTA algorithm
        m = cRho * m + (1.0f - cRho) * g * g;
        float delta = g * sqrtf((v + cEpsilon) / (m + cEpsilon));
        v = cRho * v + (1.0f - cRho) * delta * delta;
        w -= delta * learningRate;

        ASSERT(!std::isnan(m));
        ASSERT(!std::isnan(v));
        ASSERT(!std::isnan(w));
    }
}

void NeuralNetwork::QuantizeLayerWeights(size_t layerIndex, float weightRange, float biasRange, float weightQuantizationScale, float biasQuantizationScale)
{
    Layer& layer = layers[layerIndex];
    const size_t numInputs = layer.input.size();
    const size_t numOutputs = layer.output.size();

    for (size_t j = 0; j < numInputs; j++)
    {
        const bool isBiasWeight = (j == numInputs);

        for (size_t i = 0; i < numOutputs; i++)
        {
            const size_t idx = j * numOutputs + i;

            float& w = layer.weights[idx];

            if (isBiasWeight)
            {
                w = std::clamp(w * biasQuantizationScale, -biasRange, biasRange) / biasQuantizationScale;
            }
            else
            {
                w = std::clamp(w * weightQuantizationScale, -weightRange, weightRange) / weightQuantizationScale;
            }

            // avoid rounding to zero
            if (layerIndex > 0 && std::abs(std::round(w * biasQuantizationScale)) < 1.0e-5f)
            {
                if (w > 0.0f)
                {
                    w = 1.0f / (isBiasWeight ? biasQuantizationScale : weightQuantizationScale);
                }
                else if (w < 0.0f)
                {
                    w = -1.0f / (isBiasWeight ? biasQuantizationScale : weightQuantizationScale);
                }
            }
        }
    }
}

void NeuralNetwork::Train(const std::vector<TrainingVector>& trainingSet, Values& tempValues, size_t batchSize, float learningRate)
{
    size_t numBatches = (trainingSet.size() + batchSize - 1) / batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        // reset accumulated weightsDelta
        for (Layer& layer : layers)
        {
            std::fill(layer.gradient.begin(), layer.gradient.end(), 0.0f);
        }

        // TODO parallel for
        for (size_t j = 0; j < batchSize; ++j)
        {
            size_t vecIndex = batchIdx * batchSize + j;
            if (vecIndex < trainingSet.size())
            {
                const TrainingVector& vec = trainingSet[vecIndex];

                if (!vec.inputs.empty())
                {
                    tempValues = Run(vec.inputs);
                }
                else
                {
                    tempValues = Run(vec.features.data(), (uint32_t)vec.features.size());
                }

                // train last layers
                {
                    for (size_t i = 0; i < tempError.size(); i++)
                    {
                        // TODO different loss function?
                        tempError[i] = (tempValues[i] - vec.output[i]);
                    }

                    layers.back().Backpropagate(uint32_t(layers.size() - 1), tempError);
                }

                // train hidden layers
                if (layers.size() > 1)
                {
                    for (size_t i = layers.size() - 1; i-- > 0; )
                    {
                        const Values& error = layers[i + 1].nextError;
                        layers[i].Backpropagate(uint32_t(i), error);
                    }
                }
            }
        }

        for (size_t i = layers.size(); i-- > 0; )
        {
            UpdateLayerWeights(layers[i], learningRate);
        }
    }

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
            weightRange = (float)std::numeric_limits<HiddenLayerWeightType>::max();
            biasRange = (float)std::numeric_limits<HiddenLayerBiasType>::max();
        }
        else // hidden layer
        {
            weightQuantizationScale = HiddenLayerWeightQuantizationScale;
            biasQuantizationScale = HiddenLayerBiasQuantizationScale;
            weightRange = (float)std::numeric_limits<HiddenLayerWeightType>::max();
            biasRange = (float)std::numeric_limits<HiddenLayerBiasType>::max();
        }

        QuantizeLayerWeights(i, weightRange, biasRange, weightQuantizationScale, biasQuantizationScale);
    }
}

template<typename WeightType, typename BiasType>
static void PackLayerWeights(const Layer& layer, WeightType* outWeights, BiasType* outBiases, float weightScale, float biasScale, bool transpose)
{
    const size_t numInputs = layer.input.size();
    const size_t numOutputs = layer.output.size();

    for (size_t i = 0; i < numOutputs; i++)
    {
        {
            const float bias = layer.weights[numInputs * numOutputs + i];
            const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
            ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
            ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());
            outBiases[i] = (BiasType)quantizedBias;
        }

        for (size_t j = 0; j < numInputs; j++)
        {
            const float weight = layer.weights[j * numOutputs + i];
            const int32_t quantizedWeight = (int32_t)std::round(weight * weightScale);
            ASSERT(quantizedWeight <= std::numeric_limits<WeightType>::max());
            ASSERT(quantizedWeight >= std::numeric_limits<WeightType>::min());

            if (transpose)
            {
                outWeights[numOutputs * j + i] = (WeightType)quantizedWeight;
            }
            else
            {
                outWeights[numInputs * i + j] = (WeightType)quantizedWeight;
            }
        }
    }
}

bool NeuralNetwork::ToPackedNetwork(PackedNeuralNetwork& outNetwork) const
{
    ASSERT(layers.size() == 4);
    ASSERT(layers[0].output.size() == FirstLayerSize);
    ASSERT(layers[1].input.size() == FirstLayerSize);
    ASSERT(layers[3].output.size() == 1);

    if (!outNetwork.Resize((uint32_t)layers[0].input.size(),
                           (uint32_t)layers[1].input.size(),
                           (uint32_t)layers[2].input.size(),
                           (uint32_t)layers[3].input.size()))
    {
        return false;
    }

    PackLayerWeights(layers[0], (FirstLayerWeightType*)outNetwork.GetAccumulatorWeights(), (FirstLayerBiasType*)outNetwork.GetAccumulatorBiases(), InputLayerWeightQuantizationScale, InputLayerBiasQuantizationScale, true);
    PackLayerWeights(layers[1], (HiddenLayerWeightType*)outNetwork.GetLayer1Weights(), (HiddenLayerBiasType*)outNetwork.GetLayer1Biases(), HiddenLayerWeightQuantizationScale, HiddenLayerBiasQuantizationScale, false);
    PackLayerWeights(layers[2], (HiddenLayerWeightType*)outNetwork.GetLayer2Weights(), (HiddenLayerBiasType*)outNetwork.GetLayer2Biases(), HiddenLayerWeightQuantizationScale, HiddenLayerBiasQuantizationScale, false);
    PackLayerWeights(layers[3], (HiddenLayerWeightType*)outNetwork.GetLayer3Weights(), (HiddenLayerBiasType*)outNetwork.GetLayer3Biases(), OutputLayerWeightQuantizationScale, OutputLayerBiasQuantizationScale, false);

    return true;
}

void NeuralNetwork::PrintStats() const
{
    float minWeight = std::numeric_limits<float>::max();
    float maxWeight = -std::numeric_limits<float>::max();

    float minBias = std::numeric_limits<float>::max();
    float maxBias = -std::numeric_limits<float>::max();

    for (const Layer& layer : layers)
    {
        const size_t numInputs = layer.input.size();
        const size_t numOutputs = layer.output.size();

        for (size_t i = 0; i < numOutputs; i++)
        {
            float bias = layer.weights[numInputs * numOutputs + i];
            minBias = std::min(minBias, bias);
            maxBias = std::max(maxBias, bias);

            for (size_t j = 0; j < numInputs; j++)
            {
                const float weight = layer.weights[j * numOutputs + i];

                minWeight = std::min(minWeight, weight);
                maxWeight = std::max(maxWeight, weight);
            }
        }
    }

    std::cout
        << "NN min weight:  " << minWeight << std::endl
        << "NN max weight:  " << maxWeight << std::endl
        << "NN min bias:    " << minBias << std::endl
        << "NN max bias:    " << maxBias << std::endl
        << std::endl;
}

} // namespace nn
