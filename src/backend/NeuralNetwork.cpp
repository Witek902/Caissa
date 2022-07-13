#include "NeuralNetwork.hpp"
#include "PackedNeuralNetwork.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

namespace nn {

static std::random_device gRandomDevice;
static std::mt19937 gRandomGenerator(gRandomDevice());

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
    const float scale = 1.0f / sqrtf((float)input.size());

    for (size_t i = 0; i < weights.size(); i++)
    {
        m[i] = 0.0f;
        v[i] = 0.0f;
    }

    size_t offs = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        weights[offs + input.size()] = 0.0f;

        for (size_t j = 0; j < input.size(); j++)
        {
            weights[offs + j] = ((float)(rand() % RAND_MAX) / (float)(RAND_MAX)-0.5f) * scale;
        }

        offs += input.size() + 1;
    }
}

static float ApplyActivationFunction(float x, ActivationFunction func)
{
    if (func == ActivationFunction::ClippedReLu)
    {
        return ClippedReLu(x);
    }
    else if (func == ActivationFunction::Sigmoid)
    {
        return Sigmoid(x);
    }
    else if (func == ActivationFunction::ATan)
    {
        return InvTan(x);
    }
    else
    {
        return x;
    }
}

void Layer::Run(const Values& in)
{
    ASSERT(in.size() == input.size());

    input = in;

    size_t offs = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        float x = weights[offs + input.size()];

        #pragma loop(hint_parallel(8))
        for (size_t j = 0; j < input.size(); j++)
        {
            x += weights[offs + j] * input[j];
        }

        ASSERT(!std::isnan(x));
        ASSERT(fabsf(x) < 10000.0f);

        linearValue[i] = x;
        output[i] = ApplyActivationFunction(x, activationFunction);

        offs += input.size() + 1;
    }
}

void Layer::Run(const uint16_t* featureIndices, uint32_t numFeatures)
{
    memset(input.data(), 0, input.size() * sizeof(float));

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        const uint16_t idx = featureIndices[i];
        ASSERT(idx < input.size());
        input[idx] = 1.0f;
    }

    size_t offs = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        float x = weights[offs + input.size()];
        for (uint32_t j = 0; j < numFeatures; ++j)
        {
            const uint16_t idx = featureIndices[j];
            x += weights[offs + idx];
        }

        ASSERT(!std::isnan(x));
        ASSERT(fabsf(x) < 10000.0f);

        linearValue[i] = x;
        output[i] = ApplyActivationFunction(x, activationFunction);

        offs += input.size() + 1;
    }
}

void Layer::Backpropagate(const Values& error)
{
    const size_t inputSize = input.size();

    size_t offs = 0;

    std::fill(nextError.begin(), nextError.end(), 0.0f);

    for (size_t i = 0; i < output.size(); i++)
    {
        float errorGradient = error[i];

        if (activationFunction == ActivationFunction::ClippedReLu)
        {
            errorGradient *= ClippedReLuDerivative(linearValue[i]);
        }
        else if (activationFunction == ActivationFunction::Sigmoid)
        {
            errorGradient *= SigmoidDerivative(linearValue[i]);
        }
        else if (activationFunction == ActivationFunction::ATan)
        {
            errorGradient *= InvTanDerivative(linearValue[i]);
        }

        #pragma loop(hint_parallel(8))
        for (size_t j = 0; j < inputSize; j++)
        {
            const size_t idx = offs + j;
            nextError[j] += weights[idx] * errorGradient;
            gradient[idx] += input[j] * errorGradient;
        }

        // update gradient for bias
        {
            const size_t idx = offs + inputSize;
            nextError[inputSize] += weights[idx] * errorGradient;
            gradient[idx] += errorGradient;
        }

        offs += inputSize + 1;
    }
}

float Layer::GetWeight(size_t neuronIdx, size_t neuronInputIdx) const
{
    size_t inputsNum = input.size();
    size_t outputsNum = output.size();

    (void)outputsNum;
    ASSERT(neuronIdx < outputsNum);
    ASSERT(neuronInputIdx <= inputsNum);

    return weights[(inputsNum + 1) * neuronIdx + neuronInputIdx];
}

void Layer::SetWeight(size_t neuronIdx, size_t neuronInputIdx, float newWeigth)
{
    size_t inputsNum = input.size();
    size_t outputsNum = output.size();

    (void)outputsNum;
    ASSERT(neuronIdx < outputsNum);
    ASSERT(neuronInputIdx <= inputsNum);

    weights[(inputsNum + 1) * neuronIdx + neuronInputIdx] = newWeigth;
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

const Layer::Values& NeuralNetwork::Run(const Layer::Values& input)
{
    layers.front().Run(input);

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Layer::Values& prevOutput = layers[i - 1].output;
        layers[i].Run(prevOutput);
    }

    return layers.back().output;
}

const Layer::Values& NeuralNetwork::Run(const uint16_t* featureIndices, uint32_t numFeatures)
{
    layers.front().Run(featureIndices, numFeatures);

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Layer::Values& prevOutput = layers[i - 1].output;
        layers[i].Run(prevOutput);
    }

    return layers.back().output;
}

void NeuralNetwork::UpdateLayerWeights(Layer& layer, float learningRate) const
{
    const size_t inputSize = layer.input.size();

    size_t offs = 0;

    for (size_t i = 0; i < layer.output.size(); i++)
    {
        for (size_t j = 0; j <= inputSize; j++)
        {
            const size_t idx = offs + j;

            float& m = layer.m[idx];
            float& v = layer.v[idx];
            float& w = layer.weights[idx];
            const float g = layer.gradient[idx];

            /*
            // ADADELTA algorithm
            const float rho = 0.01f;
            v = rho * v + (1.0f - rho) * g * g;
            const float delta = -sqrtf(m / v) * g;
            m = rho * m + (1.0f - rho) * delta;
            w += delta * scale;
            */

            const float cRho = 0.9f;
            const float cEpsilon = 1.0e-6f;

            // ADADELTA algorithm
            m = cRho * m + (1.0f - cRho) * g * g;
            float delta = sqrtf((v + cEpsilon) / (m + cEpsilon)) * g;
            v = cRho * v + (1.0f - cRho) * delta * delta;
            w -= delta * learningRate;

            ASSERT(!std::isnan(m));
            ASSERT(!std::isnan(v));
            ASSERT(!std::isnan(w));
        }

        offs += inputSize + 1;
    }
}

void NeuralNetwork::QuantizeLayerWeights(size_t layerIndex, float weightRange, float biasRange, float weightQuantizationScale, float biasQuantizationScale)
{
    Layer& layer = layers[layerIndex];
    const size_t inputSize = layer.input.size();

    size_t offs = 0;

    for (size_t i = 0; i < layer.output.size(); i++)
    {
        for (size_t j = 0; j <= inputSize; j++)
        {
            const size_t idx = offs + j;
            const bool isBiasWeight = (j == inputSize);

            float& w = layer.weights[idx];

            const float prevWeightValue = w;
            
            if (isBiasWeight)
            {
                w = std::clamp(w * biasQuantizationScale, -biasRange, biasRange) / biasQuantizationScale;
            }
            else
            {
                w = std::clamp(w * weightQuantizationScale, -weightRange, weightRange) / weightQuantizationScale;
            }

            // avoid rounding to zero
            if (layerIndex > 0 && std::abs(w) < FLT_EPSILON)
            {
                if (prevWeightValue > 0.0f)
                {
                    w = 1.0f / (isBiasWeight ? biasQuantizationScale : weightQuantizationScale);
                }
                else if (prevWeightValue < 0.0f)
                {
                    w = -1.0f / (isBiasWeight ? biasQuantizationScale : weightQuantizationScale);
                }
            }
        }

        offs += inputSize + 1;
    }
}

void NeuralNetwork::Train(const std::vector<TrainingVector>& trainingSet, Layer::Values& tempValues, size_t batchSize, float learningRate)
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
                        tempError[i] = (tempValues[i] - vec.output[i]);
                    }

                    layers.back().Backpropagate(tempError);
                }

                // train hidden layers
                if (layers.size() > 1)
                {
                    for (size_t i = layers.size() - 1; i-- > 0; )
                    {
                        const Layer::Values& error = layers[i + 1].nextError;
                        layers[i].Backpropagate(error);
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
    size_t offs = 0;
    for (size_t i = 0; i < layer.output.size(); i++)
    {
        {
            float bias = layer.weights[offs + layer.input.size()];
            const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
            ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
            ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());
            outBiases[i] = (BiasType)quantizedBias;
        }

        for (size_t j = 0; j < layer.input.size(); j++)
        {
            const float weight = layer.weights[offs + j];
            const int32_t quantizedWeight = (int32_t)std::round(weight * weightScale);
            ASSERT(quantizedWeight <= std::numeric_limits<WeightType>::max());
            ASSERT(quantizedWeight >= std::numeric_limits<WeightType>::min());

            if (transpose)
            {
                outWeights[layer.output.size() * j + i] = (WeightType)quantizedWeight;
            }
            else
            {
                outWeights[layer.input.size() * i + j] = (WeightType)quantizedWeight;
            }
        }

        offs += layer.input.size() + 1;
    }
}

bool NeuralNetwork::ToPackedNetwork(PackedNeuralNetwork& outNetwork) const
{
    ASSERT(layers.size() == 4);
    ASSERT(layers[0].output.size() == FirstLayerSize);
    ASSERT(layers[1].input.size() == FirstLayerSize);
    ASSERT(layers[1].output.size() == SecondLayerSize);
    ASSERT(layers[2].input.size() == SecondLayerSize);
    ASSERT(layers[2].output.size() == ThirdLayerSize);
    ASSERT(layers[3].input.size() == ThirdLayerSize);
    ASSERT(layers[3].output.size() == 1);

    if (!outNetwork.Resize((uint32_t)layers.front().input.size()))
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
        size_t offs = 0;
        for (size_t i = 0; i < layer.output.size(); i++)
        {
            float bias = layer.weights[offs + layer.input.size()];    
            minBias = std::min(minBias, bias);
            maxBias = std::max(maxBias, bias);

            for (size_t j = 0; j < layer.input.size(); j++)
            {
                const float weight = layer.weights[offs + j];

                minWeight = std::min(minWeight, weight);
                maxWeight = std::max(maxWeight, weight);
            }

            offs += layer.input.size() + 1;
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
