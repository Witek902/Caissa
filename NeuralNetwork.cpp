#include "NeuralNetwork.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nn {

static const float cAdamLearningRate = 0.001f;
static const float cAdamBeta1 = 0.9f;
static const float cAdamBeta2 = 0.999f;
static const float cAdamEpsilon = 1.0e-8f;

// min/max value of a weight
static const float cWeightsRange = 127.0f;

Layer::Layer(size_t inputSize, size_t outputSize)
{
    activationFunction = ActivationFunction::ReLu;

    linearValue.resize(outputSize);
    output.resize(outputSize);
    input.resize(inputSize);
    weights.resize((inputSize + 1) * outputSize);

    nextError.resize(input.size() + 1);
    gradient.resize(weights.size());
    adam_m.resize(weights.size(), 0.0f);
    adam_v.resize(weights.size(), 0.0f);
}

void Layer::InitWeights()
{
    //const float scale = 0.1f;
    const float scale = 1.0f / sqrtf((float)input.size());

    for (size_t i = 0; i < weights.size(); i++)
    {
        adam_m[i] = 0.0f;
        adam_v[i] = 0.0f;
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

void Layer::Run(const Values& in)
{
    assert(in.size() == input.size());

    input = in;

    size_t offs = 0;
    for (size_t i = 0; i < output.size(); i++)
    {
        float x = weights[offs + input.size()];
        for (size_t j = 0; j < input.size(); j++)
        {
            x += weights[offs + j] * input[j];
        }

        assert(!std::isnan(x));
        assert(fabsf(x) < 10000.0f);

        linearValue[i] = x;

        if (activationFunction == ActivationFunction::ReLu)
        {
            output[i] = ReLu(x);
        }
        else if (activationFunction == ActivationFunction::Sigmoid)
        {
            output[i] = Sigmoid(x);
        }
        else if (activationFunction == ActivationFunction::ATan)
        {
            output[i] = InvTan(x);
        }
        else
        {
            output[i] = x;
        }

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

        if (activationFunction == ActivationFunction::ReLu)
        {
            errorGradient *= ReLuDerivative(linearValue[i]);
        }
        else if (activationFunction == ActivationFunction::Sigmoid)
        {
            errorGradient *= SigmoidDerivative(linearValue[i]);
        }
        else if (activationFunction == ActivationFunction::ATan)
        {
            errorGradient *= InvTanDerivative(linearValue[i]);
        }

        for (size_t j = 0; j <= inputSize; j++)
        {
            size_t idx = offs + j;
            const float inputValue = j < inputSize ? input[j] : 1.0f;

            nextError[j] += weights[idx] * errorGradient;
            gradient[idx] += inputValue * errorGradient;
        }

        offs += inputSize + 1;
    }
}

float Layer::GetWeight(size_t neuronIdx, size_t neuronInputIdx) const
{
    size_t inputsNum = input.size();
    size_t outputsNum = output.size();

    (void)outputsNum;
    assert(neuronIdx < outputsNum);
    assert(neuronInputIdx <= inputsNum);

    return weights[(inputsNum + 1) * neuronIdx + neuronInputIdx];
}

void Layer::SetWeight(size_t neuronIdx, size_t neuronInputIdx, float newWeigth)
{
    size_t inputsNum = input.size();
    size_t outputsNum = output.size();

    (void)outputsNum;
    assert(neuronIdx < outputsNum);
    assert(neuronInputIdx <= inputsNum);

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
        return false;
    }

    adamBeta1 = cAdamBeta1;
    adamBeta2 = cAdamBeta2;

    uint32_t numInputs = 0;
    if (1 != fread(&numInputs, sizeof(uint32_t), 1, file))
    {
        fclose(file);
        return false;
    }

    if (numInputs == 0 || numInputs > 10000)
    {
        std::cout << "Failed to load neural network. Invalid number of first layer inputs" << std::endl;
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

    layers.back().activationFunction = ActivationFunction::Linear;
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

void NeuralNetwork::Init(size_t inputSize, const std::vector<size_t>& layersSizes)
{
    adamBeta1 = cAdamBeta1;
    adamBeta2 = cAdamBeta2;

    layers.reserve(layersSizes.size());
    size_t prevLayerSize = inputSize;

    for (size_t i = 0; i < layersSizes.size(); i++)
    {
        layers.push_back(Layer(prevLayerSize, layersSizes[i]));
        layers[i].InitWeights();
        prevLayerSize = layersSizes[i];
    }

    layers.back().activationFunction = ActivationFunction::Linear;
    tempError.resize(layersSizes.back());
}

const Layer::Values& NeuralNetwork::Run(const Layer::Values& input)
{
    layers.front().Run(input);

    for (size_t i = 1; i < layers.size(); i++)
    {
        const Layer::Values& prevOutput = layers[i - 1].GetOutput();
        layers[i].Run(prevOutput);
    }

    return layers.back().GetOutput();
}

void NeuralNetwork::UpdateLayerWeights(Layer& layer, float scale) const
{
    const size_t inputSize = layer.input.size();

    size_t offs = 0;

    for (size_t i = 0; i < layer.output.size(); i++)
    {
        for (size_t j = 0; j <= inputSize; j++)
        {
            const size_t idx = offs + j;

            float& m = layer.adam_m[idx];
            float& v = layer.adam_v[idx];
            float& w = layer.weights[idx];
            const float g = layer.gradient[idx];

            m = cAdamBeta1 * m + (1.0f - cAdamBeta1) * g;
            v = cAdamBeta2 * v + (1.0f - cAdamBeta2) * g * g;
            float hm = m / (1.0f - adamBeta1); // +(1.0f - cAdamBeta1) * g / (1.0f - adamBeta1);
            float hv = v / (1.0f - adamBeta2);

            w -= hm / (sqrtf(hv) + cAdamEpsilon) * scale;

            //w -= g * scale;

            assert(!std::isnan(m));
            assert(!std::isnan(v));
            assert(!std::isnan(w));
            assert(fabsf(w) < cWeightsRange);

            // clamp
            //w = std::max(-cWeightsRange, std::min(cWeightsRange, w));
        }

        offs += inputSize + 1;
    }
}

void NeuralNetwork::Train(const std::vector<TrainingVector>& trainingSet, Layer::Values& tempValues, size_t batchSize)
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

                tempValues = Run(vec.input);

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
                        const Layer::Values& error = layers[i + 1].GetNextError();
                        layers[i].Backpropagate(error);
                    }
                }
            }
        }

        //const float updateScale = cAdamLearningRate / (float)batchSize;
        const float updateScale = cAdamLearningRate;

        for (size_t i = layers.size(); i-- > 0; )
        {
            UpdateLayerWeights(layers[i], updateScale);
        }
    }
}

void NeuralNetwork::NextEpoch()
{
    adamBeta1 *= cAdamBeta1;
    adamBeta2 *= cAdamBeta2;
}

} // namespace nn
