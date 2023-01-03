#include "NeuralNetwork.hpp"
#include "ThreadPool.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Waitable.hpp"
#include "minitrace/minitrace.h"

#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>

using namespace threadpool;

static constexpr float c_activationEpsilon = 1.0e-10f;

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
        else if (std::abs(sparseInputs[i].value) < c_activationEpsilon)
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

void NeuralNetworkRunContext::Init(const NeuralNetwork& network)
{
    layers.resize(network.layers.size());

    for (size_t i = 0; i < network.layers.size(); ++i)
    {
        layers[i].Init(network.layers[i]);
    }

    tempValues.resize(network.GetOutputSize());
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

NeuralNetworkTrainer::NeuralNetworkTrainer()
{
    m_perThreadData.resize(ThreadPool::GetInstance().GetNumThreads());
}

void NeuralNetworkTrainer::Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder)
{
    for (PerThreadData& threadData : m_perThreadData)
    {
        threadData.runContext.Init(network);
        threadData.gradients.resize(network.GetLayersNumber());
        for (size_t i = 0; i < network.layers.size(); ++i)
        {
            threadData.gradients[i].Init(network.layers[i].numInputs, network.layers[i].numOutputs);
        }
    }

    const size_t numBatches = (trainingSet.size() + params.batchSize - 1) / params.batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        const auto clearGradientsFunc = [this, &network](uint32_t threadIdx) INLINE_LAMBDA
        {
            PerThreadData& threadData = m_perThreadData[threadIdx];

            // at the first layer, clear only dirty gradients (most of them are zero)
            threadData.gradients.front().Clear();

            // reset accumulated gradients for remaining layers
            for (size_t i = 1; i < network.layers.size(); ++i)
            {
                std::fill(threadData.gradients[i].m_values.begin(), threadData.gradients[i].m_values.end(), 0.0f);
            }
        };

        const auto backpropagateFunc = [this, &network, &trainingSet, batchIdx, params](uint32_t threadIdx, uint32_t indexInBatch) INLINE_LAMBDA
        {
            PerThreadData& perThreadData = m_perThreadData[threadIdx];
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
                const float errorScale = 2.0f;

                // compute gradient (error derivative)
                if (vec.outputMode == OutputMode::Single)
                {
                    ASSERT(ctx.tempValues.size() == 1);
                    ctx.tempValues[0] = errorScale * (ctx.tempValues[0] - vec.singleOutput);
                }
                else
                {
                    for (size_t i = 0; i < ctx.tempValues.size(); i++)
                    {
                        // compute gradient (error derivative)
                        ctx.tempValues[i] = errorScale * (ctx.tempValues[i] - vec.outputs[i]);
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
        };

        const auto updateWeightsFunc = [this, &network, params]() INLINE_LAMBDA
        {
            const float gradientScale = 1.0f;  // (float)params.batchSize;
            for (size_t layerIdx = 0; layerIdx < network.layers.size(); ++layerIdx)
            {
                Layer& layer = network.layers[layerIdx];

                float weightQuantizationScale = 0.0f, biasQuantizationScale = 0.0f;
                float weightRange = 0.0f, biasRange = 0.0f;
                float weightDecay = 0.0f;

                if (layerIdx == 0) // input layer
                {
                    weightQuantizationScale = InputLayerWeightQuantizationScale;
                    biasQuantizationScale = InputLayerBiasQuantizationScale;
                    // divide by number of active input features to avoid accumulator overflow
                    weightRange = (float)std::numeric_limits<FirstLayerWeightType>::max() / 32;
                    biasRange = (float)std::numeric_limits<FirstLayerBiasType>::max() / 32;
                    weightDecay = 1.0e-7f;
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
                {
                    MTR_SCOPE("NeuralNetworkTrainer::Train", "AccumulateGradients");
                    for (size_t threadIdx = 1; threadIdx < m_perThreadData.size(); ++threadIdx)
                    {
                        Gradients& targetGradients = m_perThreadData.front().gradients[layerIdx];
                        Gradients& srcGradients = m_perThreadData[threadIdx].gradients[layerIdx];

                        const size_t numGradients = targetGradients.m_values.size();
                        ASSERT(srcGradients.m_values.size() == numGradients);

                        if (layerIdx == 0)
                        {
                            // in case of first layer copy only dirty gradients
                            targetGradients.Accumulate(srcGradients);
                        }
                        else
                        {
                            for (size_t i = 0; i <= layer.numInputs; ++i)
                            {
                                targetGradients.m_dirty[i] = true;
                            }

                            for (size_t i = 0; i < numGradients; ++i)
                            {
                                targetGradients.m_values[i] += srcGradients.m_values[i];
                            }
                        }
                    }
                }

                layer.UpdateWeights(
                    params.learningRate, m_perThreadData.front().gradients[layerIdx], gradientScale,
                    params.clampWeights ? (weightRange / weightQuantizationScale) : 10000.0f,
                    params.clampWeights ? (biasRange / biasQuantizationScale) : 10000.0f,
                    weightDecay);
            }
        };

        if (taskBuilder) // multi-threaded
        {
            if (batchIdx > 0)
            {
                taskBuilder->Fence();
            }

            // clear accumulated gradients
            taskBuilder->ParallelFor("ClearGradients", (uint32_t)m_perThreadData.size(),
                                     [this, &network, clearGradientsFunc](const TaskContext&, uint32_t threadIdx)
            {
                clearGradientsFunc(threadIdx);
            });

            taskBuilder->Fence();

            taskBuilder->ParallelFor("Backpropagate", (uint32_t)params.batchSize,
                                     [this, &network, &trainingSet, backpropagateFunc, batchIdx, params](const TaskContext& taskCtx, uint32_t indexInBatch)
            {
                backpropagateFunc(taskCtx.threadId, indexInBatch);
            });

            taskBuilder->Fence();

            taskBuilder->Task("UpdateWeights",
                              [this, &network, updateWeightsFunc, params](const TaskContext&)
            {
                updateWeightsFunc();
            });
        }
        else // single-threaded
        {
            const uint32_t dummyThreadIdx = 0;

            clearGradientsFunc(dummyThreadIdx);

            for (uint32_t indexInBatch = 0; indexInBatch < (uint32_t)params.batchSize; ++indexInBatch)
            {
                backpropagateFunc(dummyThreadIdx, indexInBatch);
            }

            updateWeightsFunc();
        }
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
