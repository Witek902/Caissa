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
        goto onError;
    }

    if (!layers.empty())
    {
        const uint32_t numLayerInputs = layers.front().numInputs;
        if (1 != fwrite(&numLayerInputs, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const uint32_t numLayerOutputs = layers[i].numOutputs;
        if (1 != fwrite(&numLayerOutputs, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }

        const uint32_t numVariants = (uint32_t)layers[i].variants.size();
        if (1 != fwrite(&numVariants, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const Layer& layer = layers[i];
        for (const Layer::Variant& variant : layer.variants)
        {
            const size_t numWeights = variant.weights.size();
            if (numWeights != fwrite(variant.weights.data(), sizeof(float), numWeights, file))
            {
                goto onError;
            }
        }
    }

    fclose(file);
    return true;

onError:
    fclose(file);
    return false;
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
    uint32_t numInputs = 0;
    uint32_t prevLayerSize = 0;

    if (1 != fread(&numLayers, sizeof(uint32_t), 1, file))
    {
        goto onError;
    }

    if (numLayers == 0 || numLayers > 10)
    {
        std::cout << "Failed to load neural network. Invalid number of layers" << std::endl;
        goto onError;
    }

    
    if (1 != fread(&numInputs, sizeof(uint32_t), 1, file))
    {
        goto onError;
    }

    if (numInputs == 0 || numInputs > 10000)
    {
        std::cout << "Failed to load neural network. Invalid number of first layer inputs" << std::endl;
        goto onError;
    }

    layers.clear();
    layers.reserve(numLayers);
    prevLayerSize = numInputs;

    for (uint32_t i = 0; i < numLayers; i++)
    {
        uint32_t numLayerOutputs = 0;
        if (1 != fread(&numLayerOutputs, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }

        uint32_t numLayerVariants = 0;
        if (1 != fread(&numLayerVariants, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }

        if (numLayerOutputs == 0 || numInputs > 10000)
        {
            std::cout << "Failed to load neural network. Invalid number of layer outputs" << std::endl;
            goto onError;
        }

        if (numLayerVariants == 0 || numLayerVariants > 10000)
        {
            std::cout << "Failed to load neural network. Invalid number of layer variants" << std::endl;
            goto onError;
        }

        layers.push_back(Layer(prevLayerSize, numLayerOutputs, numLayerVariants));
        layers[i].InitWeights();
        prevLayerSize = numLayerOutputs;
    }

    layers.back().activationFunc = ActivationFunction::Sigmoid;

    // read weights
    for (uint32_t i = 0; i < numLayers; ++i)
    {
        Layer& layer = layers[i];
        for (Layer::Variant& variant : layer.variants)
        {
            const size_t numWeights = variant.weights.size();
            if (numWeights != fread(variant.weights.data(), sizeof(float), numWeights, file))
            {
                std::cout << "Failed to load neural network weights" << std::endl;
                goto onError;
            }
        }
    }

    fclose(file);
    return true;

onError:
    fclose(file);
    return false;
}

void NeuralNetwork::Init(uint32_t inputSize, const std::vector<uint32_t>& layersSizes,
                         ActivationFunction outputLayerActivationFunc,
                         const std::vector<uint32_t>& layerVariants)
{
    layers.reserve(layersSizes.size());
    uint32_t prevLayerSize = inputSize;

    for (size_t i = 0; i < layersSizes.size(); i++)
    {
        const uint32_t numVariants = layerVariants.size() > i ? layerVariants[i] : 1u;
        layers.push_back(Layer(prevLayerSize, layersSizes[i], numVariants));
        prevLayerSize = layersSizes[i];
    }

    layers.back().activationFunc = outputLayerActivationFunc;

    for (size_t i = 0; i < layersSizes.size(); i++)
    {
        layers[i].InitWeights();
    }
}

const Values& NeuralNetwork::Run(const InputDesc& input, NeuralNetworkRunContext& ctx) const
{
    ASSERT(layers.size() == ctx.layers.size());

    // first layer
    {
        const Layer& layer = layers.front();
        const uint32_t variantIndex = layer.variants.size() > 0 ? input.variant : 0u;
        switch (input.mode)
        {
        case InputMode::Full:
            layer.Run(variantIndex, input.floatValues, ctx.layers.front());
            break;
        case InputMode::Sparse:
            layer.Run(variantIndex, input.numFeatures, input.floatFeatures, ctx.layers.front());
            break;
        case InputMode::SparseBinary:
            layer.Run(variantIndex, input.numFeatures, input.binaryFeatures, ctx.layers.front());
            break;
        }
    }
    
    for (size_t i = 1; i < layers.size(); i++)
    {
        const uint32_t variantIndex = layers[i].variants.size() > 0 ? input.variant : 0u;

        const float additionalBias = (i + 1) < layers.size() ? 0.0f : input.lastLayerBias;

        const Values& prevOutput = ctx.layers[i - 1].output;
        layers[i].Run(variantIndex, prevOutput.data(), ctx.layers[i], additionalBias);
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
            const Layer& layer = network.layers[i];
            threadData.gradients[i].resize(layer.variants.size());
            for (size_t j = 0; j < layer.variants.size(); ++j)
            {
                threadData.gradients[i][j].Init(network.layers[i].numInputs, network.layers[i].numOutputs);
            }
        }
    }

    const size_t numBatches = (trainingSet.size() + params.batchSize - 1) / params.batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        const auto clearGradientsFunc = [this, &network](uint32_t threadIdx) INLINE_LAMBDA
        {
            PerThreadData& threadData = m_perThreadData[threadIdx];

            // at the first layer, clear only dirty gradients (most of them are zero)
            for (auto& layerGradients : threadData.gradients.front())
            {
                layerGradients.Clear();
            }

            // reset accumulated gradients for remaining layers
            for (size_t i = 1; i < network.layers.size(); ++i)
            {
                for (auto& layerGradients : threadData.gradients[i])
                {
                    std::fill(layerGradients.m_values.begin(), layerGradients.m_values.end(), 0.0f);
                }
            }
        };

        const auto backpropagateFunc = [this, &network, &trainingSet, batchIdx, params](uint32_t threadIdx, uint32_t indexInBatch) INLINE_LAMBDA
        {
            PerThreadData& perThreadData = m_perThreadData[threadIdx];
            NeuralNetworkRunContext& ctx = perThreadData.runContext;

            const size_t vecIndex = batchIdx * params.batchSize + indexInBatch;
            if (vecIndex >= trainingSet.size()) return;

            const TrainingVector& vec = trainingSet[vecIndex];

            NeuralNetwork::InputDesc inputDesc;
            inputDesc.mode = vec.inputMode;
            inputDesc.variant = vec.networkVariant;
            inputDesc.lastLayerBias = vec.lastLayerBias;

            switch (vec.inputMode)
            {
            case InputMode::Full:
                inputDesc.floatValues = vec.inputs.data();
                break;
            case InputMode::Sparse:
                inputDesc.numFeatures = (uint32_t)vec.sparseInputs.size();
                inputDesc.floatFeatures = vec.sparseInputs.data();
                break;
            case InputMode::SparseBinary:
                inputDesc.numFeatures = (uint32_t)vec.sparseBinaryInputs.size();
                inputDesc.binaryFeatures = vec.sparseBinaryInputs.data(); break;
                break;
            default:
                DEBUG_BREAK();
            }

            ctx.tempValues = network.Run(inputDesc, ctx);

            // train last layers
            {
                const float errorScale = 1.0f;

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

                network.layers.back().Backpropagate(inputDesc.variant,
                                                    ctx.tempValues,
                                                    ctx.layers.back(),
                                                    perThreadData.gradients.back()[inputDesc.variant]);
            }

            // train hidden layers
            if (network.layers.size() > 1)
            {
                for (size_t i = network.layers.size() - 1; i-- > 0; )
                {
                    const Layer& layer = network.layers[i];
                    const uint32_t layerVariantIndex = inputDesc.variant < layer.variants.size() ? inputDesc.variant : 0;
                    layer.Backpropagate(inputDesc.variant,
                                        ctx.layers[i + 1].inputGradient,
                                        ctx.layers[i],
                                        perThreadData.gradients[i][layerVariantIndex]);
                }
            }
        };

        const auto updateWeightsFunc = [this, &network, params]() INLINE_LAMBDA
        {
            for (size_t layerIdx = 0; layerIdx < network.layers.size(); ++layerIdx)
            {
                Layer& layer = network.layers[layerIdx];

                Layer::WeightsUpdateOptions updateOptions;
                updateOptions.learningRate = params.learningRate;
                updateOptions.gradientScale = 1.0f; // (float)params.batchSize;

                float weightQuantizationScale = 0.0f, biasQuantizationScale = 0.0f;
                float weightRange = 0.0f, biasRange = 0.0f;
                if (layerIdx == 0) // input layer
                {
                    updateOptions.weightDecay = 1.0e-6f;
                    weightQuantizationScale = InputLayerWeightQuantizationScale;
                    biasQuantizationScale = InputLayerBiasQuantizationScale;
                    // divide by number of active input features to avoid accumulator overflow
                    weightRange = (float)std::numeric_limits<FirstLayerWeightType>::max() / 32;
                    biasRange = (float)std::numeric_limits<FirstLayerBiasType>::max() / 32;
                }
                else if (layerIdx + 1 == network.layers.size()) // output layer
                {
                    updateOptions.weightDecay = 1.0e-3f;
                    weightQuantizationScale = OutputLayerWeightQuantizationScale;
                    biasQuantizationScale = OutputLayerBiasQuantizationScale;
                    weightRange = (float)std::numeric_limits<LastLayerWeightType>::max();
                    biasRange = (float)std::numeric_limits<LastLayerBiasType>::max();
                }
                else // hidden layer
                {
                    updateOptions.weightDecay = 1.0e-3f;
                    weightQuantizationScale = HiddenLayerWeightQuantizationScale;
                    biasQuantizationScale = HiddenLayerBiasQuantizationScale;
                    weightRange = (float)std::numeric_limits<HiddenLayerWeightType>::max();
                    biasRange = (float)std::numeric_limits<HiddenLayerBiasType>::max();
                }

                updateOptions.weightsRange = params.clampWeights ? (weightRange / weightQuantizationScale) : 10000.0f;
                updateOptions.biasRange = params.clampWeights ? (biasRange / biasQuantizationScale) : 10000.0f;

                for (size_t variantIdx = 0; variantIdx < layer.variants.size(); ++variantIdx)
                {
                    // accumulate gradients from all per-thread gradients
                    {
                        MTR_SCOPE("NeuralNetworkTrainer::Train", "AccumulateGradients");
                        for (size_t threadIdx = 1; threadIdx < m_perThreadData.size(); ++threadIdx)
                        {
                            Gradients& targetGradients = m_perThreadData.front().gradients[layerIdx][variantIdx];
                            Gradients& srcGradients = m_perThreadData[threadIdx].gradients[layerIdx][variantIdx];

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

                    layer.UpdateWeights((uint32_t)variantIdx, m_perThreadData.front().gradients[layerIdx][variantIdx], updateOptions);
                }
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
static void PackLayerWeights(const Layer& layer, uint32_t variantIdx, WeightType* outWeights, BiasType* outBiases, float weightScale, float biasScale, bool transpose)
{
    const Layer::Variant& variant = layer.variants[variantIdx];

    // weights
    for (uint32_t j = 0; j < layer.numInputs; j++)
    {
        uint32_t i = 0;
#ifdef USE_AVX2
        const float* weightsPtr = variant.weights.data() + j * layer.numOutputs;
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
            const float weight = variant.weights[j * layer.numOutputs + i];
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
        const float bias = variant.weights[layer.numInputs * layer.numOutputs + i];
        const int32_t quantizedBias = (int32_t)std::round(bias * biasScale);
        ASSERT(quantizedBias <= std::numeric_limits<BiasType>::max());
        ASSERT(quantizedBias >= std::numeric_limits<BiasType>::min());
        outBiases[i] = (BiasType)quantizedBias;
    }
}

bool NeuralNetwork::ToPackedNetwork(PackedNeuralNetwork& outNetwork) const
{
    ASSERT(layers.size() <= PackedNeuralNetwork::MaxNumLayers);
    ASSERT(layers[0].numOutputs <= FirstLayerMaxSize);
    ASSERT(layers[1].numInputs <= FirstLayerMaxSize);
    ASSERT(layers.back().numOutputs == 1);
    ASSERT(layers.front().variants.size() == 1);

    {
        std::vector<uint32_t> layerSizes, layerVariants;
        for (const Layer& layer : layers)
        {
            layerSizes.push_back(layer.numInputs);
            layerVariants.push_back((uint32_t)layer.variants.size());
        }

        if (!outNetwork.Resize(layerSizes, layerVariants))
        {
            return false;
        }
    }

    // first layer
    PackLayerWeights(layers.front(),
                     0,
                     const_cast<FirstLayerWeightType*>(outNetwork.GetAccumulatorWeights()),
                     const_cast<FirstLayerBiasType*>(outNetwork.GetAccumulatorBiases()),
                     InputLayerWeightQuantizationScale,
                     InputLayerBiasQuantizationScale,
                     true);
    
    // hidden layers
    for (uint32_t i = 1; i + 1 < layers.size(); ++i)
    {
        for (uint32_t variantIdx = 0; variantIdx < layers[i].variants.size(); ++variantIdx)
        {
            PackLayerWeights(layers[i],
                             variantIdx,
                             const_cast<HiddenLayerWeightType*>(outNetwork.GetLayerWeights<HiddenLayerWeightType>(uint32_t(i), variantIdx)),
                             const_cast<HiddenLayerBiasType*>(outNetwork.GetLayerBiases<HiddenLayerBiasType>(uint32_t(i), variantIdx)),
                             HiddenLayerWeightQuantizationScale,
                             HiddenLayerBiasQuantizationScale,
                             false);
        }
    }

    // last layer
    const uint32_t lastLayerIndex = (uint32_t)layers.size() - 1;
    for (uint32_t variantIdx = 0; variantIdx < layers.back().variants.size(); ++variantIdx)
    {
        PackLayerWeights(layers.back(),
                         variantIdx,
                         const_cast<LastLayerWeightType*>(outNetwork.GetLayerWeights<LastLayerWeightType>(lastLayerIndex, variantIdx)),
                         const_cast<LastLayerBiasType*>(outNetwork.GetLayerBiases<LastLayerBiasType>(lastLayerIndex, variantIdx)),
                         OutputLayerWeightQuantizationScale,
                         OutputLayerBiasQuantizationScale,
                         false);
    }

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

        for (const Layer::Variant& variant : layer.variants)
        {
            for (uint32_t i = 0; i < layer.numOutputs; i++)
            {
                float bias = variant.weights[layer.numInputs * layer.numOutputs + i];
                minBias = std::min(minBias, bias);
                maxBias = std::max(maxBias, bias);

                for (uint32_t j = 0; j < layer.numInputs; j++)
                {
                    const float weight = variant.weights[j * layer.numOutputs + i];

                    minWeight = std::min(minWeight, weight);
                    maxWeight = std::max(maxWeight, weight);
                }
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
