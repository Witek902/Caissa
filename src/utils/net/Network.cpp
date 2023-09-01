#include "Network.hpp"
#include "WeightsStorage.hpp"
#include "Gradient.hpp"
#include "FullyConnectedNode.hpp"
#include "SparseBinaryInputNode.hpp"
#include "SparseInputNode.hpp"
#include "CombiningNode.hpp"
#include "../ThreadPool.hpp"
#include "../../backend/PackedNeuralNetwork.hpp"
#include "../../backend/Waitable.hpp"
#include "../minitrace/minitrace.h"

#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef USE_SSE
#include <immintrin.h>
#endif // USE_SSE

using namespace threadpool;

namespace nn {

void NodeInput::Validate() const
{
    if (mode == InputMode::Full)
    {
        for (uint32_t i = 0; i < numFeatures; ++i)
        {
            ASSERT(!std::isnan(floatValues[i]));
        }
    }
    else if (mode == InputMode::SparseBinary)
    {
        // check for duplicates
        for (uint32_t i = 0; i < numFeatures; ++i)
        {
            for (uint32_t j = i + 1; j < numFeatures; ++j)
            {
                ASSERT(binaryFeatures[i] != binaryFeatures[j]);
            }
        }
    }
    else if (mode == InputMode::Sparse)
    {
        // check for duplicates
        for (uint32_t i = 0; i < numFeatures; ++i)
        {
            for (uint32_t j = i + 1; j < numFeatures; ++j)
            {
                ASSERT(floatFeatures[i].index != floatFeatures[j].index);
            }
        }

        for (uint32_t i = 0; i < numFeatures; ++i)
        {
            ASSERT(!std::isnan(floatFeatures[i].value));
        }
    }
    else
    {
        DEBUG_BREAK();
    }
}

void NeuralNetworkRunContext::Init(const NeuralNetwork& network)
{
    ASSERT(network.m_nodes.size() > 0);

    nodeContexts.resize(network.m_nodes.size());
    inputErrors.resize(network.m_nodes.size(), nullptr);

    // initialize context for each node
    for (size_t i = 0; i < network.m_nodes.size(); ++i)
    {
        ASSERT(!nodeContexts[i]);
        nodeContexts[i].reset(network.m_nodes[i]->CreateContext());
    }

    // assign input error pointers for each node
    for (size_t i = 0; i < network.m_nodes.size(); ++i)
    {
        const NodePtr& node = network.m_nodes[i];

        if (node->IsCombining())
        {
            const ICombiningNode* concatNode = static_cast<const ICombiningNode*>(node.get());
            ICombiningNode::Context& nodeCtx = static_cast<ICombiningNode::Context&>(*nodeContexts[i]);

            // match node context inputs to the outputs of the previous nodes
            for (size_t j = 0; j < i; j++)
            {
                if (network.m_nodes[j].get() == concatNode->GetInputNode(0))
                {
                    ASSERT(inputErrors[j] == nullptr);
                    inputErrors[j] = &nodeCtx.inputError;
                }
                else if (network.m_nodes[j].get() == concatNode->GetInputNode(1))
                {
                    ASSERT(inputErrors[j] == nullptr);
                    inputErrors[j] = &nodeCtx.secondaryInputError;
                }
            }
        }
    }

    // this assumes that the network is a linear chain
    for (size_t i = 0; i < network.m_nodes.size(); ++i)
    {
        if (!inputErrors[i] && i + 1 < network.m_nodes.size())
        {
            inputErrors[i] = &(nodeContexts[i + 1]->inputError);
        }
    }

    tempValues.resize(network.m_nodes.back()->GetNumOutputs());
}

bool NeuralNetwork::Save(const char* filePath) const
{
    // TODO
    (void)filePath;
    return false;

    /*
    FILE* file = fopen(filePath, "wb");

    const uint32_t numLayers = (uint32_t)m_nodes.size();
    if (1 != fwrite(&numLayers, sizeof(uint32_t), 1, file))
    {
        goto onError;
    }

    if (!m_nodes.empty())
    {
        const uint32_t numLayerInputs = m_nodes.front().numInputs;
        if (1 != fwrite(&numLayerInputs, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const uint32_t numLayerOutputs = m_nodes[i].numOutputs;
        if (1 != fwrite(&numLayerOutputs, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }

        const uint32_t numVariants = (uint32_t)m_nodes[i].variants.size();
        if (1 != fwrite(&numVariants, sizeof(uint32_t), 1, file))
        {
            goto onError;
        }
    }

    for (uint32_t i = 0; i < numLayers; ++i)
    {
        const INode& node = m_nodes[i];
        for (const INode::Variant& variant : node.variants)
        {
            const size_t numWeights = variant.weights.size();
            if (numWeights != fwrite(variant.weights.data(), sizeof(float), numWeights, file))
            {
                goto onError;
            }
            if (numWeights != fwrite(variant.gradientMoment1.data(), sizeof(float), numWeights, file))
            {
                goto onError;
            }
            if (numWeights != fwrite(variant.gradientMoment2.data(), sizeof(float), numWeights, file))
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
    */
}

bool NeuralNetwork::Load(const char* filePath)
{
    // TODO
    (void)filePath;
    return false;

    /*
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
        std::cout << "Failed to load neural network. Invalid number of m_nodes" << std::endl;
        goto onError;
    }

    
    if (1 != fread(&numInputs, sizeof(uint32_t), 1, file))
    {
        goto onError;
    }

    if (numInputs == 0 || numInputs > 10000)
    {
        std::cout << "Failed to load neural network. Invalid number of first node inputs" << std::endl;
        goto onError;
    }

    m_nodes.clear();
    m_nodes.reserve(numLayers);
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
            std::cout << "Failed to load neural network. Invalid number of node outputs" << std::endl;
            goto onError;
        }

        if (numLayerVariants == 0 || numLayerVariants > 10000)
        {
            std::cout << "Failed to load neural network. Invalid number of node variants" << std::endl;
            goto onError;
        }

        m_nodes.push_back(INode(prevLayerSize, numLayerOutputs, numLayerVariants));
        m_nodes[i].InitWeights();
        prevLayerSize = numLayerOutputs;
    }

    m_nodes.back().activationFunc = ActivationFunction::Sigmoid;

    // read weights
    for (uint32_t i = 0; i < numLayers; ++i)
    {
        INode& node = m_nodes[i];
        for (INode::Variant& variant : node.variants)
        {
            const size_t numWeights = variant.weights.size();
            if (numWeights != fread(variant.weights.data(), sizeof(float), numWeights, file))
            {
                std::cout << "Failed to load neural network weights" << std::endl;
                goto onError;
            }
            if (numWeights != fread(variant.gradientMoment1.data(), sizeof(float), numWeights, file))
            {
                std::cout << "Failed to load gradient 1st moments" << std::endl;
                goto onError;
            }
            if (numWeights != fread(variant.gradientMoment2.data(), sizeof(float), numWeights, file))
            {
                std::cout << "Failed to load gradient 2nd moments" << std::endl;
                goto onError;
            }
        }
    }

    fclose(file);
    return true;

onError:
    fclose(file);
    return false;
    */
}

void NeuralNetwork::Init(const std::vector<NodePtr>& nodes)
{
    m_nodes = nodes;
}

const Values& NeuralNetwork::Run(const InputDesc& inputDesc, NeuralNetworkRunContext& ctx) const
{
    ASSERT(m_nodes.size() == ctx.nodeContexts.size());

    for (size_t i = 0; i < m_nodes.size(); i++)
    {
        const NodePtr& node = m_nodes[i];

        if (node->IsCombining())
        {
            const ICombiningNode* concatNode = static_cast<const ICombiningNode*>(node.get());
            ICombiningNode::Context& nodeCtx = static_cast<ICombiningNode::Context&>(*ctx.nodeContexts[i]);

            // match node context inputs to the outputs of the previous nodes
            for (size_t j = 0; j < i; j++)
            {
                if (m_nodes[j].get() == concatNode->GetInputNode(0))
                {
                    nodeCtx.inputs = ctx.nodeContexts[j]->outputs;
                }
                else if (m_nodes[j].get() == concatNode->GetInputNode(1))
                {
                    nodeCtx.secondaryInputs = ctx.nodeContexts[j]->outputs;
                }
            }
        }
        else if (node->IsInputNode())
        {
            ASSERT(i < MaxInputNodes);
            const NodeInput& input = inputDesc.inputs[i];

            switch (input.mode)
            {
            case InputMode::Full:
            {
                FullyConnectedNode::Context& nodeCtx = static_cast<FullyConnectedNode::Context&>(*ctx.nodeContexts[i]);
                ASSERT(node->GetInputMode() == InputMode::Full);
                ASSERT(input.numFeatures == node->GetNumInputs());
                nodeCtx.inputs = std::span<const float>(input.floatValues, input.numFeatures);
                node->Run(nodeCtx);
                break;
            }
            case InputMode::Sparse:
            {
                SparseInputNode::Context& nodeCtx = static_cast<SparseInputNode::Context&>(*ctx.nodeContexts[i]);
                ASSERT(node->GetInputMode() == InputMode::Sparse);
                ASSERT(input.numFeatures <= node->GetNumInputs());
                nodeCtx.sparseInputs = std::span<const ActiveFeature>(input.floatFeatures, input.numFeatures);
                node->Run(nodeCtx);
                break;
            }
            case InputMode::SparseBinary:
            {
                SparseBinaryInputNode::Context& nodeCtx = static_cast<SparseBinaryInputNode::Context&>(*ctx.nodeContexts[i]);
                ASSERT(node->GetInputMode() == InputMode::SparseBinary);
                ASSERT(input.numFeatures <= node->GetNumInputs());
                nodeCtx.sparseInputs = std::span<const SparseBinaryInputNode::IndexType>(input.binaryFeatures, input.numFeatures);
                node->Run(nodeCtx);
                break;
            }
            default:
                ASSERT(false);
            }
        }
        else
        {
            ctx.nodeContexts[i]->inputs = ctx.nodeContexts[i - 1]->outputs;
        }

        ctx.nodeContexts[i]->variant = inputDesc.variant;

        node->Run(*ctx.nodeContexts[i]);
    }

    return ctx.nodeContexts.back()->outputs;
}

NeuralNetworkTrainer::NeuralNetworkTrainer()
{
    m_perThreadData.resize(ThreadPool::GetInstance().GetNumThreads());
}

void NeuralNetworkTrainer::Init(NeuralNetwork& network)
{
    // create a list of weights storages used by the network
    for (size_t i = 0; i < network.m_nodes.size(); ++i)
    {
        if (network.m_nodes[i]->IsTrainable())
        {
            WeightsStorage* weightsStorage = static_cast<const ITrainableNode*>(network.m_nodes[i].get())->GetWeightsStorage();
            ASSERT(weightsStorage);

            // check if the storage is already in the list
            if (std::find(m_weightsStorages.begin(), m_weightsStorages.end(), weightsStorage) == m_weightsStorages.end())
            {
                m_weightsStorages.push_back(weightsStorage);
            }
        }
    }

    for (PerThreadData& threadData : m_perThreadData)
    {
        threadData.runContext.Init(network);

        threadData.perWeightsStorageGradients.resize(m_weightsStorages.size());
        for (size_t i = 0; i < m_weightsStorages.size(); ++i)
        {
            const WeightsStorage* weightsStorage = m_weightsStorages[i];
            threadData.perWeightsStorageGradients[i].Init(
                weightsStorage->m_inputSize, weightsStorage->m_outputSize, (uint32_t)weightsStorage->m_variants.size(), weightsStorage->m_isSparse);
        }

        // assign per-node gradients pointers
        threadData.perNodeGradients.resize(network.m_nodes.size(), nullptr);
        for (size_t i = 0; i < network.m_nodes.size(); ++i)
        {
            if (network.m_nodes[i]->IsTrainable())
            {
                WeightsStorage* weightsStorage = static_cast<const ITrainableNode*>(network.m_nodes[i].get())->GetWeightsStorage();
                ASSERT(weightsStorage);

                // find the index of the storage in the list
                const auto it = std::find(m_weightsStorages.begin(), m_weightsStorages.end(), weightsStorage);
                ASSERT(it != m_weightsStorages.end());
                const size_t storageIdx = std::distance(m_weightsStorages.begin(), it);
                threadData.perNodeGradients[i] = &threadData.perWeightsStorageGradients[storageIdx];
            }
        }
    }
}

size_t NeuralNetworkTrainer::Train(NeuralNetwork& network, const TrainingSet& trainingSet, const TrainParams& params, threadpool::TaskBuilder* taskBuilder)
{
    const size_t numBatches = (trainingSet.size() + params.batchSize - 1) / params.batchSize;

    for (size_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        const auto clearGradientsFunc = [this](uint32_t threadIdx)
        {
            // clear gradients
            for (Gradients& gradients : m_perThreadData[threadIdx].perWeightsStorageGradients)
            {
                gradients.Clear();
            }
        };

        const auto backpropagateFunc = [this, &network, &trainingSet, batchIdx, params](uint32_t threadIdx, uint32_t indexInBatch)
        {
            PerThreadData& perThreadData = m_perThreadData[threadIdx];
            NeuralNetworkRunContext& ctx = perThreadData.runContext;

            const size_t vecIndex = batchIdx * params.batchSize + indexInBatch;
            if (vecIndex >= trainingSet.size()) return;

            const TrainingVector& vec = trainingSet[vecIndex];

            ctx.tempValues = network.Run(vec.input, ctx);

            // train last node
            {
                const float errorScale = 2.0f;

                // compute gradient (error derivative)
                if (vec.output.mode == OutputMode::Single)
                {
                    ASSERT(ctx.tempValues.size() == 1u);
                    ctx.tempValues[0] = errorScale * (ctx.tempValues[0] - vec.output.singleValue);
                }
                else if (vec.output.mode == OutputMode::Full)
                {
                    ASSERT(ctx.tempValues.size() == vec.output.numValues);
                    for (size_t i = 0; i < ctx.tempValues.size(); i++)
                    {
                        // compute gradient (error derivative)
                        ctx.tempValues[i] = errorScale * (ctx.tempValues[i] - vec.output.floatValues[i]);
                    }
                }
                else
                {
                    ASSERT(false);
                }

                ctx.nodeContexts.back()->variant = vec.input.variant;

                network.m_nodes.back()->Backpropagate(
                    ctx.tempValues,
                    *ctx.nodeContexts.back(),
                    *perThreadData.perNodeGradients.back());
            }

            // train hidden m_nodes
            if (network.m_nodes.size() > 1)
            {
                for (size_t i = network.m_nodes.size() - 1; i-- > 0; )
                {
                    const NodePtr& node = network.m_nodes[i];
                    ASSERT(node);
                    ASSERT(ctx.inputErrors[i]);

                    ctx.nodeContexts[i]->variant = vec.input.variant;

                    node->Backpropagate(*ctx.inputErrors[i],
                                        *ctx.nodeContexts[i],
                                        *perThreadData.perNodeGradients[i]);
                }
            }
        };

        /*
        const auto updateWeightsFunc = [this, batchIdx, params]()
        {
            for (size_t weightsStorageIndex = 0; weightsStorageIndex < m_weightsStorages.size(); ++weightsStorageIndex)
            {
                MTR_SCOPE("NeuralNetworkTrainer::Train", "UpdateWeights");

                WeightsStorage* weightsStorage = m_weightsStorages[weightsStorageIndex];
                ASSERT(weightsStorage);

                if (!weightsStorage->m_updateWeights) continue;

                WeightsStorage::WeightsUpdateOptions updateOptions;
                updateOptions.iteration = params.iteration + batchIdx;
                updateOptions.weightDecay = params.weightDecay;
                updateOptions.learningRate = params.learningRate;
                updateOptions.gradientScale = 1.0f; // 1.0f / (float)params.batchSize;

                Gradients& gradients = m_perThreadData.front().perWeightsStorageGradients[weightsStorageIndex];

                for (uint32_t inputIndex = 0; inputIndex <= weightsStorage->m_inputSize; ++inputIndex)
                {
                    // accumulate gradients from all per-thread gradients
                    for (size_t threadIdx = 1; threadIdx < m_perThreadData.size(); ++threadIdx)
                    {
                        Gradients& srcGradients = m_perThreadData[threadIdx].perWeightsStorageGradients[weightsStorageIndex];
                        gradients.Accumulate(srcGradients, inputIndex);
                    }

                    // apply weights update
                    switch (params.optimizer)
                    {
                    case Optimizer::Adadelta:
                        weightsStorage->Update_Adadelta(gradients, inputIndex, updateOptions);
                        break;
                    case Optimizer::Adam:
                        weightsStorage->Update_Adam(gradients, inputIndex, updateOptions);
                        break;
                    default:
                        DEBUG_BREAK();
                    }
                }
            }
        };
        */

        if (taskBuilder && params.batchSize > 64) // multi-threaded
        {
            if (batchIdx > 0)
            {
                taskBuilder->Fence();
            }

            // clear accumulated gradients
            taskBuilder->ParallelFor("ClearGradients", (uint32_t)m_perThreadData.size(),
                                     [clearGradientsFunc](const TaskContext&, uint32_t threadIdx)
            {
                clearGradientsFunc(threadIdx);
            });

            taskBuilder->Fence();

            taskBuilder->ParallelFor("Backpropagate", (uint32_t)params.batchSize,
                                     [backpropagateFunc](const TaskContext& taskCtx, uint32_t indexInBatch)
            {
                backpropagateFunc(taskCtx.threadId, indexInBatch);
            });

            taskBuilder->Fence();

            for (size_t weightsStorageIndex = 0; weightsStorageIndex < m_weightsStorages.size(); ++weightsStorageIndex)
            {
                MTR_SCOPE("NeuralNetworkTrainer::Train", "UpdateWeights");

                WeightsStorage* weightsStorage = m_weightsStorages[weightsStorageIndex];
                ASSERT(weightsStorage);

                if (!weightsStorage->m_updateWeights) continue;

                //for (uint32_t inputIndex = 0; inputIndex <= weightsStorage->m_inputSize; ++inputIndex)
                taskBuilder->ParallelFor("UpdateWeights", weightsStorage->m_inputSize + 1,
                    [this, weightsStorageIndex, params, batchIdx](const TaskContext&, uint32_t inputIndex)
                {
                    WeightsStorage* weightsStorage = m_weightsStorages[weightsStorageIndex];
                    ASSERT(weightsStorage);

                    Gradients& gradients = m_perThreadData.front().perWeightsStorageGradients[weightsStorageIndex];

                    // accumulate gradients from all per-thread gradients
                    for (size_t threadIdx = 1; threadIdx < m_perThreadData.size(); ++threadIdx)
                    {
                        Gradients& srcGradients = m_perThreadData[threadIdx].perWeightsStorageGradients[weightsStorageIndex];
                        gradients.Accumulate(srcGradients, inputIndex);
                    }

                    WeightsStorage::WeightsUpdateOptions updateOptions;
                    updateOptions.iteration = params.iteration + batchIdx;
                    updateOptions.weightDecay = params.weightDecay;
                    updateOptions.learningRate = params.learningRate;
                    updateOptions.gradientScale = 1.0f; // 1.0f / (float)params.batchSize;

                    // apply weights update
                    switch (params.optimizer)
                    {
                    case Optimizer::Adadelta:
                        weightsStorage->Update_Adadelta(gradients, inputIndex, updateOptions);
                        break;
                    case Optimizer::Adam:
                        weightsStorage->Update_Adam(gradients, inputIndex, updateOptions);
                        break;
                    default:
                        DEBUG_BREAK();
                    }
                });
            }

            taskBuilder->Fence();

            taskBuilder->Task("UpdateGradientsDirtyFlag", [this](const TaskContext&)
            {
                for (size_t weightsStorageIndex = 0; weightsStorageIndex < m_weightsStorages.size(); ++weightsStorageIndex)
                {
                    WeightsStorage* weightsStorage = m_weightsStorages[weightsStorageIndex];
                    ASSERT(weightsStorage);

                    if (!weightsStorage->m_updateWeights) continue;
                    if (!weightsStorage->m_isSparse) continue;

                    Gradients& gradients = m_perThreadData.front().perWeightsStorageGradients[weightsStorageIndex];

                    for (size_t threadIdx = 1; threadIdx < m_perThreadData.size(); ++threadIdx)
                    {
                        Gradients& srcGradients = m_perThreadData[threadIdx].perWeightsStorageGradients[weightsStorageIndex];
                        for (uint32_t i = 0; i < weightsStorage->m_inputSize + 1; ++i)
                        {
                            gradients.Accumulate_UpdateDirtyFlags(srcGradients, i);

                            //for (const Gradients::Variant& variant : srcGradients.m_variants)
                            //{
                            //    ASSERT(!variant.m_dirty[i]);
                            //}
                        }
                    }
                }
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

            //TODO
            //updateWeightsFunc();
        }
    }

    return numBatches;
}

void NeuralNetwork::PrintStats() const
{
    /*
    for (size_t layerIndex = 0; layerIndex < m_nodes.size(); ++layerIndex)
    {
        const INode& node = m_nodes[layerIndex];

        float minWeight = std::numeric_limits<float>::max();
        float maxWeight = -std::numeric_limits<float>::max();
        float minBias = std::numeric_limits<float>::max();
        float maxBias = -std::numeric_limits<float>::max();

        for (const INode::Variant& variant : node.variants)
        {
            for (uint32_t i = 0; i < node.numOutputs; i++)
            {
                float bias = variant.weights[node.numInputs * node.numOutputs + i];
                minBias = std::min(minBias, bias);
                maxBias = std::max(maxBias, bias);

                for (uint32_t j = 0; j < node.numInputs; j++)
                {
                    const float weight = variant.weights[j * node.numOutputs + i];

                    minWeight = std::min(minWeight, weight);
                    maxWeight = std::max(maxWeight, weight);
                }
            }
        }

        std::cout
            << "INode #" << layerIndex
            << ": weight range: [" << minWeight << " ... " << maxWeight
            << "], bias range: [" << minBias << " ... " << maxBias
            << "]" << std::endl;
    }
    */
}

} // namespace nn
