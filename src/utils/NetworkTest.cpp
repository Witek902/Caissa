#include "Common.hpp"
#include "net/Network.hpp"
#include "net/SparseBinaryInputNode.hpp"
#include "net/FullyConnectedNode.hpp"
#include "net/ActivationNode.hpp"
#include "net/WeightsStorage.hpp"
#include "../backend/PackedNeuralNetwork.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

// #define USE_PACKED_NET

static const uint32_t cNumTrainingVectorsPerIteration = 4;
static const uint32_t cBatchSize = 10;

bool TestNetwork()
{
    nn::NeuralNetwork network;

    const uint32_t numNetworkInputs = 2;
    const uint32_t hiddenLayerSize = 64;

    nn::WeightsStoragePtr layer1Weights = std::make_shared<nn::WeightsStorage>(numNetworkInputs, hiddenLayerSize, 1);
    nn::WeightsStoragePtr layer2Weights = std::make_shared<nn::WeightsStorage>(hiddenLayerSize, 1, 1);

    layer1Weights->Init(64);
    layer2Weights->Init(64);

    nn::NodePtr inputNode = std::make_shared<nn::SparseBinaryInputNode>(numNetworkInputs, hiddenLayerSize, layer1Weights);
    nn::NodePtr activationNode = std::make_shared<nn::ActivationNode>(inputNode, nn::ActivationFunction::CReLU);
    nn::NodePtr hiddenNode = std::make_shared<nn::FullyConnectedNode>(activationNode, hiddenLayerSize, 1, layer2Weights);
    nn::NodePtr outputNode = std::make_shared<nn::ActivationNode>(hiddenNode, nn::ActivationFunction::Sigmoid);

    std::vector<nn::NodePtr> nodes =
    {
        inputNode,
        activationNode,
        hiddenNode,
        outputNode,
    };

    network.Init(nodes);

    nn::NeuralNetworkRunContext networkRunCtx;
    networkRunCtx.Init(network);

    nn::NeuralNetworkTrainer trainer;
    trainer.Init(network);

#ifdef USE_PACKED_NET
    nn::PackedNeuralNetwork packedNetwork;
#endif // USE_PACKED_NET

    std::vector<nn::TrainingVector> trainingSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);

    nn::TrainingVector validationVector;

    uint32_t numTrainingVectorsPassed = 0;

    const uint16_t features[] = {0, 1};

    {
        trainingSet[0].input.inputs[0].mode = nn::InputMode::SparseBinary;
        trainingSet[0].input.inputs[0].numFeatures = 0;
        trainingSet[0].output.mode = nn::OutputMode::Single;
        trainingSet[0].output.singleValue = 0.0f;

        trainingSet[1].input.inputs[0].mode = nn::InputMode::SparseBinary;
        trainingSet[1].input.inputs[0].numFeatures = 1;
        trainingSet[1].input.inputs[0].binaryFeatures = features; // 0
        trainingSet[1].output.mode = nn::OutputMode::Single;
        trainingSet[1].output.singleValue = 1.0f;

        trainingSet[2].input.inputs[0].mode = nn::InputMode::SparseBinary;
        trainingSet[2].input.inputs[0].numFeatures = 1;
        trainingSet[2].input.inputs[0].binaryFeatures = features + 1; // 1
        trainingSet[2].output.mode = nn::OutputMode::Single;
        trainingSet[2].output.singleValue = 0.0f;

        trainingSet[3].input.inputs[0].mode = nn::InputMode::SparseBinary;
        trainingSet[3].input.inputs[0].numFeatures = 2;
        trainingSet[3].input.inputs[0].binaryFeatures = features; // 0, 1
        trainingSet[3].output.mode = nn::OutputMode::Single;
        trainingSet[3].output.singleValue = 0.0f;
    }

    for (;;)
    {
        nn::TrainParams params;
        params.batchSize = cBatchSize;

        trainer.Train(network, trainingSet, params);
#ifdef USE_PACKED_NET
        network.ToPackedNetwork(packedNetwork);
#endif // USE_PACKED_NET

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;
            
        float nnMinError = std::numeric_limits<float>::max();
        float nnMaxError = 0.0f, nnErrorSum = 0.0f;

#ifdef USE_PACKED_NET
        float nnPackedQuantizationErrorSum = 0.0f;
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f, nnPackedErrorSum = 0.0f;
#endif // USE_PACKED_NET

        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            nn::InputDesc inputDesc;
            inputDesc.inputs[0].mode = nn::InputMode::SparseBinary;
            if (i == 0)
            {
                inputDesc.inputs[0].numFeatures = 0;
            }
            else if (i == 1)
            {
                inputDesc.inputs[0].numFeatures = 1;
                inputDesc.inputs[0].binaryFeatures = features; // 0
            }
            else if (i == 2)
            {
                inputDesc.inputs[0].numFeatures = 1;
                inputDesc.inputs[0].binaryFeatures = features + 1; // 1
            }
            else if (i == 3)
            {
                inputDesc.inputs[0].numFeatures = 2;
                inputDesc.inputs[0].binaryFeatures = features; // 0, 1
            }
            else
            {
                ASSERT(false);
            }

            const auto& networkOutput = network.Run(inputDesc, networkRunCtx);
            const float expectedValue = trainingSet[i].output.singleValue;
            const float nnValue = networkOutput[0];

            {
                const float error = expectedValue - nnValue;
                const float errorDiff = std::abs(error);
                nnErrorSum += error * error;
                nnMinError = std::min(nnMinError, errorDiff);
                nnMaxError = std::max(nnMaxError, errorDiff);
            }

#ifdef USE_PACKED_NET
            int32_t packedNetworkOutput = packedNetwork.Run(features.data(), (uint32_t)features.size(), 0u);
            const float nnPackedValue = (float)packedNetworkOutput / (float)nn::WeightScale / (float)nn::OutputScale;
            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            {
                const float error = expectedValue - nnPackedValue;
                const float errorDiff = std::abs(error);
                nnPackedErrorSum += error * error;
                nnPackedMinError = std::min(nnPackedMinError, errorDiff);
                nnPackedMaxError = std::max(nnPackedMaxError, errorDiff);
            }
#endif // USE_PACKED_NET

        }

        nnErrorSum = sqrtf(nnErrorSum / cNumTrainingVectorsPerIteration);
        std::cout << std::right << std::setw(6) << numTrainingVectorsPassed; std::cout << "  |  ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnMaxError; std::cout << "  |  ";

#ifdef USE_PACKED_NET
        nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumTrainingVectorsPerIteration);
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedQuantizationErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedMaxError; std::cout << "  |  ";
#endif // USE_PACKED_NET

        std::cout << std::endl;
    }

    return true;
}
