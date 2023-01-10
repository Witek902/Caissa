#include "Common.hpp"
#include "NeuralNetwork.hpp"
#include "../backend/PackedNeuralNetwork.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

static const uint32_t cNumTrainingVectorsPerIteration = 4;
static const uint32_t cBatchSize = 10;

bool TestNetwork()
{
    nn::NeuralNetwork network;
    network.Init(2, { 64, 16, 1 });

    nn::NeuralNetworkRunContext networkRunCtx;
    networkRunCtx.Init(network);

    nn::NeuralNetworkTrainer trainer;

    nn::PackedNeuralNetwork packedNetwork;

    std::vector<nn::TrainingVector> trainingSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);

    nn::TrainingVector validationVector;

    uint32_t numTrainingVectorsPassed = 0;

    {
        trainingSet[0].inputMode = nn::InputMode::SparseBinary;
        trainingSet[0].sparseBinaryInputs = {};
        trainingSet[0].singleOutput = 0.0f;

        trainingSet[1].inputMode = nn::InputMode::SparseBinary;
        trainingSet[1].sparseBinaryInputs = { 0 };
        trainingSet[1].singleOutput = 1.0f;

        trainingSet[2].inputMode = nn::InputMode::SparseBinary;
        trainingSet[2].sparseBinaryInputs = { 1 };
        trainingSet[2].singleOutput = 0.0f;

        trainingSet[3].inputMode = nn::InputMode::SparseBinary;
        trainingSet[3].sparseBinaryInputs = { 0, 1 };
        trainingSet[3].singleOutput = 0.0f;
    }

    for (uint32_t iteration = 0; ; ++iteration)
    {
		nn::TrainParams params;
		params.batchSize = cBatchSize;

        trainer.Train(network, trainingSet, params);
        //network.PrintStats();
        network.ToPackedNetwork(packedNetwork);

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;
            
        float nnMinError = std::numeric_limits<float>::max();
        float nnMaxError = 0.0f, nnErrorSum = 0.0f;

        float nnPackedQuantizationErrorSum = 0.0f;
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f, nnPackedErrorSum = 0.0f;

        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            std::vector<uint16_t> features;

            if (i == 1)
            {
                features.push_back(0);
            }
            else if (i == 2)
            {
                features.push_back(1);
            }
            else if (i == 3)
            {
                features.push_back(0);
                features.push_back(1);
            }

            const auto& networkOutput = network.Run(nn::NeuralNetwork::InputDesc(features), networkRunCtx);
            int32_t packedNetworkOutput = packedNetwork.Run(features.data(), (uint32_t)features.size(), 0u);

            const float expectedValue = trainingSet[i].singleOutput;
            const float nnValue = networkOutput[0];
            const float nnPackedValue = (float)packedNetworkOutput / (float)nn::WeightScale / (float)nn::OutputScale;
            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            {
                const float error = expectedValue - nnValue;
                const float errorDiff = std::abs(error);
                nnErrorSum += error * error;
                nnMinError = std::min(nnMinError, errorDiff);
                nnMaxError = std::max(nnMaxError, errorDiff);
            }

            {
                const float error = expectedValue - nnPackedValue;
                const float errorDiff = std::abs(error);
                nnPackedErrorSum += error * error;
                nnPackedMinError = std::min(nnPackedMinError, errorDiff);
                nnPackedMaxError = std::max(nnPackedMaxError, errorDiff);
            }
        }

        nnErrorSum = sqrtf(nnErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumTrainingVectorsPerIteration);

        std::cout << std::right << std::setw(6) << numTrainingVectorsPassed; std::cout << "  |  ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnMaxError; std::cout << "  |  ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedQuantizationErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedMaxError; std::cout << "  |  ";
        std::cout << std::endl;
    }

    return true;
}
