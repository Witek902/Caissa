#include "Common.hpp"

#include "../backend/NeuralNetwork.hpp"
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
    network.Init(2, { nn::FirstLayerSize, nn::SecondLayerSize, 1 });

    nn::PackedNeuralNetwork packedNetwork;

    std::vector<nn::TrainingVector> trainingSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);

    nn::TrainingVector validationVector;
    nn::Layer::Values tempValues;

    uint32_t numTrainingVectorsPassed = 0;

    {
        trainingSet[0].inputFeatures = {};
        trainingSet[0].output.resize(1);
        trainingSet[0].output[0] = 0.0f;

        trainingSet[1].output.resize(1);
        trainingSet[1].inputFeatures = { 0 };
        trainingSet[1].output[0] = 1.0f;

        trainingSet[2].output.resize(1);
        trainingSet[2].inputFeatures = { 1 };
        trainingSet[2].output[0] = 0.0f;

        trainingSet[3].output.resize(1);
        trainingSet[3].inputFeatures = { 0, 1 };
        trainingSet[3].output[0] = 0.0f;
    }

    for (uint32_t iteration = 0; ; ++iteration)
    {
        network.Train(trainingSet, tempValues, cBatchSize);
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
            uint32_t numFeatures = 0;
            uint16_t featureIndices[2] = { 0, 0 };

            if (i == 1)
            {
                numFeatures = 1;
                featureIndices[0] = 0;
            }
            else if (i == 2)
            {
                numFeatures = 1;
                featureIndices[0] = 1;
            }
            else if (i == 3)
            {
                numFeatures = 2;
                featureIndices[0] = 0;
                featureIndices[1] = 1;
            }

            tempValues = network.Run(featureIndices, numFeatures);
            int32_t packedNetworkOutput = packedNetwork.Run(featureIndices, numFeatures);

            const float expectedValue = trainingSet[i].output[0];
            const float nnValue = tempValues[0];
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
