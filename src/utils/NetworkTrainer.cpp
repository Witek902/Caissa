#include "Common.hpp"
#include "ThreadPool.hpp"
#include "TrainerCommon.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Material.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/Waitable.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

using namespace threadpool;

static const uint32_t cMaxIterations = 10000000;
static const uint32_t cNumTrainingVectorsPerIteration = 32 * 1024;
static const uint32_t cNumValidationVectorsPerIteration = 32 * 1024;
static const uint32_t cBatchSize = 128;
//static const uint32_t cNumNetworkInputs = 2 * 10 * 32 * 64;
static const uint32_t cNumNetworkInputs = 704;


static void PositionToSparseVector(const Position& pos, nn::TrainingVector& outVector)
{
    const uint32_t maxFeatures = 124;

    uint16_t features[maxFeatures];
    uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::Full_Symmetrical);
    ASSERT(numFeatures <= maxFeatures);

    outVector.output.resize(1);
    outVector.features.clear();
    outVector.features.reserve(numFeatures);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outVector.features.push_back(features[i]);
    }
}



struct PerThreadData
{
    nn::NeuralNetwork network;
    nn::NeuralNetworkRunContext runCtx;
    nn::NeuralNetworkTrainer trainer;
    nn::PackedNeuralNetwork packedNet;
};


bool TrainNetwork()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    nn::NeuralNetwork checkpointNetwork;
    checkpointNetwork.Load("eval_fullSymmetrical.nn");

    const uint32_t numNetworks = 16;

    std::vector<PerThreadData> networksData;
    networksData.resize(numNetworks);

    for (uint32_t i = 0; i < numNetworks; ++i)
    {
        networksData[i].network.Init(cNumNetworkInputs, { nn::FirstLayerSize, 32, 32, 1 }, nn::ActivationFunction::Sigmoid);
        networksData[i].runCtx.Init(networksData[i].network);
    }

    std::vector<PositionEntry> entries;
    LoadAllPositions(entries);

    std::cout << "Training with " << entries.size() << " positions" << std::endl;

    std::vector<TrainingEntry> trainingSet, validationSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);
    validationSet.resize(cNumTrainingVectorsPerIteration);

    std::vector<int32_t> packedNetworkOutputs(cNumValidationVectorsPerIteration);

    uint32_t numTrainingVectorsPassed = 0;

    const auto generateTrainingSet = [&](std::vector<TrainingEntry>& outEntries)
    {
        // pick random test entries
        std::uniform_int_distribution<size_t> distrib(0, entries.size() - 1);
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];
            Position pos;
            VERIFY(UnpackPosition(entry.pos, pos));
            ASSERT(pos.IsValid());

            // flip the board randomly in pawnless positions
            if (pos.Whites().pawns == 0 && pos.Blacks().pawns == 0)
            {
                if (std::uniform_int_distribution<>(0, 1)(gen) != 0)
                {
                    pos.MirrorVertically();
                }
            }

            PositionToSparseVector(pos, outEntries[i].trainingVector);
            outEntries[i].trainingVector.output[0] = entry.score;
            outEntries[i].pos = pos;
        }
    };

    generateTrainingSet(validationSet);

    for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        float learningRate = std::max(0.05f, 1.0f / (1.0f + 0.0001f * iteration));

        std::mutex mutex;
        float trainingTime = 0.0f;

        // use validation set from previous iteration as training set in the current one
        trainingSet = validationSet;

        std::vector<nn::TrainingVector> batch(trainingSet.size());
        for (size_t i = 0; i < trainingSet.size(); ++i)
        {
            batch[i] = trainingSet[i].trainingVector;
        }

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);
            taskBuilder.Task("GenerateSet", [&](const TaskContext&)
            {
                generateTrainingSet(validationSet);
            });

            for (uint32_t idx = 0; idx < numNetworks; ++idx)
            {
                taskBuilder.Task("Train", [&, idx=idx](const TaskContext&)
                {
                    PerThreadData& data = networksData[idx];

                    TimePoint startTime = TimePoint::GetCurrent();
                    data.trainer.Train(data.network, batch, cBatchSize, learningRate);
                    data.network.QuantizeWeights();
                    TimePoint endTime = TimePoint::GetCurrent();

                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        trainingTime += (endTime - startTime).ToSeconds();
                    }

                    data.network.ToPackedNetwork(data.packedNet);
                });
            }
        }
        waitable.Wait();
        ASSERT(networksData.front().packedNet.IsValid());

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float nnMinError = std::numeric_limits<float>::max();
        float nnMaxError = 0.0f, nnErrorSum = 0.0f;

        float nnPackedQuantizationErrorSum = 0.0f;
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f, nnPackedErrorSum = 0.0f;

        float evalMinError = std::numeric_limits<float>::max();
        float evalMaxError = 0.0f, evalErrorSum = 0.0f;

        PerThreadData& netData = networksData.front();

        float packedNetworkRunTime = 0.0f;
        {
            TimePoint startTime = TimePoint::GetCurrent();
            for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
            {
                const std::vector<uint16_t>& features = validationSet[i].trainingVector.features;
                packedNetworkOutputs[i] = netData.packedNet.Run(features.data(), (uint32_t)features.size());
            }
            packedNetworkRunTime = (TimePoint::GetCurrent() - startTime).ToSeconds();
        }

        // TODO: parallel
        for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
        {
            const std::vector<uint16_t>& features = validationSet[i].trainingVector.features;
            const nn::Values& networkOutput = netData.network.Run(features.data(), (uint32_t)features.size(), netData.runCtx);

            const float expectedValue = validationSet[i].trainingVector.output[0];
            const float nnValue = networkOutput[0];
            const float nnPackedValue = nn::Sigmoid((float)packedNetworkOutputs[i] / (float)nn::OutputScale);
            const float evalValue = PawnToWinProbability((float)Evaluate(validationSet[i].pos) / 100.0f);

            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            if (i + 1 == cNumValidationVectorsPerIteration)
            {
                std::cout
                    << validationSet[i].pos.ToFEN() << std::endl << validationSet[i].pos.Print() << std::endl
                    << "True Score:     " << expectedValue << std::endl
                    << "NN eval:        " << nnValue << " (" << WinProbabilityToCentiPawns(nnValue) << ")" << std::endl
                    << "Packed NN eval: " << nnPackedValue << " (" << WinProbabilityToCentiPawns(nnPackedValue) << ")" << std::endl
                    << "Static eval:    " << evalValue << std::endl
                    << std::endl;
            }

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

            {
                const float error = expectedValue - evalValue;
                const float errorDiff = std::abs(error);
                evalErrorSum += error * error;
                evalMinError = std::min(evalMinError, errorDiff);
                evalMaxError = std::max(evalMaxError, errorDiff);
            }
        }

        float startPosEvaluation;
        {
            Position pos(Position::InitPositionFEN);
            nn::TrainingVector vec;
            PositionToSparseVector(pos, vec);
            startPosEvaluation = netData.network.Run(vec.features.data(), (uint32_t)vec.features.size(), netData.runCtx)[0];
        }

        nnErrorSum = sqrtf(nnErrorSum / cNumValidationVectorsPerIteration);
        nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumValidationVectorsPerIteration);
        evalErrorSum = sqrtf(evalErrorSum / cNumValidationVectorsPerIteration);
        nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumValidationVectorsPerIteration);

        std::cout
            << "Epoch:                  " << iteration << std::endl
            << "Num training vectors:   " << numTrainingVectorsPassed << std::endl
            << "Learning rate:          " << learningRate << std::endl
            << "NN avg/min/max error:   " << std::setprecision(5) << nnErrorSum << " " << std::setprecision(4) << nnMinError << " " << std::setprecision(4) << nnMaxError << std::endl
            << "PNN avg/min/max error:  " << std::setprecision(5) << nnPackedErrorSum << " " << std::setprecision(4) << nnPackedMinError << " " << std::setprecision(4) << nnPackedMaxError << std::endl
            << "Quantization error:     " << std::setprecision(5) << nnPackedQuantizationErrorSum << std::endl
            << "Eval avg/min/max error: " << std::setprecision(5) << evalErrorSum << " " << std::setprecision(4) << evalMinError << " " << std::setprecision(4) << evalMaxError << std::endl
            << "Start pos evaluation:   " << WinProbabilityToCentiPawns(startPosEvaluation) << std::endl;

        netData.network.PrintStats();

        std::cout << "Training time:    " << (1000.0f * trainingTime / numNetworks / trainingSet.size()) << " ms/pos" << std::endl;
        std::cout << "Network run time: " << 1000.0f * packedNetworkRunTime << " ms" << std::endl << std::endl;

        if (iteration % 10 == 0)
        {
            for (uint32_t i = 0; i < numNetworks; ++i)
            {
                const std::string name = "eval-" + std::to_string(i);
                networksData[i].network.Save((name + ".nn").c_str());
                networksData[i].packedNet.Save((name + ".pnn").c_str());
                networksData[i].packedNet.SaveAsImage((name + ".raw").c_str());
            }
        }
    }

    return true;
}
