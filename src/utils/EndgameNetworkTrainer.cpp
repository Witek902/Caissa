#include "Common.hpp"
#include "ThreadPool.hpp"
#include "NeuralNetwork.hpp"

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
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

using namespace threadpool;

static const uint32_t cMaxIterations = 10000000;
static const uint32_t cNumTrainingVectorsPerIteration = 4 * 1024;
static const uint32_t cBatchSize = 128;

static void PositionToPackedVector(const Position& pos, nn::TrainingVector& outVector)
{
    const uint32_t maxFeatures = 64;

    uint16_t features[maxFeatures];
    //uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::MaterialPacked_KingPiece_Symmetrical);
    //uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::MaterialPacked_Symmetrical);
    uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::Full_Symmetrical);
    ASSERT(numFeatures <= maxFeatures);

    outVector.output.resize(1);
    outVector.inputMode = nn::InputMode::SparseBinary;
    outVector.sparseBinaryInputs.clear();
    outVector.sparseBinaryInputs.reserve(numFeatures);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outVector.sparseBinaryInputs.emplace_back(features[i]);
    }
}

static float ScoreFromNN(float score)
{
    return std::clamp(score, 0.0f, 1.0f);
}

bool TrainEndgame()
{
    struct TrainingEntry
    {
        Position pos;
        nn::TrainingVector trainingVector;
    };

    const auto generateTrainingSet = [&](TaskBuilder& taskBuilder, std::vector<TrainingEntry>& outSet)
    {
        taskBuilder.ParallelFor("", (uint32_t)outSet.size(), [&](const TaskContext&, const uint32_t i)
        {
            const uint32_t longestDTM = 253; // for 5 pieces

            std::random_device randomDevice;
            std::mt19937 randomGenerator(randomDevice());
            std::uniform_int_distribution<uint32_t> pieceIndexDistr(0, 9);
            std::uniform_int_distribution<uint32_t> scoreDistr(0, 18);
            std::uniform_int_distribution<uint32_t> numPiecesDistr(0, 63);

            for (;;)
            {
                MaterialKey materialKey;

                const uint32_t numPieces = numPiecesDistr(randomGenerator) == 0 ? 4 : 5;

                for (uint32_t j = 0; j < numPieces - 2; ++j)
                {
                    const uint32_t pieceIndex = pieceIndexDistr(randomGenerator);
                    ASSERT(pieceIndex < 10);
                    switch (pieceIndex)
                    {
                    case 0: materialKey.numWhitePawns++; break;
                    case 1: materialKey.numWhiteKnights++; break;
                    case 2: materialKey.numWhiteBishops++; break;
                    case 3: materialKey.numWhiteRooks++; break;
                    case 4: materialKey.numWhiteQueens++; break;
                    case 5: materialKey.numBlackPawns++; break;
                    case 6: materialKey.numBlackKnights++; break;
                    case 7: materialKey.numBlackBishops++; break;
                    case 8: materialKey.numBlackRooks++; break;
                    case 9: materialKey.numBlackQueens++; break;
                    }
                }

                // generate unbalanced positions with lower probability
                const int64_t whitesScore = materialKey.numWhitePawns + 3 * materialKey.numWhiteKnights + 3 * materialKey.numWhiteBishops + 5 * materialKey.numWhiteRooks + 9 * materialKey.numWhiteQueens;
                const int64_t blacksScore = materialKey.numBlackPawns + 3 * materialKey.numBlackKnights + 3 * materialKey.numBlackBishops + 5 * materialKey.numBlackRooks + 9 * materialKey.numBlackQueens;
                const int64_t scoreDiff = std::abs(whitesScore - blacksScore);
                if (scoreDiff > 2)
                {
                    if (scoreDistr(randomGenerator) < scoreDiff) continue;
                }

                Position pos;
                GenerateRandomPosition(randomGenerator, materialKey, pos);

                // generate only quiet position
                if (!pos.IsQuiet() || pos.GetNumLegalMoves() == 0)
                {
                    continue;
                }

                int32_t wdl = 0;
                if (!ProbeSyzygy_WDL(pos, &wdl))
                {
                    continue;
                }

                float score = 0.5f;
                if (wdl != 0)
                {
                    uint32_t gaviotaDTM = 0;
                    int32_t gaviotaWDL = 0;

                    if (ProbeGaviota(pos, &gaviotaDTM, &gaviotaWDL))
                    {
                        ASSERT(gaviotaDTM > 0);
                        ASSERT(wdl == gaviotaWDL);

                        const float power = 1.5f;
                        const float scale = 0.45f;
                        const float offset = 0.001f;

                        if (wdl < 0) score = offset + scale * powf((float)(gaviotaDTM - 1) / (float)longestDTM, power);
                        if (wdl > 0) score = 1.0f - offset - scale * powf((float)(gaviotaDTM - 1) / (float)longestDTM, power);
                    }
                    else
                    {
                        if (wdl < 0) score = 0.0f;
                        if (wdl > 0) score = 1.0f;
                    }
                }

                PositionToPackedVector(pos, outSet[i].trainingVector);
                outSet[i].trainingVector.output[0] = score;
                outSet[i].pos = pos;

                break;
            }
        });
    };

    const uint32_t numNetworkInputs = 704;
    //const uint32_t numNetworkInputs = materialKey.GetNeuralNetworkInputsNumber();
    //const uint32_t numNetworkInputs = 2 * 3 * 32 * 64;

    std::string name = "endgame_5_4";

    nn::NeuralNetwork network;
    network.Init(numNetworkInputs, { 256, 32, 32, 1 });
    network.Load((name + ".nn").c_str());

    nn::NeuralNetworkRunContext networkRunCtx;
    networkRunCtx.Init(network);

    nn::NeuralNetworkTrainer trainer;
    nn::PackedNeuralNetwork packedNetwork;

    std::vector<TrainingEntry> trainingSet, validationSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);
    validationSet.resize(cNumTrainingVectorsPerIteration);

    std::vector<int32_t> packedNetworkOutputs(cNumTrainingVectorsPerIteration);

    uint32_t numTrainingVectorsPassed = 0;

    {
        Waitable waitable;
        {
            TaskBuilder childTaskBuilder(waitable);
            generateTrainingSet(childTaskBuilder, validationSet);
        }
        waitable.Wait();
    }

    for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        float learningRate = std::max(0.05f, 1.0f / (1.0f + 0.0002f * iteration));

        // use validation set from previous iteration as training set in the current one
        trainingSet = validationSet;

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);
            taskBuilder.Task("GenerateSet", [&](const TaskContext& ctx)
            {
                TaskBuilder childTaskBuilder(ctx);
                generateTrainingSet(childTaskBuilder, validationSet);
            });
        }

        float trainingTime = 0.0f;
        {
            std::vector<nn::TrainingVector> batch(trainingSet.size());
            for (size_t i = 0; i < trainingSet.size(); ++i)
            {
                batch[i] = trainingSet[i].trainingVector;
            }

            TimePoint startTime = TimePoint::GetCurrent();
            trainer.Train(network, batch, cBatchSize, learningRate);
            trainingTime = (TimePoint::GetCurrent() - startTime).ToSeconds();

            network.ToPackedNetwork(packedNetwork);
        }
        waitable.Wait();

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float nnMinError = std::numeric_limits<float>::max();
        float nnMaxError = 0.0f, nnErrorSum = 0.0f;

        float nnPackedQuantizationErrorSum = 0.0f;
        float nnPackedMinError = std::numeric_limits<float>::max();
        float nnPackedMaxError = 0.0f, nnPackedErrorSum = 0.0f;

        float evalMinError = std::numeric_limits<float>::max();
        float evalMaxError = 0.0f, evalErrorSum = 0.0f;

        uint32_t correctPredictions = 0;

        float packedNetworkRunTime = 0.0f;
        {
            TimePoint startTime = TimePoint::GetCurrent();
            for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
            {
                const std::vector<uint16_t>& features = validationSet[i].trainingVector.sparseBinaryInputs;
                packedNetworkOutputs[i] = packedNetwork.Run(features.data(), (uint32_t)features.size());
            }
            packedNetworkRunTime = (TimePoint::GetCurrent() - startTime).ToSeconds();
        }

        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const std::vector<uint16_t>& features = validationSet[i].trainingVector.sparseBinaryInputs;
            const nn::Values& networkOutput = network.Run((uint32_t)features.size(), features.data(), networkRunCtx);
            const int32_t packedNetworkOutput = packedNetworkOutputs[i];

            const float expectedValue = ScoreFromNN(validationSet[i].trainingVector.output[0]);
            const float nnValue = ScoreFromNN(networkOutput[0]);
            const float nnPackedValue = ScoreFromNN(nn::Sigmoid((float)packedNetworkOutput / (float)nn::OutputScale));
            const float evalValue = PawnToWinProbability((float)Evaluate(validationSet[i].pos) / 100.0f);

            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            if ((expectedValue >= 2.0f/3.0f && nnValue >= 2.0f/3.0f) ||
                (expectedValue <= 1.0f/3.0f && nnValue <= 1.0f/3.0f) ||
                (expectedValue > 1.0f/3.0f && expectedValue < 2.0f/3.0f && nnValue > 1.0f/3.0f && nnValue < 2.0f/3.0f))
            {
                correctPredictions++;
            }

            if (i + 1 == cNumTrainingVectorsPerIteration)
            {
                std::cout
                    << validationSet[i].pos.ToFEN() << std::endl << validationSet[i].pos.Print() << std::endl
                    << "True Score:     " << expectedValue << " (" << WinProbabilityToCentiPawns(expectedValue) << ")" << std::endl
                    << "NN eval:        " << nnValue << " (" << WinProbabilityToCentiPawns(nnValue) << ")" << std::endl
                    << "Packed NN eval: " << nnPackedValue << " (" << WinProbabilityToCentiPawns(nnPackedValue) << ")" << std::endl
                    << "Static eval:    " << evalValue << " (" << WinProbabilityToCentiPawns(evalValue) << ")" << std::endl
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

        const float accuracy = (float)correctPredictions / (float)cNumTrainingVectorsPerIteration;
        nnErrorSum = sqrtf(nnErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumTrainingVectorsPerIteration);
        evalErrorSum = sqrtf(evalErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumTrainingVectorsPerIteration);

        std::cout
            << "Epoch:                  " << iteration << std::endl
            << "Num training vectors:   " << numTrainingVectorsPassed << std::endl
            << "Learning rate:          " << learningRate << std::endl
            << "Accuracy:               " << (100.0f * accuracy) << "%" << std::endl
            << "NN avg/min/max error:   " << std::setprecision(5) << nnErrorSum << " " << std::setprecision(4) << nnMinError << " " << std::setprecision(4) << nnMaxError << std::endl
            << "PNN avg/min/max error:  " << std::setprecision(5) << nnPackedErrorSum << " " << std::setprecision(4) << nnPackedMinError << " " << std::setprecision(4) << nnPackedMaxError << std::endl
            << "Quantization error:     " << std::setprecision(5) << nnPackedQuantizationErrorSum << std::endl
            << "Eval avg/min/max error: " << std::setprecision(5) << evalErrorSum << " " << std::setprecision(4) << evalMinError << " " << std::setprecision(4) << evalMaxError << std::endl;

        network.PrintStats();

        std::cout << "Training time:    " << trainingTime << " sec" << std::endl;
        std::cout << "Network run time: " << 1000.0f * packedNetworkRunTime << " ms" << std::endl << std::endl;

        if (iteration % 10 == 0)
        {
            network.Save((name + ".nn").c_str());
            packedNetwork.Save((name + ".pnn").c_str());
            packedNetwork.SaveAsImage((name + ".raw").c_str());
        }
    }

    return true;
}