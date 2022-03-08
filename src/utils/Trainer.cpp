#include "Common.hpp"
#include "ThreadPool.hpp"

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
#include "../backend/NeuralNetwork.hpp"
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

static const uint32_t cMaxIterations = 100000000;
static const uint32_t cNumTrainingVectorsPerIteration = 100000;
static const uint32_t cBatchSize = 64;

static void PositionToPackedVector(const Position& pos, nn::TrainingVector& outVector)
{
    const uint32_t maxFeatures = 64;

    uint16_t features[maxFeatures];
    uint32_t numFeatures = pos.ToPackedFeaturesVector(features);
    ASSERT(numFeatures <= maxFeatures);

    outVector.output.resize(1);
    outVector.inputFeatures.clear();
    outVector.inputFeatures.reserve(numFeatures);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outVector.inputFeatures.push_back(features[i]);
    }
}

static float ScoreToNN(float score)
{
    return score;
}

static float ScoreFromNN(float score)
{
    return std::clamp(score, 0.0f, 1.0f);
}

bool TrainEndgame()
{
    TranspositionTable tt{ 2048ull * 1024ull * 1024ull };
    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };

    SearchParam searchParam{ tt };
    searchParam.limits.maxDepth = 10;
    searchParam.limits.maxNodes = 100000;
    searchParam.limits.analysisMode = true;
    searchParam.debugLog = false;

    struct TrainingEntry
    {
        Position pos;
        nn::TrainingVector trainingVector;
    };

    MaterialKey materialKey;
    materialKey.numWhitePawns   = 1;
    materialKey.numWhiteKnights = 0;
    materialKey.numWhiteBishops = 0;
    materialKey.numWhiteRooks   = 0;
    materialKey.numWhiteQueens  = 0;
    materialKey.numBlackPawns   = 1;
    materialKey.numBlackKnights = 0;
    materialKey.numBlackBishops = 0;
    materialKey.numBlackRooks   = 0;
    materialKey.numBlackQueens  = 0;

    std::cout << "Training network for: " << materialKey.ToString() << "..." << std::endl;


    std::random_device randomDevice;

    const auto generateTrainingSet = [&](TaskBuilder& taskBuilder, std::vector<TrainingEntry>& outSet)
    {
        taskBuilder.ParallelFor("", (uint32_t)outSet.size(), [&](const TaskContext& ctx, const uint32_t i)
        {
            std::mt19937_64 randomGenerator(randomDevice());

            Search& search = searchArray[ctx.threadId];

            for (;;)
            {
                Position pos;
                GenerateRandomPosition(randomGenerator, materialKey, pos);

                // don't generate positions where side to move is in check
                if (pos.IsInCheck(pos.GetSideToMove()))
                {
                    continue;
                }

                // generate only quiet position
                MoveList moves;
                pos.GenerateMoveList(moves, MOVE_GEN_ONLY_TACTICAL);

                if (moves.Size() > 0)
                {
                    continue;
                }

                uint32_t dtz = 0;
                int32_t wdl = 0;
                Move tbMove;
                if (!ProbeTablebase_Root(pos, tbMove, &dtz, &wdl))
                {
                    continue;
                }

                float score = 0.5f;
                if (wdl < 0) score = 0.0f;
                if (wdl > 0) score = 1.0f;

                (void)search;
/*
                Game game;
                game.Reset(pos);

                SearchResult searchResult;
                search.DoSearch(game, searchParam, searchResult);

                const bool isStalemate = pos.IsStalemate();
                if (!isStalemate)
                {
                    if (searchResult.empty())
                    {
                        std::cout << "Broken position: " << std::endl;
                        std::cout << pos.Print() << std::endl;
                    }

                    ASSERT(!searchResult.empty());
                }

                float score = PawnToWinProbability(isStalemate ? 0.0f : (float)searchResult[0].score / 100.0f);
*/

                PositionToPackedVector(pos, outSet[i].trainingVector);
                outSet[i].trainingVector.output[0] = ScoreToNN(score);
                outSet[i].pos = pos;

                break;
            }
        });
    };

    const uint32_t numNetworkInputs = materialKey.GetNeuralNetworkInputsNumber();

    nn::NeuralNetwork network;
    network.Init(numNetworkInputs, { nn::FirstLayerSize, nn::SecondLayerSize, 1 });

    nn::PackedNeuralNetwork packedNetwork;

    std::vector<TrainingEntry> trainingSet, validationSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);
    validationSet.resize(cNumTrainingVectorsPerIteration);

    nn::Layer::Values tempValues;

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
        // use validation set from previous iteration as training set in the current one
        trainingSet = validationSet;

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);

            taskBuilder.Task("Train", [&](const TaskContext&)
            {
                std::vector<nn::TrainingVector> batch(trainingSet.size());
                for (size_t i = 0; i < trainingSet.size(); ++i)
                {
                    batch[i] = trainingSet[i].trainingVector;
                }

                network.Train(batch, tempValues, cBatchSize);
                //network.PrintStats();

                network.ToPackedNetwork(packedNetwork);
                packedNetwork.Save("pawns.nn");
            });

            taskBuilder.Task("GenerateSet", [&](const TaskContext& ctx)
            {
                TaskBuilder childTaskBuilder(ctx);
                generateTrainingSet(childTaskBuilder, validationSet);
            });
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

        FILE* f = nullptr;
        if (iteration == 0)
        {
            const std::string fileName = materialKey.ToString() + ".epd";
            f = fopen(fileName.c_str(), "w");
        }

        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            if (iteration == 0)
            {
                const std::string fen = validationSet[i].pos.ToFEN();
                fwrite(fen.c_str(), fen.size(), 1, f);
                fwrite("\n", 1, 1, f);
            }

            const std::vector<uint16_t>& features = validationSet[i].trainingVector.inputFeatures;
            tempValues = network.Run(features.data(), (uint32_t)features.size());
            int32_t packedNetworkOutput = packedNetwork.Run(features.data(), (uint32_t)features.size());

            const float expectedValue = ScoreFromNN(validationSet[i].trainingVector.output[0]);
            const float nnValue = ScoreFromNN(tempValues[0]);
            const float nnPackedValue = ScoreFromNN(nn::Sigmoid((float)packedNetworkOutput / (float)nn::OutputScale));
            const float evalValue = PawnToWinProbability((float)Evaluate(validationSet[i].pos) / 100.0f);

            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            if ((expectedValue >= 0.7f && nnValue >= 0.7f) ||
                (expectedValue <= 0.3f && nnValue <= 0.3f) ||
                (expectedValue > 0.3f && expectedValue < 0.7f && nnValue > 0.3f && nnValue < 0.7f))
            {
                correctPredictions++;
            }

            //if (i < 10)
            //{
            //    std::cout
            //        << validationSet[i].pos.ToFEN()
            //        << ", True Score: " << expectedValue
            //        << ", NN eval: " << nnValue
            //        << ", Packed NN eval: " << nnPackedValue
            //        << ", Static eval: " << evalValue << std::endl;
            //}

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

        std::cout << std::right << std::setw(6) << numTrainingVectorsPassed; std::cout << "  |  ";
        std::cout << std::right << std::fixed << std::setprecision(4) << accuracy; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnMaxError; std::cout << "  |  ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedQuantizationErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << nnPackedMaxError; std::cout << "  |  ";
        std::cout << std::right << std::fixed << std::setprecision(4) << evalErrorSum; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << evalMinError; std::cout << " ";
        std::cout << std::right << std::fixed << std::setprecision(4) << evalMaxError;
        std::cout << std::endl;

        network.Save((materialKey.ToString() + ".nn").c_str());
        packedNetwork.Save((materialKey.ToString() + ".pnn").c_str());
    }

    return true;
}