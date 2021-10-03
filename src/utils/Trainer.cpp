#include "Common.hpp"

#include "../backend/Position.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/NeuralNetwork.hpp"
#include "../backend/PackedNeuralNetwork.hpp"
#include "../backend/ThreadPool.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

using namespace threadpool;

void PositionEntryToTrainingVector(const PositionEntry& entry, nn::TrainingVector& outVector)
{
    outVector.input.resize(12 * 64);
    outVector.output.resize(1);

    PositionEntry flippedEntry = entry;
    if (entry.sideToMove == 1)
    {
        flippedEntry.whiteKing = Bitboard(entry.blackKing).Rotated180();
        flippedEntry.whitePawns = Bitboard(entry.blackPawns).Rotated180();
        flippedEntry.whiteKnights = Bitboard(entry.blackKnights).Rotated180();
        flippedEntry.whiteBishops = Bitboard(entry.blackBishops).Rotated180();
        flippedEntry.whiteRooks = Bitboard(entry.blackRooks).Rotated180();
        flippedEntry.whiteQueens = Bitboard(entry.blackQueens).Rotated180();

        flippedEntry.blackKing = Bitboard(entry.whiteKing).Rotated180();
        flippedEntry.blackPawns = Bitboard(entry.whitePawns).Rotated180();
        flippedEntry.blackKnights = Bitboard(entry.whiteKnights).Rotated180();
        flippedEntry.blackBishops = Bitboard(entry.whiteBishops).Rotated180();
        flippedEntry.blackRooks = Bitboard(entry.whiteRooks).Rotated180();
        flippedEntry.blackQueens = Bitboard(entry.whiteQueens).Rotated180();

        flippedEntry.eval *= -1;
        flippedEntry.gameResult *= -1;
    }

    for (uint32_t i = 0; i < 64u; ++i)
    {
        outVector.input[ 0 * 64 + i] = (float)((flippedEntry.whiteKing >> i) & 1);
        outVector.input[ 1 * 64 + i] = (float)((flippedEntry.whitePawns >> i) & 1);
        outVector.input[ 2 * 64 + i] = (float)((flippedEntry.whiteKnights >> i) & 1);
        outVector.input[ 3 * 64 + i] = (float)((flippedEntry.whiteBishops >> i) & 1);
        outVector.input[ 4 * 64 + i] = (float)((flippedEntry.whiteRooks >> i) & 1);
        outVector.input[ 5 * 64 + i] = (float)((flippedEntry.whiteQueens >> i) & 1);

        outVector.input[ 6 * 64 + i] = (float)((flippedEntry.blackKing >> i) & 1);
        outVector.input[7 * 64 + i] = (float)((flippedEntry.blackPawns >> i) & 1);
        outVector.input[8 * 64 + i] = (float)((flippedEntry.blackKnights >> i) & 1);
        outVector.input[9 * 64 + i] = (float)((flippedEntry.blackBishops >> i) & 1);
        outVector.input[10 * 64 + i] = (float)((flippedEntry.blackRooks >> i) & 1);
        outVector.input[11 * 64 + i] = (float)((flippedEntry.blackQueens >> i) & 1);
    }

    outVector.output[0] = (float)flippedEntry.eval / 100.0f;
    outVector.output[0] = 2.0f * PawnToWinProbability(outVector.output[0]) - 1.0f;
}

static const uint32_t cMaxSearchDepth = 12;
static const uint32_t cMaxIterations = 100000000;
static const uint32_t cNumTrainingVectorsPerIteration = 1000;
static const uint32_t cBatchSize = 10;

bool Train()
{
    FILE* dumpFile = fopen("selfplay.dat", "rb");

    if (!dumpFile)
    {
        std::cout << "ERROR: Failed to load selfplay data file!" << std::endl;
        return false;
    }

    fseek(dumpFile, 0, SEEK_END);
    const size_t fileSize = ftell(dumpFile);
    fseek(dumpFile, 0, SEEK_SET);

    const size_t numEntries = fileSize / sizeof(PositionEntry);

    std::vector<PositionEntry> entries;
    entries.resize(numEntries);

    if (numEntries != fread(entries.data(), sizeof(PositionEntry), numEntries, dumpFile))
    {
        std::cout << "ERROR: Failed read selfplay data file!" << std::endl;
        fclose(dumpFile);
        return false;
    }

    std::cout << "INFO: Loaded " << numEntries << " entries" << std::endl;
    fclose(dumpFile);

    nn::NeuralNetwork network;
    network.Init(12 * 64, { 256, 32, 32, 1 });
    //if (!network.Load("network.dat")) return false;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<nn::TrainingVector> trainingSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);

    nn::TrainingVector validationVector;

    nn::Layer::Values tempValues;

    uint32_t numTrainingVectorsPassed = 0;

    for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        // pick random test entries
        std::uniform_int_distribution<size_t> distrib(0, numEntries - 1);
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];
            PositionEntryToTrainingVector(entry, trainingSet[i]);
        }
        network.Train(trainingSet, tempValues, cBatchSize);

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float minError = std::numeric_limits<float>::max();
        float maxError = -std::numeric_limits<float>::max();
        float errorSum = 0.0f;
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];
            PositionEntryToTrainingVector(entry, validationVector);

            tempValues = network.Run(validationVector.input);

            float expectedValue = validationVector.output[0];

            if (i < 10)
            {
                //std::cout << "    value= " << tempValues[0] << ", expected=" << expectedValue << std::endl;
            }

            const float error = expectedValue - tempValues[0];
            minError = std::min(minError, fabsf(error));
            maxError = std::max(maxError, fabsf(error));
            const float sqrError = error * error;
            errorSum += sqrError;
        }
        errorSum = sqrtf(errorSum / cNumTrainingVectorsPerIteration);

        float epoch = (float)numTrainingVectorsPassed / (float)numEntries;
        std::cout << epoch << "\t" << errorSum << "\t" << minError << "\t" << maxError;
        std::cout << std::endl;

        network.Save("network.dat");
    }

    return true;
}

//static const uint32_t networkInputs = 32 + 64 + 2 * 64 + 2 * 48;
static const uint32_t networkInputs = 32 + 64 + 2 * 48;

static void PositionToVector(const Position& pos, nn::TrainingVector& outVector, std::vector<uint32_t>& outFeatures)
{
    uint32_t features[32];
    uint32_t numFeatures = pos.ToFeaturesVector(features);

    outVector.input.resize(networkInputs);
    outVector.output.resize(1);
    outFeatures.clear();

    memset(outVector.input.data(), 0, sizeof(float) * networkInputs);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outFeatures.push_back(features[i]);
        outVector.input[features[i]] = 1.0f;
    }
}

bool TrainEndgame()
{
    TranspositionTable tt{ 8192ull * 1024ull * 1024ull };
    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };

    SearchParam searchParam{ tt };
    searchParam.limits.maxDepth = cMaxSearchDepth;
    searchParam.limits.analysisMode = true;
    searchParam.debugLog = false;

    const auto generateTrainingSet = [&](TaskBuilder& taskBuilder, std::vector<nn::TrainingVector>& outSet, std::vector<std::vector<uint32_t>>& outFeatures, std::vector<Position>& outPositions)
    {
        taskBuilder.ParallelFor("", (uint32_t)outSet.size(), [&](const TaskContext& context, const uint32_t i)
        {
            Search& search = searchArray[context.threadId];

            for (;;)
            {
                Position pos;

                MaterialKey material;
                material.numWhitePawns = 1;// +(rand() % 7);
                material.numBlackPawns = 1;// +(rand() % 7);

                GenerateRandomPosition(material, pos);

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

                PositionToVector(pos, outSet[i], outFeatures[i]);

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

                float score = isStalemate ? 0.0f : (float)searchResult[0].score;
                score = std::clamp(score / 100.0f, -15.0f, 15.0f);
                outSet[i].output[0] = PawnToWinProbability(score);
                outPositions[i] = pos;

                break;
            }
        });
    };

    {
        nn::NeuralNetwork network;
        network.Init(networkInputs, { nn::FirstLayerSize, nn::SecondLayerSize, 1 });

        nn::PackedNeuralNetwork packedNetwork;

        std::vector<nn::TrainingVector> trainingSet, validationSet;
        trainingSet.resize(cNumTrainingVectorsPerIteration);
        validationSet.resize(cNumTrainingVectorsPerIteration);

        std::vector<Position> trainingPositions, validationPositions;
        std::vector<std::vector<uint32_t>> trainingFeatures, validationFeatures;
        trainingPositions.resize(cNumTrainingVectorsPerIteration);
        trainingFeatures.resize(cNumTrainingVectorsPerIteration);
        validationPositions.resize(cNumTrainingVectorsPerIteration);
        validationFeatures.resize(cNumTrainingVectorsPerIteration);

        nn::TrainingVector validationVector;

        nn::Layer::Values tempValues;

        uint32_t numTrainingVectorsPassed = 0;

        {
            Waitable waitable;
            {
                TaskBuilder childTaskBuilder(waitable);
                generateTrainingSet(childTaskBuilder, validationSet, validationFeatures, validationPositions);
            }
            waitable.Wait();
        }

        for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
        {
            // use validation set from previous iteration as training set in the current one
            trainingSet = validationSet;
            trainingFeatures = validationFeatures;
            trainingPositions = validationPositions;

            // validation vectors generation can be done in parallel with training
            Waitable waitable;
            {
                TaskBuilder taskBuilder(waitable);

                taskBuilder.Task("Train", [&](const TaskContext&)
                {
                    network.Train(trainingSet, tempValues, cBatchSize);
                    network.PrintStats();

                    network.ToPackedNetwork(packedNetwork);
                    packedNetwork.Save("pawns.nn");
                });

                taskBuilder.Task("GenerateSet", [&](const TaskContext& ctx)
                {
                    TaskBuilder childTaskBuilder(ctx);
                    generateTrainingSet(childTaskBuilder, validationSet, validationFeatures, validationPositions);
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

            for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
            {
                tempValues = network.Run(validationSet[i].input);
                int32_t packedNetworkOutput = packedNetwork.Run((uint32_t)validationFeatures[i].size(), validationFeatures[i].data());

                const float expectedValue = validationSet[i].output[0];
                const float nnValue = tempValues[0];
                const float nnPackedValue = (float)packedNetworkOutput / (float)nn::WeightScale / (float)nn::OutputScale;
                const float evalValue = PawnToWinProbability((float)Evaluate(validationPositions[i]) / 100.0f);

                nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

                if (i < 10)
                {
                    std::cout << validationPositions[i].ToFEN() << std::endl;
                    std::cout << " Score:            " << expectedValue << std::endl;
                    std::cout << " NN eval:          " << nnValue << std::endl;
                    std::cout << " Packed NN eval:   " << nnPackedValue << std::endl;
                    std::cout << " Static eval:      " << evalValue << std::endl;
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

            nnErrorSum = sqrtf(nnErrorSum / cNumTrainingVectorsPerIteration);
            nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumTrainingVectorsPerIteration);
            evalErrorSum = sqrtf(evalErrorSum / cNumTrainingVectorsPerIteration);
            nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumTrainingVectorsPerIteration);

            std::cout << std::right << std::setw(6) << numTrainingVectorsPassed; std::cout << "  |  ";
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

            if (iteration % 10 == 0)
            {
                const auto ttUsage = tt.GetNumUsedEntries();
                std::cout << "transposition table usage: " << ttUsage << " entries (" <<
                    std::fixed << std::setprecision(4) << (float)ttUsage / (float)tt.GetSize() * 100.0f << "%)" << std::endl;
            }
        }

        network.Save("network.dat");
    }

    return true;
}
