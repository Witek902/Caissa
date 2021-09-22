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
#include "../backend/ThreadPool.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

using namespace threadpool;

static float PawnToWinProbability(float cp)
{
    return 1.0f / (1.0f + powf(10.0, -cp / 4.0f));
}

uint64_t RotateBitboard180(uint64_t x)
{
    const uint64_t h1 = 0x5555555555555555ull;
    const uint64_t h2 = 0x3333333333333333ull;
    const uint64_t h4 = 0x0F0F0F0F0F0F0F0Full;
    const uint64_t v1 = 0x00FF00FF00FF00FFull;
    const uint64_t v2 = 0x0000FFFF0000FFFFull;
    x = ((x >> 1) & h1) | ((x & h1) << 1);
    x = ((x >> 2) & h2) | ((x & h2) << 2);
    x = ((x >> 4) & h4) | ((x & h4) << 4);
    x = ((x >> 8) & v1) | ((x & v1) << 8);
    x = ((x >> 16) & v2) | ((x & v2) << 16);
    x = (x >> 32) | (x << 32);
    return x;
}

void PositionEntryToTrainingVector(const PositionEntry& entry, nn::TrainingVector& outVector)
{
    outVector.input.resize(12 * 64);
    outVector.output.resize(1);

    PositionEntry flippedEntry = entry;
    if (entry.sideToMove == 1)
    {
        flippedEntry.whiteKing = RotateBitboard180(entry.blackKing);
        flippedEntry.whitePawns = RotateBitboard180(entry.blackPawns);
        flippedEntry.whiteKnights = RotateBitboard180(entry.blackKnights);
        flippedEntry.whiteBishops = RotateBitboard180(entry.blackBishops);
        flippedEntry.whiteRooks = RotateBitboard180(entry.blackRooks);
        flippedEntry.whiteQueens = RotateBitboard180(entry.blackQueens);

        flippedEntry.blackKing = RotateBitboard180(entry.whiteKing);
        flippedEntry.blackPawns = RotateBitboard180(entry.whitePawns);
        flippedEntry.blackKnights = RotateBitboard180(entry.whiteKnights);
        flippedEntry.blackBishops = RotateBitboard180(entry.whiteBishops);
        flippedEntry.blackRooks = RotateBitboard180(entry.whiteRooks);
        flippedEntry.blackQueens = RotateBitboard180(entry.whiteQueens);

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

static const uint32_t cMaxIterations = 100000000;
static const uint32_t cNumTrainingVectorsPerIteration = 1000;
static const uint32_t cNumValidationVectorsPerIteration = 1000;
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
    uint32_t numTrainingVectorsPassedInEpoch = 0;

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
        numTrainingVectorsPassedInEpoch += cNumTrainingVectorsPerIteration;

        if (numTrainingVectorsPassedInEpoch > numEntries)
        {
            numTrainingVectorsPassedInEpoch = 0;
        }

        float minError = std::numeric_limits<float>::max();
        float maxError = -std::numeric_limits<float>::max();
        float errorSum = 0.0f;
        for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
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
        errorSum = sqrtf(errorSum / cNumValidationVectorsPerIteration);

        float epoch = (float)numTrainingVectorsPassed / (float)numEntries;
        std::cout << epoch << "\t" << errorSum << "\t" << minError << "\t" << maxError;
        std::cout << std::endl;

        network.Save("network.dat");
    }

    return true;
}

static const uint32_t networkInputs = 32 + 64 + 2 * 64 + 2 * 48;

static uint32_t ComputeNetworkInputs(const MaterialKey& key)
{
    uint32_t inputs = 0;

    if (key.numWhitePawns > 0 || key.numBlackPawns > 0)
    {
        // has pawns, so can't exploit vertical/diagonal symmetry
        inputs += 32; // white king on left files
        inputs += 64; // black king on any file
    }
    else
    {
        // pawnless position, can exploit vertical/horizonal/diagonal symmetry
        inputs += 16; // white king on files A-D, ranks 1-4
        inputs += 36; // black king on bottom-right triangle (a1, b1, b2, c1, c2, c3, ...)
    }

    // knights/bishops/rooks/queens on any square
    if (key.numWhiteQueens)     inputs += 64;
    if (key.numBlackQueens)     inputs += 64;
    if (key.numWhiteRooks)      inputs += 64;
    if (key.numBlackRooks)      inputs += 64;
    if (key.numWhiteBishops)    inputs += 64;
    if (key.numBlackBishops)    inputs += 64;
    if (key.numWhiteKnights)    inputs += 64;
    if (key.numBlackKnights)    inputs += 64;

    // pawns on ranks 2-7
    if (key.numWhitePawns)      inputs += 48;
    if (key.numBlackPawns)      inputs += 48;

    return inputs;
}

static void PositionToVector_KPvKP(const Position& pos, nn::TrainingVector& outVector)
{
    Square whiteKingSquare = Square(FirstBitSet(pos.Whites().king));
    Square blackKingSquare = Square(FirstBitSet(pos.Blacks().king));

    Bitboard whiteRooks = pos.Whites().rooks;
    Bitboard blackRooks = pos.Blacks().rooks;
    Bitboard whitePawns = pos.Whites().pawns;
    Bitboard blackPawns = pos.Blacks().pawns;

    if (whiteKingSquare.File() >= 4)
    {
        whiteKingSquare = whiteKingSquare.FlippedFile();
        blackKingSquare = blackKingSquare.FlippedFile();
        whiteRooks = whiteRooks.MirroredHorizontally();
        blackPawns = blackPawns.MirroredHorizontally();
        whitePawns = whitePawns.MirroredHorizontally();
        blackPawns = blackPawns.MirroredHorizontally();
    }

    outVector.input.resize(networkInputs);
    outVector.output.resize(1);

    memset(outVector.input.data(), 0, sizeof(float) * networkInputs);

    uint32_t inputOffset = 0;

    // white king
    {
        const uint32_t whiteKingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
        outVector.input[whiteKingIndex] = 1.0f;
        inputOffset += 32;
    }

    // black king
    {
        outVector.input[inputOffset + blackKingSquare.Index()] = 1.0f;
        inputOffset += 64;
    }

    if (whiteRooks)
    {
        for (uint32_t i = 0; i < 64u; ++i)
        {
            outVector.input[inputOffset + i] = (float)((whiteRooks >> i) & 1);
        }
        inputOffset += 64;
    }

    if (blackRooks)
    {
        for (uint32_t i = 0; i < 64u; ++i)
        {
            outVector.input[inputOffset + i] = (float)((blackRooks >> i) & 1);
        }
        inputOffset += 64;
    }

    if (whitePawns)
    {
        for (uint32_t i = 0; i < 48u; ++i)
        {
            const uint32_t squreIndex = i + 8u;
            outVector.input[inputOffset + i] = (float)((whitePawns >> squreIndex) & 1);
        }
        inputOffset += 48;
    }

    if (blackPawns)
    {
        for (uint32_t i = 0; i < 48u; ++i)
        {
            const uint32_t squreIndex = i + 8u;
            outVector.input[inputOffset + i] = (float)((blackPawns >> squreIndex) & 1);
        }
        inputOffset += 48;
    }

    ASSERT(inputOffset == outVector.input.size());
}

bool TrainEndgame()
{
    TranspositionTable tt{ 128 * 1024 * 1024 };
    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };

    SearchParam searchParam{ tt };
    searchParam.limits.maxDepth = 10;
    searchParam.debugLog = false;

    const auto generateTrainingSet = [&](std::vector<nn::TrainingVector>& outSet, std::vector<Position>& outPositions)
    {
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);
            taskBuilder.ParallelFor("", (uint32_t)outSet.size(), [&](const TaskContext& context, const uint32_t i)
            {
                Search& search = searchArray[context.threadId];

                for (;;)
                {
                    Position pos;

                    MaterialKey material;
                    material.numWhitePawns = 3;
                    material.numBlackPawns = 3;
                    material.numWhiteRooks = 1;
                    material.numBlackRooks = 1;

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

                    PositionToVector_KPvKP(pos, outSet[i]);

                    Game game;
                    game.Reset(pos);

                    SearchResult searchResult;
                    search.DoSearch(game, searchParam, searchResult);

                    float score = searchResult.empty() ? 0.0f : (float)searchResult[0].score;
                    score = std::clamp(score / 100.0f, -10.0f, 10.0f);
                    outSet[i].output[0] = PawnToWinProbability(score);
                    outPositions[i] = pos;

                    break;
                }
            });
        }
        waitable.Wait();
    };

    {
        nn::NeuralNetwork network;
        network.Init(networkInputs, { 64, 32, 1 });

        std::vector<nn::TrainingVector> trainingSet, validationSet;
        trainingSet.resize(cNumTrainingVectorsPerIteration);
        validationSet.resize(cNumValidationVectorsPerIteration);

        std::vector<Position> trainingPositions, validationPositions;
        trainingPositions.resize(cNumTrainingVectorsPerIteration);
        validationPositions.resize(cNumValidationVectorsPerIteration);

        nn::TrainingVector validationVector;

        nn::Layer::Values tempValues;

        uint32_t numTrainingVectorsPassed = 0;
        uint32_t numTrainingVectorsPassedInEpoch = 0;

        for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
        {
            generateTrainingSet(trainingSet, trainingPositions);
            network.Train(trainingSet, tempValues, cBatchSize);

            numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;
            numTrainingVectorsPassedInEpoch += cNumTrainingVectorsPerIteration;

            generateTrainingSet(validationSet, validationPositions);
            
            float nnMinError = std::numeric_limits<float>::max();
            float nnMaxError = 0.0f, nnErrorSum = 0.0f;

            float evalMinError = -std::numeric_limits<float>::max();
            float evalMaxError = 0.0f, evalErrorSum = 0.0f;

            for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
            {
                tempValues = network.Run(validationSet[i].input);

                const float expectedValue = validationSet[i].output[0];
                const float nnValue = std::clamp(tempValues[0], 0.0f, 1.0f);
                const float evalValue = PawnToWinProbability((float)Evaluate(validationPositions[i]) / 100.0f);

                if (i < 10)
                {
                    std::cout << validationPositions[i].ToFEN() << std::endl;
                    std::cout << "Score:       " << expectedValue << std::endl;
                    std::cout << "Static eval: " << evalValue << std::endl;
                    std::cout << "NN eval:     " << nnValue << std::endl;
                }

                {
                    const float error = expectedValue - nnValue;
                    const float errorDiff = std::abs(error);
                    nnErrorSum += error * error;
                    nnMinError = std::min(nnMinError, errorDiff);
                    nnMaxError = std::max(nnMaxError, errorDiff);
                }

                {
                    const float error = expectedValue - evalValue;
                    const float errorDiff = std::abs(error);
                    evalErrorSum += error * error;
                    evalMinError = std::min(evalMinError, errorDiff);
                    evalMaxError = std::max(evalMaxError, errorDiff);
                }
            }
            nnErrorSum = sqrtf(nnErrorSum / cNumValidationVectorsPerIteration);
            evalErrorSum = sqrtf(evalErrorSum / cNumValidationVectorsPerIteration);

            std::cout
                << std::setw(8) << numTrainingVectorsPassed << " | \t"
                << std::setw(8) << std::setprecision(4) << nnErrorSum << "\t"
                << std::setw(8) << std::setprecision(4) << nnMinError << "\t"
                << std::setw(8) << std::setprecision(4) << nnMaxError << " | \t"
                << std::setw(8) << std::setprecision(4) << evalErrorSum << "\t"
                << std::setw(8) << std::setprecision(4) << evalMinError << "\t"
                << std::setw(8) << std::setprecision(4) << evalMaxError
                << std::endl;
        }

        network.Save("network.dat");
    }

    return true;
}
