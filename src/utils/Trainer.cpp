#include "Common.hpp"

#include "../backend/Position.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"
#include "../backend/NeuralNetwork.hpp"
#include "../backend/ThreadPool.hpp"
#include "../backend/tablebase/tbprobe.h"

#include <iostream>
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

static const uint32_t cNumIterations = 10000;
static const uint32_t cNumTrainingVectorsPerIteration = 10000;
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

    for (uint32_t iteration = 0; iteration < cNumIterations; ++iteration)
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
            network.NextEpoch();
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
    }

    network.Save("network.dat");

    return true;
}

bool TrainEndgame()
{
    const uint32_t networkInputs = 32 + 64 + 48 + 48;

    struct Position_KPvKP
    {
        Square whiteKingSquare;
        Square blackKingSquare;
        Square whitePawnSquare;
        Square blackPawnSquare;
        int8_t score;

        void ToVector(nn::TrainingVector& outVector) const
        {
            outVector.input.resize(networkInputs);
            outVector.output.resize(3);

            memset(outVector.input.data(), 0, sizeof(float) * networkInputs);

            const uint32_t whiteKingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();

            outVector.input[whiteKingIndex] = 1.0f;
            outVector.input[32 + blackKingSquare.Index()] = 1.0f;
            outVector.input[32 + 64 + (whitePawnSquare.Index() - 8)] = 1.0f;
            outVector.input[32 + 64 + 48 + (blackPawnSquare.Index() - 8)] = 1.0f;

            outVector.output[0] = score > 0 ? 1.0f : 0.0f;
            outVector.output[1] = score < 0 ? 1.0f : 0.0f;
            outVector.output[2] = score == 0 ? 1.0f : 0.0f;
        }
    };

    std::vector<Position_KPvKP> positions;

    uint32_t wins = 0, losses = 0, draws = 0;

    // generate white king on files A-D
    for (uint32_t whiteKingSquareFile = 0; whiteKingSquareFile < 4; ++whiteKingSquareFile)
    {
        for (uint32_t whiteKingSquareRank = 0; whiteKingSquareRank < 8; ++whiteKingSquareRank)
        {
            std::cout << (4 * whiteKingSquareRank + whiteKingSquareFile) << " / 32" << std::endl;

            const Square whiteKingSquare(8 * whiteKingSquareRank + whiteKingSquareFile);

            for (uint32_t blackKingSquareIndex = 0; blackKingSquareIndex < 64; ++blackKingSquareIndex)
            {
                const Square blackKingSquare(blackKingSquareIndex);

                if (Square::Distance(whiteKingSquare, blackKingSquare) <= 1)  continue;

                for (uint32_t whitePawnSqareIndex = 0; whitePawnSqareIndex < 48; ++whitePawnSqareIndex)
                {
                    const Square whitePawnSquare(whitePawnSqareIndex + 8);

                    if (whitePawnSquare == whiteKingSquare) continue;
                    if (whitePawnSquare == blackKingSquare) continue;

                    // black king cannot be in check
                    if (Bitboard::GetPawnAttacks(whitePawnSquare, Color::White) & blackKingSquare.GetBitboard()) continue;

                    for (uint32_t blackPawnSqareIndex = 0; blackPawnSqareIndex < 48; ++blackPawnSqareIndex)
                    {
                        const Square blackPawnSquare(blackPawnSqareIndex + 8);

                        if (blackPawnSquare == whiteKingSquare) continue;
                        if (blackPawnSquare == blackKingSquare) continue;
                        if (blackPawnSquare == whitePawnSquare) continue;

                        // skip check positions as those are not quiet
                        if (Bitboard::GetPawnAttacks(blackPawnSquare, Color::Black) & whiteKingSquare.GetBitboard()) continue;

                        int tbResult = tb_probe_wdl(
                            whiteKingSquare.GetBitboard() | whitePawnSquare.GetBitboard(),
                            blackKingSquare.GetBitboard() | blackPawnSquare.GetBitboard(),
                            whiteKingSquare.GetBitboard() | blackKingSquare.GetBitboard(),
                            0, 0, 0, 0,
                            whitePawnSquare.GetBitboard() | blackPawnSquare.GetBitboard(),
                            0, 0, 0, true);

                        if (tbResult == TB_RESULT_FAILED) continue;

                        int8_t score = 0;

                        if (tbResult == TB_WIN)
                        {
                            score = 1;
                            wins++;
                        }

                        if (tbResult == TB_LOSS)
                        {
                            score = -1;
                            losses++;
                        }

                        if (score == 0)
                        {
                            draws++;
                        }

                        positions.emplace_back(whiteKingSquare, blackKingSquare, whitePawnSquare, blackPawnSquare, score);
                    }
                }
            }
        }
    }

    std::cout << "KPvKP positions: " << positions.size() << std::endl;
    std::cout << "Wins:   " << wins << std::endl;
    std::cout << "Losses: " << losses << std::endl;
    std::cout << "Draws:  " << draws << std::endl;

    {
        nn::NeuralNetwork network;
        network.Init(networkInputs, { 16, 8, 3 });

        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<nn::TrainingVector> trainingSet;
        trainingSet.resize(cNumTrainingVectorsPerIteration);

        nn::TrainingVector validationVector;

        nn::Layer::Values tempValues;

        uint32_t numTrainingVectorsPassed = 0;
        uint32_t numTrainingVectorsPassedInEpoch = 0;

        for (uint32_t iteration = 0; iteration < cNumIterations; ++iteration)
        {
            // pick random test entries
            std::uniform_int_distribution<size_t> distrib(0, positions.size() - 1);
            for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
            {
                positions[distrib(gen)].ToVector(trainingSet[i]);
            }
            network.Train(trainingSet, tempValues, cBatchSize);

            numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;
            numTrainingVectorsPassedInEpoch += cNumTrainingVectorsPerIteration;

            if (numTrainingVectorsPassedInEpoch > positions.size())
            {
                network.NextEpoch();
                numTrainingVectorsPassedInEpoch = 0;
            }

            float errorSum = 0.0f;
            for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
            {
                positions[distrib(gen)].ToVector(validationVector);

                tempValues = network.Run(validationVector.input);

                float expectedValue = 0.0f;
                if (validationVector.output[0] > validationVector.output[1] && validationVector.output[0] > validationVector.output[2]) expectedValue = 1.0f;
                if (validationVector.output[1] > validationVector.output[0] && validationVector.output[1] > validationVector.output[2]) expectedValue = -1.0f;

                float returnedValue = 0.0f;
                if (tempValues[0] > tempValues[1] && tempValues[0] > tempValues[2]) returnedValue = 1.0f;
                if (tempValues[1] > tempValues[0] && tempValues[1] > tempValues[2]) returnedValue = -1.0f;

                const float error = expectedValue - returnedValue;
                errorSum += fabsf(error);
            }
            errorSum = errorSum / cNumValidationVectorsPerIteration;

            float epoch = (float)numTrainingVectorsPassed / (float)positions.size();
            std::cout << epoch << "\t" << errorSum << std::endl;
        }

        network.Save("network.dat");
    }

    return true;
}
