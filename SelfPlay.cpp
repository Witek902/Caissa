#include <iostream>
#include "Position.hpp"
#include "Move.hpp"
#include "Search.hpp"
#include "Evaluate.hpp"
#include "NeuralNetwork.hpp"

#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <intrin.h>

#include "ThreadPool.hpp"

using namespace threadpool;

struct PositionEntry
{
    uint64_t whiteKing;
    uint64_t whitePawns;
    uint64_t whiteKnights;
    uint64_t whiteBishops;
    uint64_t whiteRooks;
    uint64_t whiteQueens;

    uint64_t blackKing;
    uint64_t blackPawns;
    uint64_t blackKnights;
    uint64_t blackBishops;
    uint64_t blackRooks;
    uint64_t blackQueens;

    uint8_t sideToMove : 1;
    uint8_t whiteCastlingRights : 2;
    uint8_t blackCastlingRights : 2;

    int32_t eval;
    int32_t gameResult;
    uint16_t moveNumber;
    uint16_t totalMovesInGame;
};

void SelfPlay()
{
    FILE* dumpFile = nullptr;
    fopen_s(&dumpFile, "selfplay.dat", "wb");

    std::vector<Search> searchArray{ std::thread::hardware_concurrency() };
    
    std::mutex mutex;
    uint32_t games = 0;
    uint32_t whiteWins = 0;
    uint32_t blackWins = 0;
    uint32_t draws = 0;

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

        taskBuilder.ParallelFor("SelfPlay", 20000, [&](const TaskContext& context, uint32_t)
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            Search& search = searchArray[context.threadId];
            search.ClearPositionHistory();
            search.GetTranspositionTable().Clear();

            Position position(Position::InitPositionFEN);
            SearchResult searchResult;

            int32_t score = 0;
            std::stringstream pgnString;
            std::vector<PositionEntry> posEntries;
            posEntries.reserve(200);

            int32_t scoreDiffTreshold = 50;
            uint32_t maxMoves = 500;

            uint32_t halfMoveNumber = 0;
            for (;; ++halfMoveNumber)
            {
                search.RecordBoardPosition(position);

                SearchParam searchParam;
                searchParam.maxDepth = 8;
                searchParam.numPvLines = 4;
                searchParam.debugLog = false;

                searchResult.clear();

                search.DoSearch(position, searchParam, searchResult);

                if (searchResult.empty())
                {
                    break;
                }

                // if one of the move is much worse than the best candidate, ignore it and the rest
                for (size_t i = 1; i < searchResult.size(); ++i)
                {
                    int32_t diff = searchResult[i].score - searchResult[0].score;
                    if (diff > scoreDiffTreshold || diff < -scoreDiffTreshold)
                    {
                        searchResult.erase(searchResult.begin() + i, searchResult.end());
                        break;
                    }
                }

                // select random move
                // TODO prefer moves with higher score
                std::uniform_int_distribution<size_t> distrib(0, searchResult.size() - 1);
                const size_t moveIndex = distrib(gen);
                ASSERT(!searchResult[moveIndex].moves.empty());
                const Move move = searchResult[moveIndex].moves.front();
                score = searchResult[moveIndex].score;
                if (position.GetSideToMove() == Color::Black) score = -score;

                // if didn't picked best move, reduce treshold of picking worse move
                // this way the game will be more random at the beginning and there will be less blunders later in the game
                if (moveIndex > 0)
                {
                    scoreDiffTreshold = std::max(10, scoreDiffTreshold - 5);
                }

                // log move
                {
                    if (halfMoveNumber % 2 == 0)
                    {
                        pgnString << (1 + (halfMoveNumber / 2)) << ". ";
                    }
                    pgnString << position.MoveToString(move) << " ";
                }

                // dump position
                {
                    PositionEntry entry =
                    {
                        position.Whites().king,
                        position.Whites().pawns,
                        position.Whites().knights,
                        position.Whites().bishops,
                        position.Whites().rooks,
                        position.Whites().queens,

                        position.Blacks().king,
                        position.Blacks().pawns,
                        position.Blacks().knights,
                        position.Blacks().bishops,
                        position.Blacks().rooks,
                        position.Blacks().queens,

                        (uint8_t)position.GetSideToMove(),
                        (uint8_t)position.GetWhitesCastlingRights(),
                        (uint8_t)position.GetBlacksCastlingRights(),

                        score,
                        0, // game result
                        (uint16_t)halfMoveNumber,
                        0, // total number of moves
                    };

                    posEntries.push_back(entry);
                }

                const bool moveSuccess = position.DoMove(move);
                ASSERT(moveSuccess);

                // check for draw
                if (search.IsPositionRepeated(position) ||
                    position.GetHalfMoveCount() >= 100 ||
                    CheckInsufficientMaterial(position) ||
                    halfMoveNumber > maxMoves)
                {
                    score = 0;
                    break;
                }
            }

            // put missing data in entries
            for (PositionEntry& entry : posEntries)
            {
                if (score > 0)
                {
                    entry.gameResult = 1;
                }
                else if (score < 0)
                {
                    entry.gameResult = -1;
                }
                else
                {
                    entry.gameResult = 0;
                }
                entry.totalMovesInGame = (uint16_t)halfMoveNumber;
            }

            {
                std::unique_lock<std::mutex> lock(mutex);

                fwrite(posEntries.data(), sizeof(PositionEntry), posEntries.size(), dumpFile);
                fflush(dumpFile);

                const uint32_t gameNumber = games++;

                std::cout << "Game #" << gameNumber << " ";
                std::cout << pgnString.str();

                if (score > 0)
                {
                    //ASSERT(position.GetSideToMove() == Color::White);
                    std::cout << "(white won)";
                    whiteWins++;
                }
                else if (score < 0)
                {
                    //ASSERT(position.GetSideToMove() == Color::Black);
                    std::cout << "(black won)";
                    blackWins++;
                }
                else
                {
                    if (search.IsPositionRepeated(position)) std::cout << "(draw by repetition)"; 
                    else if (position.GetHalfMoveCount() >= 100) std::cout << "(draw by 50 move rule)";
                    else if (CheckInsufficientMaterial(position)) std::cout << "(draw by insufficient material)";
                    else std::cout << "(draw by too long game)";

                    draws++;
                }

                std::cout << " W:" << whiteWins << " B:" << blackWins << " D:" << draws << std::endl;
            }
        });

        //taskBuilder.Fence();
    }

    waitable.Wait();

    fclose(dumpFile);
}

/*
static float ClampNetworkOutput(float value)
{
    return std::max(-127.0f, std::min(127.0f, value));
}
*/

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

static const uint32_t cNumIterations = 1000;
static const uint32_t cNumTrainingVectorsPerIteration = 2048;
static const uint32_t cNumValidationVectorsPerIteration = 100;
static const uint32_t cBatchSize = 64;

#pragma optimize("",off)

bool Train()
{
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    FILE* dumpFile = nullptr;
    fopen_s(&dumpFile, "selfplay.dat", "rb");

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
    network.Init(12 * 64, { 1024, 512, 256, 1 });
    // if (!network.Load("network.dat")) return false;

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

        float minError = FLT_MAX;
        float maxError = FLT_MIN;
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