#include "Common.hpp"
#include "ThreadPool.hpp"
#include "GameCollection.hpp"

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
static const uint32_t cNumTrainingVectorsPerIteration = 256 * 1024;
static const uint32_t cBatchSize = 256;
static const uint32_t cNumNetworkInputs = 10 * 64 + 2 * 48;

struct PositionEntry
{
    PackedPosition pos;
    float score;
};

static void PositionToSparseVector(const Position& pos, nn::TrainingVector& outVector)
{
    const uint32_t maxFeatures = 64;

    uint16_t features[maxFeatures];
    uint32_t numFeatures = pos.ToSparseFeaturesVector(features);
    ASSERT(numFeatures <= maxFeatures);

    outVector.output.resize(1);
    outVector.inputFeatures.clear();
    outVector.inputFeatures.reserve(numFeatures);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outVector.inputFeatures.push_back(features[i]);
    }
}

static void NormalizeKingWeights(nn::NeuralNetwork& nn)
{
    float* weights = nn.layers[0].weights.data();

    float avgWhiteKingWeight = 0.0f;
    float avgBlackKingWeight = 0.0f;

    const uint32_t whiteKingOffset = 48 + 4 * 64;
    const uint32_t blackKingOffset = whiteKingOffset + cNumNetworkInputs / 2;

    for (uint32_t i = whiteKingOffset; i < whiteKingOffset + 64; ++i)
    {
        avgWhiteKingWeight += weights[i];
    }
    for (uint32_t i = blackKingOffset; i < blackKingOffset + 64; ++i)
    {
        avgBlackKingWeight += weights[i];
    }

    avgWhiteKingWeight /= 64.0f;
    avgBlackKingWeight /= 64.0f;

    for (uint32_t i = whiteKingOffset; i < whiteKingOffset + 64; ++i)
    {
        weights[i] -= avgWhiteKingWeight;
    }
    for (uint32_t i = blackKingOffset; i < blackKingOffset + 64; ++i)
    {
        weights[i] -= avgBlackKingWeight;
    }
    weights[cNumNetworkInputs - 1] += avgWhiteKingWeight;
    weights[cNumNetworkInputs - 1] += avgBlackKingWeight;
}

static void PrintPieceSquareTableWeigts(const nn::NeuralNetwork& nn)
{
    const float* weights = nn.layers[0].weights.data();

    uint32_t offset = 0;

    std::stringstream code;

    const auto printPieceWeights = [&](const char* name)
    {
        std::cout << name << std::endl;
        code << "{\n";

        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            std::cout << "    ";
            code << "    ";
            for (uint32_t file = 0; file < 8; file++)
            {
                const float whiteWeight = weights[offset + 8 * rank + file];
                const float blackWeight = weights[offset + 8 * (7 - rank) + file + 5 * 64 + 48]; // skip white pieces
                const float weight = (whiteWeight - blackWeight) / 2.0f;
                std::cout << std::right << std::fixed << std::setprecision(3) << std::setw(6) << weight << " ";
                code << std::right << std::fixed << std::setw(6) << int32_t(4096 * weight) << ", ";
            }
            std::cout << std::endl;
            code << "\n";
        }
        offset += 64;

        std::cout << std::endl;
        code << "},\n";
    };

    const auto writePawnWeights = [&](const char* name)
    {
        std::cout << name << std::endl;
        code << "{\n";
        code << "    0, 0, 0, 0, 0, 0, 0, 0, \n";

        // pawns cannot stand on first or last rank
        for (uint32_t rank = 1; rank < 7; ++rank)
        {
            std::cout << "    ";
            for (uint32_t file = 0; file < 8; file++)
            {
                const float whiteWeight = weights[offset + 8 * (rank - 1) + file];
                const float blackWeight = weights[offset + 8 * (6 - rank) + file + 5 * 64 + 48]; // skip white pieces
                const float weight = (whiteWeight - blackWeight) / 2.0f;
                std::cout << std::right << std::fixed << std::setprecision(3) << std::setw(6) << weight << " ";
                code << std::right << std::fixed << std::setw(6) << int32_t(4096 * weight) << ", ";
            }
            std::cout << std::endl;
            code << "\n";
        }
        offset += 48;

        code << "    0, 0, 0, 0, 0, 0, 0, 0, \n";
        std::cout << std::endl;
        code << "},\n";
    };

    writePawnWeights("Pawn");
    printPieceWeights("Knights");
    printPieceWeights("Bishop");
    printPieceWeights("Rook");
    printPieceWeights("Queen");
    printPieceWeights("King");

    std::cout << "Eval offset: " << weights[offset] << std::endl;

    std::cout << "Code:" << std::endl;
    std::cout << code.str() << std::endl;

    ASSERT(offset == 5 * 64 + 48);
}

bool LoadPositions(const char* fileName, std::vector<PositionEntry>& entries)
{
    FileInputStream gamesFile(fileName);
    if (!gamesFile.IsOpen())
    {
        std::cout << "ERROR: Failed to load selfplay data file!" << std::endl;
        return false;
    }

    std::vector<Position> positions;

    GameCollection::Reader reader(gamesFile);

    Game game;
    while (reader.ReadGame(game))
    {
        positions.clear();

        // reconstruct game positions
        {
            Position pos = game.GetInitialPosition();

            ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

            for (size_t i = 0; i < game.GetMoves().size(); ++i)
            {
                const Move move = pos.MoveFromPacked(game.GetMoves()[i]);

                if (!pos.DoMove(move))
                {
                    std::cout << "Failed to do move " << move.ToString() << " in position " << pos.ToFEN() << std::endl;
                    break;
                }

                positions.push_back(pos);
            }
        }

        Game::Score gameScore = game.GetScore();

        float score = 0.5f;
        if (gameScore == Game::Score::WhiteWins) score = 1.0f;
        if (gameScore == Game::Score::BlackWins) score = 0.0f;

        for (size_t i = 0; i < game.GetMoves().size(); ++i)
        {
            const Position& pos = positions[i];

            if (!pos.IsInCheck(pos.GetSideToMove()) &&
                pos.GetMoveCount() >= 20 &&
                pos.GetNumPieces() <= 8 && pos.GetNumPieces() >= 4)
            {
                Position normalizedPos = pos;

                if (pos.GetSideToMove() == Color::Black)
                {
                    normalizedPos = normalizedPos.SwappedColors();
                }

                PositionEntry entry{};
                PackPosition(normalizedPos, entry.pos);
                entry.score = pos.GetSideToMove() == Color::White ? score : (1.0f - score);

                entries.push_back(entry);
            }
        }
    }

    return true;
}

bool TrainPieceSquareTables()
{
    std::vector<PositionEntry> entries;
    LoadPositions("selfplay.dat", entries);
    //LoadPositions("selfplay1.dat", entries);
    //LoadPositions("selfplay2.dat", entries);
    //LoadPositions("selfplay3.dat", entries);

    std::cout << "Training with " << entries.size() << " positions" << std::endl;

    nn::NeuralNetwork network;
    network.Init(cNumNetworkInputs, { 1 }, nn::ActivationFunction::Sigmoid);

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
        std::uniform_int_distribution<size_t> distrib(0, entries.size() - 1);
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];
            Position pos;
            UnpackPosition(entry.pos, pos);
            PositionToSparseVector(pos, trainingSet[i]);
            trainingSet[i].output[0] = entry.score;
        }
        network.Train(trainingSet, tempValues, cBatchSize);
        //NormalizeKingWeights(network);

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float minError = std::numeric_limits<float>::max();
        float maxError = -std::numeric_limits<float>::max();
        float errorSum = 0.0f;
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];

            Position pos;
            UnpackPosition(entry.pos, pos);
            PositionToSparseVector(pos, validationVector);
            validationVector.output[0] = entry.score;

            tempValues = network.Run(validationVector.inputFeatures.data(), (uint32_t)validationVector.inputFeatures.size());

            const float expectedValue = validationVector.output[0];

            if (i == 0 && iteration % 10 == 0)
            {
                std::cout << pos.ToFEN() << std::endl << pos.Print();
                std::cout << "    value= " << tempValues[0] << ", expected=" << expectedValue << std::endl;
                PrintPieceSquareTableWeigts(network);
            }

            const float error = expectedValue - tempValues[0];
            minError = std::min(minError, fabsf(error));
            maxError = std::max(maxError, fabsf(error));
            const float sqrError = error * error;
            errorSum += sqrError;
        }
        errorSum = sqrtf(errorSum / cNumTrainingVectorsPerIteration);

        float epoch = (float)numTrainingVectorsPassed / (float)entries.size();
        std::cout << std::right << std::fixed << std::setprecision(4) << epoch << " | ";
        std::cout << std::right << std::fixed << std::setprecision(4) << errorSum << " | ";
        std::cout << std::right << std::fixed << std::setprecision(4) << minError << " | ";
        std::cout << std::right << std::fixed << std::setprecision(4) << maxError << " | ";
        std::cout << std::endl;
    }

    return true;
}
