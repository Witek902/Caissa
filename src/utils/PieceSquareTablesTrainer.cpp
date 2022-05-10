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
static const uint32_t cNumTrainingVectorsPerIteration = 512 * 1024;
static const uint32_t cBatchSize = 256;
static const uint32_t cNumNetworkInputs = 5 * 64 + 48;

struct PositionEntry
{
    PackedPosition pos;
    float score;
};

static void PositionToTrainingVector(const Position& pos, nn::TrainingVector& outVector)
{
    ASSERT(pos.GetSideToMove() == Color::White);

    outVector.output.resize(1);
    outVector.inputs.resize(cNumNetworkInputs);
    memset(outVector.inputs.data(), 0, sizeof(float) * cNumNetworkInputs);

    uint32_t offset = 0;

    const auto writePieceFeatures = [&](const Bitboard bitboard, const Color color) INLINE_LAMBDA
    {
        bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            outVector.inputs[offset + square] += (color == Color::White ? 1.0f : -1.0f);
        });
    };

    const auto writePawnFeatures = [&](const Bitboard bitboard, const Color color) INLINE_LAMBDA
    {
        // pawns cannot stand on first or last rank
        for (uint32_t i = 0; i < 48u; ++i)
        {
            const uint32_t squreIndex = i + 8u;
            if ((bitboard >> squreIndex) & 1)
            {
                outVector.inputs[offset + i] += (color == Color::White ? 1.0f : -1.0f);
            }
        }
    };

    writePawnFeatures(pos.Whites().pawns, Color::White);
    writePawnFeatures(pos.Blacks().pawns.MirroredVertically(), Color::Black);
    offset += 48;

    writePieceFeatures(pos.Whites().knights, Color::White);
    writePieceFeatures(pos.Blacks().knights.MirroredVertically(), Color::Black);
    offset += 64;

    writePieceFeatures(pos.Whites().bishops, Color::White);
    writePieceFeatures(pos.Blacks().bishops.MirroredVertically(), Color::Black);
    offset += 64;

    writePieceFeatures(pos.Whites().rooks, Color::White);
    writePieceFeatures(pos.Blacks().rooks.MirroredVertically(), Color::Black);
    offset += 64;

    writePieceFeatures(pos.Whites().queens, Color::White);
    writePieceFeatures(pos.Blacks().queens.MirroredVertically(), Color::Black);
    offset += 64;

    // white kings
    {
        outVector.inputs[offset + FirstBitSet(pos.Whites().king)] += 1.0f;
        outVector.inputs[offset + FirstBitSet(pos.Blacks().king.MirroredVertically())] += -1.0f;
        offset += 64;
    }

    ASSERT(offset == cNumNetworkInputs);
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

        float avg = 0.0f;
        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            for (uint32_t file = 0; file < 8; file++)
            {
                avg += weights[offset + 8 * rank + file];
            }
        }
        avg /= 64.0f;
        std::cout << "Average: " << int32_t(c_nnOutputToCentiPawns * avg) << std::endl;

        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            std::cout << "    ";
            code << "    ";
            for (uint32_t file = 0; file < 8; file++)
            {
                const float weight = c_nnOutputToCentiPawns * (weights[offset + 8 * rank + file] - avg);
                std::cout << std::right << std::fixed << std::setw(6) << int32_t(weight) << " ";
                code << std::right << std::fixed << std::setw(6) << int32_t(weight) << ", ";
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

        float avg = 0.0f;
        for (uint32_t rank = 1; rank < 7; ++rank)
        {
            for (uint32_t file = 0; file < 8; file++)
            {
                avg += weights[offset + 8 * (rank - 1) + file];
            }
        }
        avg /= 48.0f;
        std::cout << "Average: " << int32_t(c_nnOutputToCentiPawns * avg) << std::endl;

        // pawns cannot stand on first or last rank
        for (uint32_t rank = 1; rank < 7; ++rank)
        {
            std::cout << "    ";
            for (uint32_t file = 0; file < 8; file++)
            {
                const float weight = c_nnOutputToCentiPawns * (weights[offset + 8 * (rank - 1) + file] - avg);
                std::cout << std::right << std::fixed << std::setw(6) << int32_t(weight) << " ";
                code << std::right << std::fixed << std::setw(6) << int32_t(weight) << ", ";
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

    std::cout << "Eval offset: " << int32_t(c_nnOutputToCentiPawns * weights[offset]) << std::endl;

    //std::cout << "Code:" << std::endl;
    //std::cout << code.str() << std::endl;

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

    GameCollection::Reader reader(gamesFile);

    uint32_t numGames = 0;

    Game game;
    while (reader.ReadGame(game))
    {
        Game::Score gameScore = game.GetScore();

        ASSERT(game.GetMoves().size() == game.GetMoveScores().size());

        if (game.GetScore() == Game::Score::Unknown)
        {
            continue;
        }

        float score = 0.5f;
        if (gameScore == Game::Score::WhiteWins) score = 1.0f;
        if (gameScore == Game::Score::BlackWins) score = 0.0f;

        Position pos = game.GetInitialPosition();

        // replay the game
        for (size_t i = 0; i < game.GetMoves().size(); ++i)
        {
            const float gamePhase = (float)i / (float)game.GetMoves().size();
            const Move move = pos.MoveFromPacked(game.GetMoves()[i]);
            const ScoreType moveScore = game.GetMoveScores()[i];
            const MaterialKey matKey = pos.GetMaterialKey();

            if (pos.GetHalfMoveCount() > 20 && i > 40 && gameScore == Game::Score::Draw)
            {
                break;
            }

            const Square whiteKingSq(FirstBitSet(pos.Whites().king));
            const Square blackKingSq(FirstBitSet(pos.Blacks().king));

            // skip boring equal positions
            const bool equalPosition =
                matKey.numBlackPawns == matKey.numWhitePawns &&
                matKey.numBlackKnights == matKey.numWhiteKnights &&
                matKey.numBlackBishops == matKey.numWhiteBishops &&
                matKey.numBlackRooks == matKey.numWhiteRooks &&
                matKey.numBlackQueens == matKey.numWhiteQueens &&
                gameScore == Game::Score::Draw &&
                std::abs(moveScore) < 10;

            if (!equalPosition &&
                !pos.IsInCheck() && !move.IsCapture() && !move.IsPromotion() &&
                (whiteKingSq.Rank() <= 2 && blackKingSq.Rank() >= 5) &&
                pos.GetNumPieces() >= 16)
            {
                PositionEntry entry{};

                // blend between eval score and actual game score
                const float blendWeight = std::lerp(0.25f, 1.0f, gamePhase);
                entry.score = std::lerp(CentiPawnToWinProbability(moveScore), score, blendWeight);

                Position normalizedPos = pos;
                if (pos.GetSideToMove() == Color::Black)
                {
                    // make whites side to move
                    normalizedPos = normalizedPos.SwappedColors();
                    entry.score = 1.0f - entry.score;
                }

                PackPosition(normalizedPos, entry.pos);

                entries.push_back(entry);
            }

            if (!pos.DoMove(move))
            {
                break;
            }
        }

        numGames++;
    }

    std::cout << "Parsed " << numGames << " games" << std::endl;

    return true;
}

bool TrainPieceSquareTables()
{
    std::vector<PositionEntry> entries;
    LoadPositions("../../data/selfplayGames/selfplay3.dat", entries);
    LoadPositions("../../data/selfplayGames/selfplay4.dat", entries);

    std::cout << "Training with " << entries.size() << " positions" << std::endl;

    nn::NeuralNetwork network;
    network.Init(cNumNetworkInputs, { 1 }, nn::ActivationFunction::Sigmoid);

    // reset king weights
    for (uint32_t i = cNumNetworkInputs - 64; i < cNumNetworkInputs; ++i)
    {
        network.layers[0].weights[i] = 0.0f;
    }

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

            // flip the board randomly
            const bool pawnless = pos.Whites().pawns == 0 && pos.Blacks().pawns == 0;
            const bool noCastlingRights = pos.GetBlacksCastlingRights() == 0 && pos.GetWhitesCastlingRights() == 0;
            if (pawnless || noCastlingRights)
            {
                if (std::uniform_int_distribution<>(0, 1)(gen) != 0)
                {
                    pos.MirrorHorizontally();
                }
            }
            if (pawnless)
            {
                if (std::uniform_int_distribution<>(0, 1)(gen) != 0)
                {
                    pos.MirrorVertically();
                }
            }

            PositionToTrainingVector(pos, trainingSet[i]);
            trainingSet[i].output[0] = entry.score;
        }

        const float learningRate = 0.5f / (1.0f + 0.001f * iteration);
        network.Train(trainingSet, tempValues, cBatchSize, learningRate);

        // normalize king weights
        {
            float kingAvg = 0.0f;
            for (uint32_t i = cNumNetworkInputs - 64; i < cNumNetworkInputs; ++i)
            {
                kingAvg += network.layers[0].weights[i];
            }
            kingAvg /= 64.0f;
            for (uint32_t i = cNumNetworkInputs - 64; i < cNumNetworkInputs; ++i)
            {
                network.layers[0].weights[i] -= kingAvg;
            }
        }

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float minError = std::numeric_limits<float>::max();
        float maxError = -std::numeric_limits<float>::max();
        float errorSum = 0.0f;
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];

            Position pos;
            UnpackPosition(entry.pos, pos);
            PositionToTrainingVector(pos, validationVector);
            validationVector.output[0] = entry.score;

            tempValues = network.Run(validationVector.features.data(), (uint32_t)validationVector.features.size());

            const float expectedValue = validationVector.output[0];

            if (i == 0)
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
