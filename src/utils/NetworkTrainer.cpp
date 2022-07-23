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
static const uint32_t cNumTrainingVectorsPerIteration = 8192;
static const uint32_t cBatchSize = 128;
static const uint32_t cNumNetworkInputs = 10 * 64 + 2 * 48;

struct PositionEntry
{
    PackedPosition pos;
    float score;
};

struct TrainingEntry
{
    Position pos;
    nn::TrainingVector trainingVector;
};

static void PositionToSparseVector(const Position& pos, nn::TrainingVector& outVector)
{
    const uint32_t maxFeatures = 64;

    uint16_t features[maxFeatures];
    uint32_t numFeatures = pos.ToFeaturesVector(features, NetworkInputMapping::Full);
    ASSERT(numFeatures <= maxFeatures);

    outVector.output.resize(1);
    outVector.features.clear();
    outVector.features.reserve(numFeatures);

    for (uint32_t i = 0; i < numFeatures; ++i)
    {
        outVector.features.push_back(features[i]);
    }
}

static bool LoadPositions(const char* fileName, std::vector<PositionEntry>& entries)
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

            // skip boring equal positions
            const bool equalPosition =
                matKey.numBlackPawns == matKey.numWhitePawns &&
                matKey.numBlackKnights == matKey.numWhiteKnights &&
                matKey.numBlackBishops == matKey.numWhiteBishops &&
                matKey.numBlackRooks == matKey.numWhiteRooks &&
                matKey.numBlackQueens == matKey.numWhiteQueens &&
                gameScore == Game::Score::Draw &&
                std::abs(moveScore) < 15;

            if (!equalPosition &&
                !pos.IsInCheck() && !move.IsCapture() && !move.IsPromotion() &&
                pos.GetNumPieces() >= 6 &&
                std::abs(Evaluate(pos) < 1024))
            {
                PositionEntry entry{};

                // blend in future scores into current move score
                float scoreSum = 0.0f;
                float weightSum = 0.0f;
                const size_t maxLookahead = 10;
                for (size_t j = 0; j < maxLookahead; ++j)
                {
                    if (i + j >= game.GetMoves().size()) break;
                    const float weight = 1.0f / (j + 1);
                    scoreSum += weight * CentiPawnToWinProbability(game.GetMoveScores()[i + j]);
                    weightSum += weight;
                }
                ASSERT(weightSum > 0.0f);
                scoreSum /= weightSum;

                // blend between eval score and actual game score
                const float blendWeight = std::lerp(0.0f, 0.5f, gamePhase);
                entry.score = std::lerp(scoreSum, score, blendWeight);

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

bool TrainNetwork()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    nn::NeuralNetwork network;
    network.Init(cNumNetworkInputs,
                 { nn::FirstLayerSize, 32, 64, 1 },
                 nn::ActivationFunction::Sigmoid);

    nn::PackedNeuralNetwork packedNetwork;

    std::vector<PositionEntry> entries;
    LoadPositions("../../data/selfplayGames/selfplay2.dat", entries);
    //LoadPositions("../../data/selfplayGames/selfplay3.dat", entries);
    //LoadPositions("../../data/selfplayGames/selfplay4.dat", entries);

    std::cout << "Training with " << entries.size() << " positions" << std::endl;

    std::vector<TrainingEntry> trainingSet, validationSet;
    trainingSet.resize(cNumTrainingVectorsPerIteration);
    validationSet.resize(cNumTrainingVectorsPerIteration);

    nn::Layer::Values tempValues;

    uint32_t numTrainingVectorsPassed = 0;

    const auto generateTrainingSet = [&](std::vector<TrainingEntry>& outEntries)
    {
        // pick random test entries
        std::uniform_int_distribution<size_t> distrib(0, entries.size() - 1);
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            const PositionEntry& entry = entries[distrib(gen)];
            Position pos;
            UnpackPosition(entry.pos, pos);

            /*
            // flip the board randomly
            const bool pawnless = pos.Whites().pawns == 0 && pos.Blacks().pawns == 0;
            const bool noCastlingRights = pos.GetBlacksCastlingRights() == 0 && pos.GetWhitesCastlingRights() == 0;
            if (pawnless || noCastlingRights)
            {
                if (std::uniform_int_distribution<>(0,1)(gen) != 0)
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
            */

            PositionToSparseVector(pos, outEntries[i].trainingVector);
            outEntries[i].trainingVector.output[0] = entry.score;
            outEntries[i].pos = pos;
        }
    };

    generateTrainingSet(validationSet);

    for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        float learningRate = 1.0f / (1.0f + 0.0001f * iteration);

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

                TimePoint startTime = TimePoint::GetCurrent();
                network.Train(batch, tempValues, cBatchSize, learningRate);
                TimePoint endTime = TimePoint::GetCurrent();

                std::cout << "Training took " << (endTime - startTime).ToSeconds() << " sec" << std::endl;

                for (uint32_t i = 0; i < nn::FirstLayerSize; ++i)
                {
                    network.layers[0].weights[(i + 1) * (cNumNetworkInputs + 1) - 1] = 0.0f;
                }

                network.ToPackedNetwork(packedNetwork);
                packedNetwork.Save("pawns.nn");
            });

            taskBuilder.Task("GenerateSet", [&](const TaskContext&)
            {
                generateTrainingSet(validationSet);
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
            const std::vector<uint16_t>& features = validationSet[i].trainingVector.features;
            tempValues = network.Run(features.data(), (uint32_t)features.size());
            int32_t packedNetworkOutput = packedNetwork.Run(features.data(), (uint32_t)features.size());

            const float expectedValue = validationSet[i].trainingVector.output[0];
            const float nnValue = tempValues[0];
            const float nnPackedValue = nn::Sigmoid((float)packedNetworkOutput / (float)nn::OutputScale);
            const float evalValue = PawnToWinProbability((float)Evaluate(validationSet[i].pos) / 100.0f);

            nnPackedQuantizationErrorSum += (nnValue - nnPackedValue) * (nnValue - nnPackedValue);

            if (i + 1 == cNumTrainingVectorsPerIteration)
            {
                std::cout
                    << validationSet[i].pos.ToFEN() << std::endl << validationSet[i].pos.Print() << std::endl
                    << "True Score:     " << expectedValue << std::endl
                    << "NN eval:        " << nnValue << std::endl
                    << "Packed NN eval: " << nnPackedValue << std::endl
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

        nnErrorSum = sqrtf(nnErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedErrorSum = sqrtf(nnPackedErrorSum / cNumTrainingVectorsPerIteration);
        evalErrorSum = sqrtf(evalErrorSum / cNumTrainingVectorsPerIteration);
        nnPackedQuantizationErrorSum = sqrtf(nnPackedQuantizationErrorSum / cNumTrainingVectorsPerIteration);

        std::cout
            << "Num training vectors:   " << numTrainingVectorsPassed << std::endl
            << "Learning rate:          " << learningRate << std::endl
            << "NN avg/min/max error:   " << std::setprecision(5) << nnErrorSum << " " << std::setprecision(4) << nnMinError << " " << std::setprecision(4) << nnMaxError << std::endl
            << "PNN avg/min/max error:  " << std::setprecision(5) << nnPackedErrorSum << " " << std::setprecision(4) << nnPackedMinError << " " << std::setprecision(4) << nnPackedMaxError << std::endl
            << "Quantization error:     " << std::setprecision(5) << nnPackedQuantizationErrorSum << std::endl
            << "Eval avg/min/max error: " << std::setprecision(5) << evalErrorSum << " " << std::setprecision(4) << evalMinError << " " << std::setprecision(4) << evalMaxError << std::endl;

        network.PrintStats();

        network.Save("eval.nn");
        packedNetwork.Save("eval.pnn");
    }

    return true;
}
