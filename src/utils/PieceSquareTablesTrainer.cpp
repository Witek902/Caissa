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

static const uint32_t cMaxIterations = 100000000;
static const uint32_t cNumTrainingVectorsPerIteration = 512 * 1024;
static const uint32_t cBatchSize = 1024;
//static const uint32_t cNumNetworkInputs = 5 * 64 + 48;
static const uint32_t cNumNetworkInputs = 704; // 2 * 10 * 32 * 64;

static void PositionToTrainingVector(const Position& pos, nn::TrainingVector& outVector)
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

    /*
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
    */
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

bool TrainPieceSquareTables()
{
    std::vector<PositionEntry> entries;
    LoadAllPositions(entries);

    std::cout << "Training with " << entries.size() << " positions" << std::endl;

    nn::NeuralNetwork network;
    network.Init(cNumNetworkInputs, { 1 }, nn::ActivationFunction::Sigmoid);

    nn::NeuralNetworkRunContext networkRunCtx;
    networkRunCtx.Init(network);

    nn::NeuralNetworkTrainer trainer;

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

            //// flip the board randomly
            //const bool pawnless = pos.Whites().pawns == 0 && pos.Blacks().pawns == 0;
            //const bool noCastlingRights = pos.GetBlacksCastlingRights() == 0 && pos.GetWhitesCastlingRights() == 0;
            //if (pawnless || noCastlingRights)
            //{
            //    if (std::uniform_int_distribution<>(0, 1)(gen) != 0)
            //    {
            //        pos.MirrorHorizontally();
            //    }
            //}
            //if (pawnless)
            //{
            //    if (std::uniform_int_distribution<>(0, 1)(gen) != 0)
            //    {
            //        pos.MirrorVertically();
            //    }
            //}

            PositionToTrainingVector(pos, trainingSet[i]);
            trainingSet[i].output[0] = entry.score;
        }

        const float learningRate = std::max(0.05f, 1.0f / (1.0f + 0.001f * iteration));
        trainer.Train(network, trainingSet, cBatchSize, learningRate);

        //// normalize king weights
        //{
        //    float kingAvg = 0.0f;
        //    for (uint32_t i = cNumNetworkInputs - 64; i < cNumNetworkInputs; ++i)
        //    {
        //        kingAvg += network.layers[0].weights[i];
        //    }
        //    kingAvg /= 64.0f;
        //    for (uint32_t i = cNumNetworkInputs - 64; i < cNumNetworkInputs; ++i)
        //    {
        //        network.layers[0].weights[i] -= kingAvg;
        //    }
        //}

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

            //const nn::Values& networkOutput = network.Run(validationVector.inputs, networkRunCtx);
            const nn::Values& networkOutput = network.Run(validationVector.features.data(), (uint32_t)validationVector.features.size(), networkRunCtx);

            const float expectedValue = validationVector.output[0];

            if (i == 0)
            {
                std::cout << pos.ToFEN() << std::endl << pos.Print();
                std::cout << "Value:    " << networkOutput[0] << std::endl;
                std::cout << "Expected: " << expectedValue << std::endl;
                //PrintPieceSquareTableWeigts(network);
            }


            if (i == 0)
            {
                Position pos2("rnbqkbnr/pppppppp/8/K7/8/8/PPPPPPPP/RNBQ1BNR w kq - 0 1");
                PositionToTrainingVector(pos2, validationVector);
                const nn::Values& networkOutput2 = network.Run(validationVector.features.data(), (uint32_t)validationVector.features.size(), networkRunCtx);
                std::cout << pos.ToFEN() << std::endl << pos2.Print();
                std::cout << "Value:    " << networkOutput2[0] << std::endl;
            }

            const float error = expectedValue - networkOutput[0];
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
