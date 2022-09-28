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
#include "../backend/Pawns.hpp"
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

//#define USE_MOBILITY
//#define USE_KING_TROPISM
//#define USE_PASSED_PAWNS

using namespace threadpool;

static const uint32_t cMaxIterations = 100000000;
static const uint32_t cNumTrainingVectorsPerIteration = 256 * 1024;
static const uint32_t cBatchSize = 64;

static const uint32_t cNumNetworkInputs =
    2 * 5 +                     // piece values
    2 * (5 * 64 + 48) +         // PSQT
    2                           // bishop pair
#ifdef USE_MOBILITY
    + 2 * (9 + 14 + 15 + 28)
#endif
#ifdef USE_KING_TROPISM
    + 2 * (2 * 5 * 7)           // king tropism: [white/black] x [5 pieces] x [7 possible distances]
#endif
#ifdef USE_PASSED_PAWNS
    + 2 * 5                     // passed pawn bonus (ranks 1 - 5)
#endif
    ;

static float GetGamePhase(const Position& pos)
{
    const int32_t gamePhase =
        1 * (pos.Whites().pawns.Count() + pos.Blacks().pawns.Count()) +
        2 * (pos.Whites().knights.Count() + pos.Blacks().knights.Count()) +
        2 * (pos.Whites().bishops.Count() + pos.Blacks().bishops.Count()) +
        4 * (pos.Whites().rooks.Count() + pos.Blacks().rooks.Count()) +
        8 * (pos.Whites().queens.Count() + pos.Blacks().queens.Count());

    return std::min(1.0f, (float)gamePhase / 64.0f);
}

static void PositionToTrainingVector(const Position& pos, nn::TrainingVector& outVector)
{
    ASSERT(pos.GetSideToMove() == Color::White);

    outVector.output.resize(1);
    outVector.inputMode = nn::InputMode::Sparse;
    outVector.sparseInputs.clear();
    std::vector<nn::ActiveFeature>& inputs = outVector.sparseInputs;

    uint32_t offset = 0;

    const float mg = GetGamePhase(pos);
    const float eg = 1.0f - mg;

    const Square whiteKingSq(FirstBitSet(pos.Whites().king));
    const Square blackKingSq(FirstBitSet(pos.Blacks().king));

    const int32_t wp = pos.Whites().pawns.Count();
    const int32_t wn = pos.Whites().knights.Count();
    const int32_t wb = pos.Whites().bishops.Count();
    const int32_t wr = pos.Whites().rooks.Count();
    const int32_t wq = pos.Whites().queens.Count();
    const int32_t bp = pos.Blacks().pawns.Count();
    const int32_t bn = pos.Blacks().knights.Count();
    const int32_t bb = pos.Blacks().bishops.Count();
    const int32_t br = pos.Blacks().rooks.Count();
    const int32_t bq = pos.Blacks().queens.Count();

    // piece values
    {
        inputs.emplace_back(offset++, mg * (wp - bp));
        inputs.emplace_back(offset++, eg * (wp - bp));
        inputs.emplace_back(offset++, mg * (wn - bn));
        inputs.emplace_back(offset++, eg * (wn - bn));
        inputs.emplace_back(offset++, mg * (wb - bb));
        inputs.emplace_back(offset++, eg * (wb - bb));
        inputs.emplace_back(offset++, mg * (wr - br));
        inputs.emplace_back(offset++, eg * (wr - br));
        inputs.emplace_back(offset++, mg * (wq - bq));
        inputs.emplace_back(offset++, eg * (wq - bq));
    }

    // piece-square tables
    {
        const auto writePieceFeatures = [&](const Bitboard bitboard, const Color color) INLINE_LAMBDA
        {
            bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                inputs.emplace_back(offset + 2 * square + 0, (color == Color::White ? 1.0f : -1.0f) * mg);
                inputs.emplace_back(offset + 2 * square + 1, (color == Color::White ? 1.0f : -1.0f) * eg);
            });
        };

        const auto writePawnFeatures = [&](const Bitboard bitboard, const Color color) INLINE_LAMBDA
        {
            // pawns cannot stand on first or last rank
            constexpr Bitboard mask = ~(Bitboard::RankBitboard<0>() | Bitboard::RankBitboard<7>());
            (bitboard & mask).Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                inputs.emplace_back(offset + 2 * (square - 8u) + 0, (color == Color::White ? 1.0f : -1.0f) * mg);
                inputs.emplace_back(offset + 2 * (square - 8u) + 1, (color == Color::White ? 1.0f : -1.0f) * eg);
            });
        };

        writePawnFeatures(pos.Whites().pawns, Color::White);
        writePawnFeatures(pos.Blacks().pawns.MirroredVertically(), Color::Black);
        offset += 2 * 48;

        writePieceFeatures(pos.Whites().knights, Color::White);
        writePieceFeatures(pos.Blacks().knights.MirroredVertically(), Color::Black);
        offset += 2 * 64;

        writePieceFeatures(pos.Whites().bishops, Color::White);
        writePieceFeatures(pos.Blacks().bishops.MirroredVertically(), Color::Black);
        offset += 2 * 64;

        writePieceFeatures(pos.Whites().rooks, Color::White);
        writePieceFeatures(pos.Blacks().rooks.MirroredVertically(), Color::Black);
        offset += 2 * 64;

        writePieceFeatures(pos.Whites().queens, Color::White);
        writePieceFeatures(pos.Blacks().queens.MirroredVertically(), Color::Black);
        offset += 2 * 64;

        inputs.emplace_back(offset + 2 * FirstBitSet(pos.Whites().king) + 0, mg);
        inputs.emplace_back(offset + 2 * FirstBitSet(pos.Whites().king) + 1, eg);
        inputs.emplace_back(offset + 2 * FirstBitSet(pos.Blacks().king.MirroredVertically()) + 0, -mg);
        inputs.emplace_back(offset + 2 * FirstBitSet(pos.Blacks().king.MirroredVertically()) + 1, -eg);
        offset += 2 * 64;
    }

    // bishop pair
    {
        int32_t bishopPair = 0;
        if ((pos.Whites().bishops & Bitboard::LightSquares()) && (pos.Whites().bishops & Bitboard::DarkSquares())) bishopPair += 1;
        if ((pos.Blacks().bishops & Bitboard::LightSquares()) && (pos.Blacks().bishops & Bitboard::DarkSquares())) bishopPair -= 1;
        if (bishopPair)
        {
            inputs.emplace_back(offset + 0, (float)bishopPair * mg);
            inputs.emplace_back(offset + 1, (float)bishopPair * eg);
        }
        offset += 2;
    }

#ifdef USE_MOBILITY

    // mobility
    {
        const Bitboard blockers = pos.Occupied();
        const Bitboard whitePawnsAttacks = Bitboard::GetPawnAttacks<Color::White>(pos.Whites().pawns);
        const Bitboard blackPawnsAttacks = Bitboard::GetPawnAttacks<Color::Black>(pos.Blacks().pawns);

        pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GetKnightAttacks(Square(square)) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, eg);
        });
        pos.Blacks().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GetKnightAttacks(Square(square)) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, -mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, -eg);
        });
        offset += 2 * 9;

        pos.Whites().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateBishopAttacks(Square(square), blockers) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, eg);
        });
        pos.Blacks().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateBishopAttacks(Square(square), blockers) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, -mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, -eg);
        });
        offset += 2 * 14;

        pos.Whites().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateRookAttacks(Square(square), blockers) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, eg);
        });
        pos.Blacks().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateRookAttacks(Square(square), blockers) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, -mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, -eg);
        });
        offset += 2 * 15;

        pos.Whites().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateQueenAttacks(Square(square), blockers) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, eg);
        });
        pos.Blacks().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateQueenAttacks(Square(square), blockers) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.emplace_back(offset + 2 * attacks.Count() + 0, -mg);
            inputs.emplace_back(offset + 2 * attacks.Count() + 1, -eg);
        });
        offset += 2 * 28;
    }
#endif // USE_MOBILITY

#ifdef USE_KING_TROPISM
    // king tropism
    {
        for (uint32_t pieceType = 0; pieceType < 5; ++pieceType)
        {
            const Bitboard w = pos.Whites().GetPieceBitBoard((Piece)(pieceType + (uint32_t)Piece::Pawn));
            const Bitboard b = pos.Blacks().GetPieceBitBoard((Piece)(pieceType + (uint32_t)Piece::Pawn));
            w.Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                const uint32_t d = Square::Distance(whiteKingSq, Square(square)) - 1;
                inputs.emplace_back(offset + 2 * d + 0, mg);
                inputs.emplace_back(offset + 2 * d + 1, eg);
            });
            b.Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                const uint32_t d = Square::Distance(blackKingSq, Square(square)) - 1;
                inputs.emplace_back(offset + 2 * d + 0, -mg);
                inputs.emplace_back(offset + 2 * d + 1, -eg);
            });
            offset += 2 * 7;
        }

        for (uint32_t pieceType = 0; pieceType < 5; ++pieceType)
        {
            const Bitboard w = pos.Whites().GetPieceBitBoard((Piece)(pieceType + (uint32_t)Piece::Pawn));
            const Bitboard b = pos.Blacks().GetPieceBitBoard((Piece)(pieceType + (uint32_t)Piece::Pawn));
            w.Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                const uint32_t d = Square::Distance(blackKingSq, Square(square)) - 1;
                inputs.emplace_back(offset + 2 * d + 0, -mg);
                inputs.emplace_back(offset + 2 * d + 1, -eg);
            });
            b.Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                const uint32_t d = Square::Distance(whiteKingSq, Square(square)) - 1;
                inputs.emplace_back(offset + 2 * d + 0, mg);
                inputs.emplace_back(offset + 2 * d + 1, eg);
            });
            offset += 2 * 7;
        }
    }
#endif // USE_KING_TROPISM

#ifdef USE_PASSED_PAWNS
    // passed pawns
    {
        pos.Whites().pawns.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            if (IsPassedPawn(Square(square), pos.Whites().pawns, pos.Blacks().pawns))
            {
                const uint32_t rank = Square(square).Rank();
                ASSERT(rank > 0 && rank < 6);
                inputs.emplace_back(offset + 2 * (rank - 1) + 0, mg);
                inputs.emplace_back(offset + 2 * (rank - 1) + 1, eg);
            }
        });

        const Bitboard whitesFlipped = pos.Whites().pawns.MirroredVertically();
        const Bitboard blacksFlipped = pos.Blacks().pawns.MirroredVertically();

        blacksFlipped.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            if (IsPassedPawn(Square(square), blacksFlipped, whitesFlipped))
            {
                const uint32_t rank = Square(square).Rank();
                ASSERT(rank > 0 && rank < 6);
                inputs.emplace_back(offset + 2 * (rank - 1) + 0, -mg);
                inputs.emplace_back(offset + 2 * (rank - 1) + 1, -eg);
            }
        });

        offset += 2 * 5;
    }
#endif // USE_PASSED_PAWNS

    outVector.CombineSparseInputs();

    ASSERT(offset == cNumNetworkInputs);
}

static void PrintPieceSquareTableWeigts(const nn::NeuralNetwork& nn)
{
    const float* weights = nn.layers[0].weights.data();

    uint32_t offset = 0;

    const auto printValue = [&]()
    {
        std::cout << "S(" << std::right
            << std::fixed << std::setw(4) << int32_t(c_nnOutputToCentiPawns * weights[offset++]) << ","
            << std::fixed << std::setw(4) << int32_t(c_nnOutputToCentiPawns * weights[offset++]) << "), ";
    };

    const auto printPieceWeights = [&](const char* name)
    {
        std::cout << name << std::endl;

        /*
        float avgMG = 0.0f;
        float avgEG = 0.0f;
        for (uint32_t i = 0; i < 64; ++i)
        {
            for (uint32_t file = 0; file < 8; file++)
            {
                avgMG += weights[offset + 2 * i + 0];
                avgEG += weights[offset + 2 * i + 1];
            }
        }
        avgMG /= 64.0f;
        avgEG /= 64.0f;
        */

        for (uint32_t rank = 0; rank < 8; ++rank)
        {
            std::cout << "    ";
            for (uint32_t file = 0; file < 8; file++)
            {
                const float weightMG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (8 * rank + file) + 0]));
                const float weightEG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (8 * rank + file) + 1]));
                // << std::fixed << std::setw(4)
                std::cout << std::right << "S(" << std::fixed << std::setw(4) << int32_t(weightMG) << "," << std::fixed << std::setw(4) << int32_t(weightEG) << "), ";
            }
            std::cout << std::endl;
        }
        offset += 2 * 64;

        std::cout << std::endl;
    };

    const auto writePawnWeights = [&](const char* name)
    {
        std::cout << name << std::endl;

        /*
        float avgMG = 0.0f;
        float avgEG = 0.0f;
        for (uint32_t rank = 1; rank < 7; ++rank)
        {
            for (uint32_t file = 0; file < 8; file++)
            {
                avgMG += weights[offset + 2 * (8 * (rank - 1) + file) + 0];
                avgEG += weights[offset + 2 * (8 * (rank - 1) + file) + 1];
            }
        }
        avgMG /= 48.0f;
        avgEG /= 48.0f;
        std::cout << "Average: (" << int32_t(c_nnOutputToCentiPawns * avgMG) << ", " << int32_t(c_nnOutputToCentiPawns * avgEG) << ")" << std::endl;
        */

        // pawns cannot stand on first or last rank
        for (uint32_t rank = 1; rank < 7; ++rank)
        {
            std::cout << "    ";
            for (uint32_t file = 0; file < 8; file++)
            {
                const float weightMG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (8 * (rank - 1) + file) + 0]));
                const float weightEG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (8 * (rank - 1) + file) + 1]));
                std::cout << std::right << "S(" << std::fixed << std::setw(4) << int32_t(weightMG) << "," << std::fixed << std::setw(4) << int32_t(weightEG) << "), ";
            }
            std::cout << std::endl;
        }
        offset += 2 * 48;

        std::cout << std::endl;
    };

    std::cout << "Pawn value:       "; printValue(); std::cout << std::endl;
    std::cout << "Knight value:     "; printValue(); std::cout << std::endl;
    std::cout << "Bishop value:     "; printValue(); std::cout << std::endl;
    std::cout << "Rook value:       "; printValue(); std::cout << std::endl;
    std::cout << "Queen value:      "; printValue(); std::cout << std::endl;
    std::cout << std::endl;

    writePawnWeights("Pawn");
    printPieceWeights("Knights");
    printPieceWeights("Bishop");
    printPieceWeights("Rook");
    printPieceWeights("Queen");
    printPieceWeights("King");

    std::cout << "Bishop Pair:           S(" << int32_t(c_nnOutputToCentiPawns * weights[offset++]) << ", " << int32_t(c_nnOutputToCentiPawns * weights[offset++]) << ")" << std::endl;

#ifdef USE_MOBILITY
    std::cout << "Knight mobility bonus: "; for (uint32_t i = 0; i < 9; ++i)   printValue(); std::cout << std::endl;
    std::cout << "Bishop mobility bonus: "; for (uint32_t i = 0; i < 14; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Rook mobility bonus:   "; for (uint32_t i = 0; i < 15; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Queen mobility bonus:  "; for (uint32_t i = 0; i < 28; ++i)  printValue(); std::cout << std::endl;
    std::cout << std::endl;
#endif // USE_MOBILITY

#ifdef USE_KING_TROPISM
    std::cout << "Pawn vs. King (same color) distance bonus:       "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Knight vs. King (same color) distance bonus:     "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Bishop vs. King (same color) distance bonus:     "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Rook vs. King (same color) distance bonus:       "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Queen vs. King (same color) distance bonus:      "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Pawn vs. King (opposite color)  distance bonus:  "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Knight vs. King (opposite color) distance bonus: "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Bishop vs. King (opposite color) distance bonus: "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Rook vs. King (opposite color distance bonus:    "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Queen vs. King (opposite color) distance bonus:  "; for (uint32_t i = 0; i < 7; ++i)  printValue(); std::cout << std::endl;
    std::cout << std::endl;
#endif // USE_KING_TROPISM

#ifdef USE_PASSED_PAWNS
    std::cout << "Passed pawns bonus:                              ";
    for (uint32_t i = 0; i < 5; ++i)  printValue(); std::cout << std::endl;
    std::cout << std::endl;
#endif // USE_PASSED_PAWNS

    std::cout << "Offset: " << int32_t(c_nnOutputToCentiPawns * weights[offset]) << std::endl;

    ASSERT(offset == cNumNetworkInputs);
}

bool TrainPieceSquareTables()
{
    std::vector<PositionEntry> entries;
    LoadAllPositions(entries);

    if (entries.empty())
    {
        return false;
    }

    std::cout << "Training with " << entries.size() << " positions" << std::endl;

    nn::NeuralNetwork materialNetwork;
    materialNetwork.Init(cNumNetworkInputs, { 32, 32, 1 }, nn::ActivationFunction::Sigmoid);

    nn::NeuralNetwork network;
    network.Init(cNumNetworkInputs, { 1 }, nn::ActivationFunction::Sigmoid);

    nn::NeuralNetworkRunContext networkRunCtx;
    nn::NeuralNetworkRunContext materialNetworkRunCtx;
    networkRunCtx.Init(network);
    materialNetworkRunCtx.Init(materialNetwork);

    nn::NeuralNetworkTrainer trainer;

    // reset weights
    memset(network.layers[0].weights.data(), 0, sizeof(float) * (cNumNetworkInputs + 1));

    network.layers[0].weights[0] = (float)c_pawnValue.mg / c_nnOutputToCentiPawns;
    network.layers[0].weights[1] = (float)c_pawnValue.eg / c_nnOutputToCentiPawns;
    network.layers[0].weights[2] = (float)c_knightValue.mg / c_nnOutputToCentiPawns;
    network.layers[0].weights[3] = (float)c_knightValue.eg / c_nnOutputToCentiPawns;
    network.layers[0].weights[4] = (float)c_bishopValue.mg / c_nnOutputToCentiPawns;
    network.layers[0].weights[5] = (float)c_bishopValue.eg / c_nnOutputToCentiPawns;
    network.layers[0].weights[6] = (float)c_rookValue.mg / c_nnOutputToCentiPawns;
    network.layers[0].weights[7] = (float)c_rookValue.eg / c_nnOutputToCentiPawns;
    network.layers[0].weights[8] = (float)c_queenValue.mg / c_nnOutputToCentiPawns;
    network.layers[0].weights[9] = (float)c_queenValue.eg / c_nnOutputToCentiPawns;

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

        const float learningRate = std::max(0.05f, 1.0f / (1.0f + 0.001f * iteration));

        TimePoint startTime = TimePoint::GetCurrent();
        trainer.Train(network, trainingSet, cBatchSize, learningRate, false);
        TimePoint endTime = TimePoint::GetCurrent();
        const float trainingTime = (endTime - startTime).ToSeconds();

        // normalize king weights
        {
            float kingAvgMG = 0.0f;
            float kingAvgEG = 0.0f;
            for (uint32_t i = 4 * 64 + 48; i < 5 * 64 + 48; ++i)
            {
                kingAvgMG += network.layers[0].weights[2 * i + 0];
                kingAvgEG += network.layers[0].weights[2 * i + 1];
            }
            kingAvgMG /= 64.0f;
            kingAvgEG /= 64.0f;
            for (uint32_t i = 4 * 64 + 48; i < 5 * 64 + 48; ++i)
            {
                network.layers[0].weights[2 * i + 0] -= kingAvgMG;
                network.layers[0].weights[2 * i + 1] -= kingAvgEG;
            }
        }

        // decay all weights so they don't explode to infinite
        for (uint32_t i = 10; i < cNumNetworkInputs; ++i)
        {
            network.layers[0].weights[i] *= 0.99f;
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

            const nn::Values& networkOutput = network.Run((uint32_t)validationVector.sparseInputs.size(), validationVector.sparseInputs.data(), networkRunCtx);

            const float expectedValue = validationVector.output[0];

            if (i == 0)
            {
                std::cout << pos.ToFEN() << std::endl << pos.Print();
                std::cout << "Value:    " << networkOutput[0] << std::endl;
                std::cout << "Expected: " << expectedValue << std::endl << std::endl;
                PrintPieceSquareTableWeigts(network);
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

        std::cout << "Training time:    " << (1000000.0f * trainingTime / trainingSet.size()) << " us/pos" << std::endl;
    }

    return true;
}
