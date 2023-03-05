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

#define USE_PSQT
#define USE_BISHOP_PAIR
//#define USE_CASTLING_RIGHTS
//#define USE_IMBALANCE
//#define USE_MOBILITY
//#define USE_PAWN_STRUCTURE
//#define USE_PASSED_PAWNS

using namespace threadpool;

static const uint32_t cMaxIterations = 100000000;
static const uint32_t cNumTrainingVectorsPerIteration = 1024 * 1024;
static const uint32_t cNumValidationVectorsPerIteration = 256 * 1024;
static const uint32_t cBatchSize = 8192;

static const uint32_t cNumNetworkInputs =
    2 * 5                       // piece values
#ifdef USE_PSQT
    + 2 * 32 * 64 * 10          // king-relative PSQT
#endif
#ifdef USE_BISHOP_PAIR
    + 2                         // bishop pair
#endif
#ifdef USE_IMBALANCE
    + 2 * 55
#endif
#ifdef USE_CASTLING_RIGHTS
    + 2
#endif
#ifdef USE_MOBILITY
    + 2 * (9 + 9 + 14 + 15 + 28)
#endif
#ifdef USE_PAWN_STRUCTURE
    + 2 * 48 * 48 * 2
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

    outVector.inputMode = nn::InputMode::Sparse;
    outVector.sparseInputs.clear();
    std::vector<nn::ActiveFeature>& inputs = outVector.sparseInputs;

    uint32_t offset = 0;

    const float mg = GetGamePhase(pos);
    const float eg = 1.0f - mg;

    const Square whiteKingSq(FirstBitSet(pos.Whites().king));
    const Square blackKingSq(FirstBitSet(pos.Blacks().king));
    const Square whiteKingSqFlipped = whiteKingSq.File() >= 4 ? whiteKingSq.FlippedFile() : whiteKingSq;
    const Square blackKingSqFlipped = blackKingSq.File() >= 4 ? blackKingSq.FlippedRank().FlippedFile() : blackKingSq.FlippedRank();
    const Bitboard blackPawnsFlipped = pos.Blacks().pawns.MirroredVertically();

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
        inputs.push_back(nn::ActiveFeature{offset++, mg * (wp - bp)});
        inputs.push_back(nn::ActiveFeature{offset++, eg * (wp - bp)});
        inputs.push_back(nn::ActiveFeature{offset++, mg * (wn - bn)});
        inputs.push_back(nn::ActiveFeature{offset++, eg * (wn - bn)});
        inputs.push_back(nn::ActiveFeature{offset++, mg * (wb - bb)});
        inputs.push_back(nn::ActiveFeature{offset++, eg * (wb - bb)});
        inputs.push_back(nn::ActiveFeature{offset++, mg * (wr - br)});
        inputs.push_back(nn::ActiveFeature{offset++, eg * (wr - br)});
        inputs.push_back(nn::ActiveFeature{offset++, mg * (wq - bq)});
        inputs.push_back(nn::ActiveFeature{offset++, eg * (wq - bq)});
    }

#ifdef USE_PSQT
    // piece-square tables
    {
        const auto writePieceFeatures = [&](const Bitboard bitboard, const Color color) INLINE_LAMBDA
        {
            bitboard.Iterate([&](uint32_t squareIndex) INLINE_LAMBDA
            {
                const Square square(squareIndex);

                ASSERT(squareIndex != whiteKingSq.Index());
                ASSERT(squareIndex != blackKingSq.Index());

                // relative to our king
                {
                    const uint32_t kingSquareIndex = 4 * whiteKingSqFlipped.Rank() + whiteKingSqFlipped.File();
                    const uint32_t featureIndex =
                        32 * 64 * (uint32_t)color +
                        64 * kingSquareIndex +
                        (whiteKingSq.File() >= 4 ? square.FlippedFile().Index() : square.Index());

                    ASSERT(featureIndex < 32 * 64 * 2);
                    inputs.push_back(nn::ActiveFeature{offset + 2 * featureIndex + 0, mg});
                    inputs.push_back(nn::ActiveFeature{offset + 2 * featureIndex + 1, eg});
                }

                // relative to their king
                {
                    const uint32_t kingSquareIndex = 4 * blackKingSqFlipped.Rank() + blackKingSqFlipped.File();
                    const uint32_t featureIndex =
                        32 * 64 * (uint32_t)GetOppositeColor(color) +
                        64 * kingSquareIndex +
                        (blackKingSq.File() >= 4 ? square.FlippedRank().FlippedFile().Index() : square.FlippedRank().Index());

                    ASSERT(featureIndex < 32 * 64 * 2);
                    inputs.push_back(nn::ActiveFeature{offset + 2 * featureIndex + 0, -mg});
                    inputs.push_back(nn::ActiveFeature{offset + 2 * featureIndex + 1, -eg});
                }
            });
        };

        writePieceFeatures(pos.Whites().pawns, Color::White);
        writePieceFeatures(pos.Blacks().pawns, Color::Black);
        offset += 2 * 32 * 64 * 2;

        writePieceFeatures(pos.Whites().knights, Color::White);
        writePieceFeatures(pos.Blacks().knights, Color::Black);
        offset += 2 * 32 * 64 * 2;

        writePieceFeatures(pos.Whites().bishops, Color::White);
        writePieceFeatures(pos.Blacks().bishops, Color::Black);
        offset += 2 * 32 * 64 * 2;

        writePieceFeatures(pos.Whites().rooks, Color::White);
        writePieceFeatures(pos.Blacks().rooks, Color::Black);
        offset += 2 * 32 * 64 * 2;

        writePieceFeatures(pos.Whites().queens, Color::White);
        writePieceFeatures(pos.Blacks().queens, Color::Black);
        offset += 2 * 32 * 64 * 2;
    }
#endif // USE_PSQT

#ifdef USE_BISHOP_PAIR
    {
        int32_t bishopPair = 0;
        if ((pos.Whites().bishops & Bitboard::LightSquares()) && (pos.Whites().bishops & Bitboard::DarkSquares())) bishopPair += 1;
        if ((pos.Blacks().bishops & Bitboard::LightSquares()) && (pos.Blacks().bishops & Bitboard::DarkSquares())) bishopPair -= 1;
        if (bishopPair)
        {
            inputs.push_back(nn::ActiveFeature{offset + 0, (float)bishopPair * mg});
            inputs.push_back(nn::ActiveFeature{offset + 1, (float)bishopPair * eg});
        }
        offset += 2;
    }
#endif // USE_BISHOP_PAIR

#ifdef USE_IMBALANCE
    {
        const int32_t matCount[] = { wp, wn, wb, wr, wq, bp, bn, bb, br, bq };
        for (uint32_t i = 0; i < 10; ++i)
        {
            for (uint32_t j = 0; j < i; ++j)
            {
                inputs.push_back(nn::ActiveFeature{offset++, mg * matCount[i] * matCount[j]});
                inputs.push_back(nn::ActiveFeature{offset++, eg * matCount[i] * matCount[j]});
            }

            inputs.push_back(nn::ActiveFeature{offset++, mg * matCount[i] * matCount[i]});
            inputs.push_back(nn::ActiveFeature{offset++, eg * matCount[i] * matCount[i]});
        }
    }
#endif // USE_IMBALANCE

#ifdef USE_CASTLING_RIGHTS
    {
        const int32_t numCastlingRights = (int32_t)PopCount(pos.GetWhitesCastlingRights()) - (int32_t)PopCount(pos.GetBlacksCastlingRights());
        if (numCastlingRights)
        {
            inputs.push_back(nn::ActiveFeature{offset + 0, (float)numCastlingRights * mg});
            inputs.push_back(nn::ActiveFeature{offset + 1, (float)numCastlingRights * eg});
        }
        offset += 2;
    }
#endif

#ifdef USE_MOBILITY

    // mobility
    {
        const Bitboard blockers = pos.Occupied();
        const Bitboard whitePawnsAttacks = Bitboard::GetPawnAttacks<Color::White>(pos.Whites().pawns);
        const Bitboard blackPawnsAttacks = Bitboard::GetPawnAttacks<Color::Black>(pos.Blacks().pawns);

        // king mobility
        {
            const Bitboard attacks = Bitboard::GetKingAttacks(pos.Whites().GetKingSquare()) & ~pos.Occupied() & ~blackPawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, eg});
        }
        {
            const Bitboard attacks = Bitboard::GetKingAttacks(pos.Blacks().GetKingSquare()) & ~pos.Occupied() & ~whitePawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, -mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, -eg});
        }
        offset += 2 * 9;

        pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GetKnightAttacks(Square(square)) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, eg});
        });
        pos.Blacks().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GetKnightAttacks(Square(square)) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, -mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, -eg});
        });
        offset += 2 * 9;

        pos.Whites().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateBishopAttacks(Square(square), blockers) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, eg});
        });
        pos.Blacks().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateBishopAttacks(Square(square), blockers) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, -mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, -eg});
        });
        offset += 2 * 14;

        pos.Whites().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateRookAttacks(Square(square), blockers) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, eg});
        });
        pos.Blacks().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateRookAttacks(Square(square), blockers) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, -mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, -eg});
        });
        offset += 2 * 15;

        pos.Whites().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateQueenAttacks(Square(square), blockers) & ~pos.Whites().Occupied() & ~blackPawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, eg});
        });
        pos.Blacks().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            const Bitboard attacks = Bitboard::GenerateQueenAttacks(Square(square), blockers) & ~pos.Blacks().Occupied() & ~whitePawnsAttacks;
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 0, -mg});
            inputs.push_back(nn::ActiveFeature{offset + 2 * attacks.Count() + 1, -eg});
        });
        offset += 2 * 28;
    }
#endif // USE_MOBILITY

#ifdef USE_PAWN_STRUCTURE
    {
        // white pawns vs. white pawns
        pos.Whites().pawns.Iterate([&](uint32_t squareA) INLINE_LAMBDA
        {
            const uint32_t pawnOffsetA = 8 * ((squareA / 8) - 1) + (squareA % 8);
            ASSERT(pawnOffsetA < 48);

            pos.Whites().pawns.Iterate([&](uint32_t squareB) INLINE_LAMBDA
            {
                const uint32_t pawnOffsetB = 8 * ((squareB / 8) - 1) + (squareB % 8);
                ASSERT(pawnOffsetB < 48);

                if (pawnOffsetA < pawnOffsetB)
                {
                    inputs.push_back(nn::ActiveFeature{offset + 2 * (48 * pawnOffsetA + pawnOffsetB) + 0, mg});
                    inputs.push_back(nn::ActiveFeature{offset + 2 * (48 * pawnOffsetA + pawnOffsetB) + 1, eg});
                }
            });
        });

        // black pawns vs. black pawns
        blackPawnsFlipped.Iterate([&](uint32_t squareA) INLINE_LAMBDA
        {
            const uint32_t pawnOffsetA = 8 * ((squareA / 8) - 1) + (squareA % 8); ASSERT(pawnOffsetA < 48);

            blackPawnsFlipped.Iterate([&](uint32_t squareB) INLINE_LAMBDA
            {
                const uint32_t pawnOffsetB = 8 * ((squareB / 8) - 1) + (squareB % 8);
                ASSERT(pawnOffsetB < 48);

                if (pawnOffsetA < pawnOffsetB)
                {
                    inputs.push_back(nn::ActiveFeature{offset + 2 * (48 * pawnOffsetA + pawnOffsetB) + 0, -mg});
                    inputs.push_back(nn::ActiveFeature{offset + 2 * (48 * pawnOffsetA + pawnOffsetB) + 1, -eg});
                }
            });
        });

        offset += 2 * 48 * 48;

        // white pawns vs. black pawns
        pos.Whites().pawns.Iterate([&](uint32_t squareA) INLINE_LAMBDA
        {
            const uint32_t pawnOffsetA = 8 * ((squareA / 8) - 1) + (squareA % 8);
            ASSERT(pawnOffsetA < 48);

            pos.Blacks().pawns.Iterate([&](uint32_t squareB) INLINE_LAMBDA
            {
                const uint32_t pawnOffsetB = 8 * ((squareB / 8) - 1) + (squareB % 8);
                ASSERT(pawnOffsetB < 48);

                inputs.push_back(nn::ActiveFeature{offset + 2 * (48 * pawnOffsetA + pawnOffsetB) + 0, mg});
                inputs.push_back(nn::ActiveFeature{offset + 2 * (48 * pawnOffsetA + pawnOffsetB) + 1, eg});
            });
        });

        offset += 2 * 48 * 48;
    }

#endif // USE_PAWN_STRUCTURE

#ifdef USE_PASSED_PAWNS
    {
        pos.Whites().pawns.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            if (IsPassedPawn(Square(square), pos.Whites().pawns, pos.Blacks().pawns))
            {
                const uint32_t rank = Square(square).Rank();
                ASSERT(rank > 0 && rank < 6);
                inputs.push_back(nn::ActiveFeature{offset + 2 * (rank - 1) + 0, mg});
                inputs.push_back(nn::ActiveFeature{offset + 2 * (rank - 1) + 1, eg});
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
                inputs.push_back(nn::ActiveFeature{offset + 2 * (rank - 1) + 0, -mg});
                inputs.push_back(nn::ActiveFeature{offset + 2 * (rank - 1) + 1, -eg});
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
    const float* weights = nn.layers[0].variants[0].weights.data();

    uint32_t offset = 0;

    std::ofstream psqtFile("generatedPSQT.hpp");
    std::ofstream pawnStructureTableFile("generatedPawnStructureTable.hpp");

    const auto printValue = [&]()
    {
        std::cout << "S(" << std::right
            << std::fixed << std::setw(4) << int32_t(c_nnOutputToCentiPawns * weights[offset]) << ","
            << std::fixed << std::setw(4) << int32_t(c_nnOutputToCentiPawns * weights[offset + 1]) << "), ";
        offset += 2;
    };

    // piece values
    {
        std::cout << "Pawn value:       "; printValue(); std::cout << std::endl;
        std::cout << "Knight value:     "; printValue(); std::cout << std::endl;
        std::cout << "Bishop value:     "; printValue(); std::cout << std::endl;
        std::cout << "Rook value:       "; printValue(); std::cout << std::endl;
        std::cout << "Queen value:      "; printValue(); std::cout << std::endl;
        std::cout << std::endl;
    }

#ifdef USE_PSQT
    {
        const auto printPieceWeights = [&](uint32_t kingSquareIndex, uint32_t pieceType, const char* name)
        {
            psqtFile << "\t// " << name << std::endl;
            psqtFile << "\t{" << std::endl;

            const uint32_t featureOffset = 32 * 64 * pieceType + 64 * kingSquareIndex;
            ASSERT(featureOffset < 32 * 64 * 10);

            for (uint32_t rank = 0; rank < 8; ++rank)
            {
                psqtFile << "\t\t";
                for (uint32_t file = 0; file < 8; file++)
                {
                    const float weightMG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (featureOffset + 8 * rank + file) + 0]));
                    const float weightEG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (featureOffset + 8 * rank + file) + 1]));
                    // << std::fixed << std::setw(4)
                    psqtFile << std::right << std::fixed << std::setw(4) << int32_t(weightMG) << "," << std::fixed << std::setw(4) << int32_t(weightEG) << ", ";
                }
                psqtFile << std::endl;
            }

            psqtFile << "\t}," << std::endl;
        };

        for (uint8_t kingSqIndex = 0; kingSqIndex < 32; ++kingSqIndex)
        {
            const uint8_t kingRank = kingSqIndex / 4;
            const uint8_t kingFile = kingSqIndex % 4;

            psqtFile << "// king on " << Square(kingFile, kingRank).ToString() << std::endl;
            psqtFile << "{" << std::endl;

            printPieceWeights(kingSqIndex, 0, "Our Pawns");
            printPieceWeights(kingSqIndex, 1, "Their Pawns");
            printPieceWeights(kingSqIndex, 2, "Our Knights");
            printPieceWeights(kingSqIndex, 3, "Their Knights");
            printPieceWeights(kingSqIndex, 4, "Our Bishops");
            printPieceWeights(kingSqIndex, 5, "Their Bishops");
            printPieceWeights(kingSqIndex, 6, "Our Rooks");
            printPieceWeights(kingSqIndex, 7, "Their Rooks");
            printPieceWeights(kingSqIndex, 8, "Our Queens");
            printPieceWeights(kingSqIndex, 9, "Their Queens");

            psqtFile << "}," << std::endl << std::endl;
        }

        offset += 10 * 32 * 64 * 2;
    }
#endif // USE_PSQT

#ifdef USE_BISHOP_PAIR
    {
        std::cout << "Bishop Pair:           "; printValue(); std::cout << std::endl;
    }
#endif // USE_BISHOP_PAIR

#ifdef USE_IMBALANCE
    {
        std::cout << "Imbalance table:" << std::endl;
        for (uint32_t i = 0; i < 10; ++i)
        {
            for (uint32_t j = 0; j <= i; ++j)
            {
                printValue();
            }
            std::cout << std::endl;
        }
    }
#endif // USE_IMBALANCE

#ifdef USE_CASTLING_RIGHTS
    std::cout << "Castling Rights:       "; printValue(); std::cout << std::endl;
#endif // USE_CASTLING_RIGHTS

#ifdef USE_MOBILITY
    std::cout << "King mobility bonus:   "; for (uint32_t i = 0; i < 9; ++i)   printValue(); std::cout << std::endl;
    std::cout << "Knight mobility bonus: "; for (uint32_t i = 0; i < 9; ++i)   printValue(); std::cout << std::endl;
    std::cout << "Bishop mobility bonus: "; for (uint32_t i = 0; i < 14; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Rook mobility bonus:   "; for (uint32_t i = 0; i < 15; ++i)  printValue(); std::cout << std::endl;
    std::cout << "Queen mobility bonus:  "; for (uint32_t i = 0; i < 28; ++i)  printValue(); std::cout << std::endl;
    std::cout << std::endl;
#endif // USE_MOBILITY

#ifdef USE_PAWN_STRUCTURE
    {
        const auto printPawnWeights = [&]()
        {
            pawnStructureTableFile << "\t{" << std::endl;

            for (uint32_t rank = 0; rank < 6; ++rank)
            {
                pawnStructureTableFile << "\t\t";
                for (uint32_t file = 0; file < 8; file++)
                {
                    const float weightMG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (8 * rank + file) + 0]));
                    const float weightEG = std::round(c_nnOutputToCentiPawns * (weights[offset + 2 * (8 * rank + file) + 1]));
                    // << std::fixed << std::setw(4)
                    pawnStructureTableFile << std::right << std::fixed << std::setw(4) << int32_t(weightMG) << "," << std::fixed << std::setw(4) << int32_t(weightEG) << ", ";
                }
                pawnStructureTableFile << std::endl;
            }

            pawnStructureTableFile << "\t}," << std::endl;
        };

        for (uint8_t pawnIndex = 0; pawnIndex < 48; ++pawnIndex)
        {
            const uint8_t pawnRank = 1 + pawnIndex / 8;
            const uint8_t pawnFile = pawnIndex % 8;
            pawnStructureTableFile << "// white pawn on " << Square(pawnFile, pawnRank).ToString() << std::endl;
            printPawnWeights();
            offset += 2 * 48;
        }

        for (uint8_t pawnIndex = 0; pawnIndex < 48; ++pawnIndex)
        {
            const uint8_t pawnRank = 1 + pawnIndex / 8;
            const uint8_t pawnFile = pawnIndex % 8;
            pawnStructureTableFile << "// white pawn on " << Square(pawnFile, pawnRank).ToString() << std::endl;
            printPawnWeights();
            offset += 2 * 48;
        }
    }
#endif // USE_PAWN_STRUCTURE

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

    // clear weights
    {
        float* weights = network.layers[0].variants[0].weights.data();
        memset(weights, 0, sizeof(float) * (cNumNetworkInputs + 1));
    }

    // reset weights
    const auto initPieceValueWeights = [&network]()
    {
        float* weights = network.layers[0].variants[0].weights.data();

        weights[0] = (float)c_pawnValue.mg / c_nnOutputToCentiPawns;
        weights[1] = (float)c_pawnValue.eg / c_nnOutputToCentiPawns;
        weights[2] = (float)c_knightValue.mg / c_nnOutputToCentiPawns;
        weights[3] = (float)c_knightValue.eg / c_nnOutputToCentiPawns;
        weights[4] = (float)c_bishopValue.mg / c_nnOutputToCentiPawns;
        weights[5] = (float)c_bishopValue.eg / c_nnOutputToCentiPawns;
        weights[6] = (float)c_rookValue.mg / c_nnOutputToCentiPawns;
        weights[7] = (float)c_rookValue.eg / c_nnOutputToCentiPawns;
        weights[8] = (float)c_queenValue.mg / c_nnOutputToCentiPawns;
        weights[9] = (float)c_queenValue.eg / c_nnOutputToCentiPawns;
    };

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<TrainingEntry> validationSet;
    std::vector<nn::TrainingVector> trainingBatch;
    validationSet.resize(cNumTrainingVectorsPerIteration);
    trainingBatch.resize(cNumTrainingVectorsPerIteration);

    uint32_t numTrainingVectorsPassed = 0;

    const auto generateTrainingEntry = [&](TrainingEntry& outEntry)
    {
        // pick random test entries
        std::uniform_int_distribution<size_t> distrib(0, entries.size() - 1);

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

        PositionToTrainingVector(pos, outEntry.trainingVector);
        outEntry.trainingVector.singleOutput = entry.score;
        outEntry.pos = pos;
    };

    const auto generateTrainingSet = [&](std::vector<TrainingEntry>& outEntries)
    {
        for (uint32_t i = 0; i < cNumTrainingVectorsPerIteration; ++i)
        {
            generateTrainingEntry(outEntries[i]);
        }
    };

    generateTrainingSet(validationSet);

    for (uint32_t iteration = 0; iteration < cMaxIterations; ++iteration)
    {
        TimePoint startTime = TimePoint::GetCurrent();

        const float learningRate = std::max(0.1f, 1.0f / (1.0f + 0.001f * iteration));

        // force fixed piece values
        initPieceValueWeights();

        // use validation set from previous iteration as training set in the current one
        for (size_t i = 0; i < trainingBatch.size(); ++i)
        {
            trainingBatch[i] = validationSet[i].trainingVector;
        }

        // validation vectors generation can be done in parallel with training
        Waitable waitable;
        {
            TaskBuilder taskBuilder(waitable);
            taskBuilder.ParallelFor("GenerateSet", cNumTrainingVectorsPerIteration, [&](const TaskContext&, uint32_t i)
            {
                generateTrainingEntry(validationSet[i]);
            });

            taskBuilder.Task("Train", [&](const TaskContext& ctx)
            {
                nn::TrainParams params;
                params.batchSize = cBatchSize;
                params.learningRate = learningRate;
                params.clampWeights = false;

                TaskBuilder taskBuilder{ ctx };
                trainer.Train(network, trainingBatch, params, &taskBuilder);
            });
        }
        waitable.Wait();

        numTrainingVectorsPassed += cNumTrainingVectorsPerIteration;

        float minError = std::numeric_limits<float>::max();
        float maxError = -std::numeric_limits<float>::max();
        float errorSum = 0.0f;
        for (uint32_t i = 0; i < cNumValidationVectorsPerIteration; ++i)
        {
            const std::vector<nn::ActiveFeature>& features = validationSet[i].trainingVector.sparseInputs;
            const nn::Values& networkOutput = network.Run(nn::NeuralNetwork::InputDesc(features), networkRunCtx);

            const float expectedValue = validationSet[i].trainingVector.singleOutput;

            if (i == 0)
            {
                std::cout << validationSet[i].pos.ToFEN() << std::endl << validationSet[i].pos.Print();
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
        errorSum = sqrtf(errorSum / cNumValidationVectorsPerIteration);

        float epoch = (float)numTrainingVectorsPassed / (float)entries.size();
        std::cout << std::right << std::fixed << std::setprecision(4) << epoch << " | ";
        std::cout << std::right << std::fixed << std::setprecision(4) << errorSum << " | ";
        std::cout << std::right << std::fixed << std::setprecision(4) << minError << " | ";
        std::cout << std::right << std::fixed << std::setprecision(4) << maxError << " | ";
        std::cout << std::endl;

        const float iterationTime = (TimePoint::GetCurrent() - startTime).ToSeconds();
        std::cout << "Iteration time:    " << (1000000.0f * iterationTime / cNumTrainingVectorsPerIteration) << " us/pos" << std::endl;
    }

    return true;
}
