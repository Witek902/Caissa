#include "Common.hpp"

#include "../backend/Position.hpp"
#include "../backend/Game.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Endgame.hpp"
#include "../backend/Tablebase.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <mutex>
#include <fstream>
#include <limits.h>

void GenerateEndgamePieceSquareTables()
{
    const Square blackKingSquares[] =
    {
        Square_a1,
        Square_b1, Square_b2,
        Square_c1, Square_c2, Square_c3,
        Square_d1, Square_d2, Square_d3, Square_d4,
    };

    const uint32_t numBlackKingPositions = 10;

    uint32_t successfullyProbed = 0;
    uint32_t maxDtz = 0;

    for (uint32_t blackSquareIndex = 0; blackSquareIndex < numBlackKingPositions; ++blackSquareIndex)
    {
        const Square blackKingSq = blackKingSquares[blackSquareIndex];

        int64_t whiteKingCounters[64] = { 0 };
        int64_t whiteKnightCounters[64] = { 0 };
        int64_t whiteBishopCounters[64] = { 0 };

        uint32_t maxNumPositions = 64 * 64 * 64;

        for (uint32_t posIndex = 0; posIndex < maxNumPositions; ++posIndex)
        {
            const Square whiteKingSq = (posIndex) & 0x3F;
            const Square whiteKnightSq = (posIndex >> 6) & 0x3F;
            const Square whiteBishopSq = (posIndex >> 12) & 0x3F;

            const Bitboard whiteKingAllowedSquares =
                ~blackKingSq.GetBitboard() &
                ~Bitboard::GetKingAttacks(blackKingSq);

            const Bitboard whiteKnightAllowedSquares =
                ~whiteKingSq.GetBitboard() &
                ~blackKingSq.GetBitboard();

            const Bitboard whiteBishopAllowedSquares =
                ~blackKingSq.GetBitboard() &
                ~whiteKingSq.GetBitboard() &
                ~whiteKnightSq.GetBitboard();

            if ((whiteKingSq.GetBitboard() & whiteKingAllowedSquares) == 0) continue;
            if ((whiteKnightSq.GetBitboard() & whiteKnightAllowedSquares) == 0) continue;
            if ((whiteBishopSq.GetBitboard() & whiteBishopAllowedSquares) == 0) continue;

            Position pos;
            pos.SetSideToMove(Color::Black);
            pos.SetPiece(blackKingSq, Piece::King, Color::Black);
            pos.SetPiece(whiteKingSq, Piece::King, Color::White);
            pos.SetPiece(whiteKnightSq, Piece::Rook, Color::Black);
            pos.SetPiece(whiteBishopSq, Piece::Queen, Color::White);
            ASSERT(pos.IsValid());

            Move bestMove;
            uint32_t dtz = UINT32_MAX;
            int32_t wdl = 0;
            bool probeResult = ProbeTablebase_Root(pos, bestMove, &dtz, &wdl);

            if (probeResult)
            {
                ASSERT(dtz < UINT8_MAX);

                successfullyProbed++;
                maxDtz = std::max(maxDtz, dtz);

                if (wdl == 0) dtz = 64;
                int32_t score = 64 - dtz; // (33 - dtz)* (33 - dtz);

                whiteKingCounters[whiteKingSq.Index()] += score;
                whiteKnightCounters[whiteKnightSq.Index()] += score;
                whiteBishopCounters[whiteBishopSq.Index()] += score;
            }
        }

        const auto printPST = [successfullyProbed](int64_t counters[64])
        {
            for (uint32_t rank = 0; rank < 8; ++rank)
            {
                for (uint32_t file = 0; file < 8; ++file)
                {
                    std::cout
                        << std::right << std::fixed << std::setprecision(3)
                        << 64.0f * float(counters[8 * rank + file]) / (float)successfullyProbed << "\t";
                }
                std::cout << std::endl;
            }
        };

        std::cout << std::endl << "Black king on: " << blackKingSq.ToString() << std::endl;
        std::cout << std::endl << "White king PST:" << std::endl;
        printPST(whiteKingCounters);
        std::cout << std::endl << "White knight PST:" << std::endl;
        printPST(whiteKnightCounters);
        std::cout << std::endl << "White bishop PST:" << std::endl;
        printPST(whiteBishopCounters);
    }

    std::cout << "Successfully probed: " << successfullyProbed << std::endl;
    std::cout << "Max DTZ:             " << maxDtz << std::endl;
}