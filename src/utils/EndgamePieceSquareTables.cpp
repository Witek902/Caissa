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

    MoveList moves;

    for (uint32_t blackSquareIndex = 0; blackSquareIndex < numBlackKingPositions; ++blackSquareIndex)
    {
        const Square blackKingSq = blackKingSquares[blackSquareIndex];

        int64_t whiteKingCounters[64] = { 0 };
        int64_t whiteQueenCounters[64] = { 0 };
        int64_t blackRookCounters[64] = { 0 };

        uint32_t maxNumPositions = 64 * 64 * 64;

        for (uint32_t posIndex = 0; posIndex < maxNumPositions; ++posIndex)
        {
            const Square whiteKingSq = (posIndex) & 0x3F;
            const Square blackRookSq = (posIndex >> 6) & 0x3F;
            const Square whiteQueenSq = (posIndex >> 12) & 0x3F;

            const Bitboard whiteKingAllowedSquares =
                ~blackKingSq.GetBitboard() &
                ~Bitboard::GetKingAttacks(blackKingSq);

            const Bitboard whiteKnightAllowedSquares =
                ~whiteKingSq.GetBitboard() &
                ~blackKingSq.GetBitboard();

            const Bitboard whiteBishopAllowedSquares =
                ~blackKingSq.GetBitboard() &
                ~whiteKingSq.GetBitboard() &
                ~blackRookSq.GetBitboard();

            if ((whiteKingSq.GetBitboard() & whiteKingAllowedSquares) == 0) continue;
            if ((blackRookSq.GetBitboard() & whiteKnightAllowedSquares) == 0) continue;
            if ((whiteQueenSq.GetBitboard() & whiteBishopAllowedSquares) == 0) continue;

            Position pos;
            pos.SetSideToMove(Color::Black);
            pos.SetPiece(blackKingSq, Piece::King, Color::Black);
            pos.SetPiece(whiteKingSq, Piece::King, Color::White);
            pos.SetPiece(blackRookSq, Piece::Rook, Color::Black);
            pos.SetPiece(whiteQueenSq, Piece::Queen, Color::White);
            ASSERT(pos.IsValid());

            // generate only quiet position
            if (!pos.IsQuiet()) continue;

            Move bestMove;
            uint32_t dtz = UINT32_MAX;
            int32_t wdl = 0;
            bool probeResult = ProbeTablebase_Root(pos, bestMove, &dtz, &wdl);

            if (probeResult)
            {
                ASSERT(dtz < UINT8_MAX);

                successfullyProbed++;
                maxDtz = std::max(maxDtz, dtz);

                if (wdl == 0) dtz = 100;

                whiteKingCounters[whiteKingSq.Index()] += dtz;
                whiteQueenCounters[whiteQueenSq.Index()] += dtz;
                blackRookCounters[blackRookSq.Index()] += dtz;
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

        std::cout << std::endl << "Black King on: " << blackKingSq.ToString() << std::endl;
        std::cout << std::endl << "White King PST:" << std::endl;
        printPST(whiteKingCounters);
        std::cout << std::endl << "White Queen PST:" << std::endl;
        printPST(whiteQueenCounters);
        std::cout << std::endl << "Black Rook PST:" << std::endl;
        printPST(blackRookCounters);
    }

    std::cout << "Successfully probed: " << successfullyProbed << std::endl;
    std::cout << "Max DTZ:             " << maxDtz << std::endl;
}