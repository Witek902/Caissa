#include "Common.hpp"

#include "../backend/Position.hpp"
#include "../backend/Material.hpp"
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

struct EndgameValidationStats
{
    uint64_t count = 0;

    uint64_t incorrectWins = 0;
    uint64_t incorrectDraws = 0;
    uint64_t incorrectLosses = 0;

    uint64_t recognizedWins = 0;
    uint64_t recognizedDraws = 0;
    uint64_t recognizedLosses = 0;

    uint64_t notRecognizedWins = 0;
    uint64_t notRecognizedDraws = 0;
    uint64_t notRecognizedLosses = 0;
};

static void ValidateEndgameForKingsPlacement(const MaterialKey matKey, const Color sideToMove, const Square whiteKingSq, const Square blackKingSq, EndgameValidationStats& stats)
{
    const uint32_t numPieces = matKey.CountAll();
    ASSERT(numPieces <= 10);

    uint64_t maxNumPositions = 1ull << (6 * numPieces);

    for (uint32_t posIndex = 0; posIndex < maxNumPositions; ++posIndex)
    {
        Position pos;
        pos.SetSideToMove(sideToMove);
        pos.SetPiece(whiteKingSq, Piece::King, Color::White);
        pos.SetPiece(blackKingSq, Piece::King, Color::Black);

        Bitboard occupied = whiteKingSq.GetBitboard() | blackKingSq.GetBitboard();

        bool positionValid = true;
        uint32_t pieceIndex = 0;

        const auto placePiece = [&](const Piece type, const Color color) INLINE_LAMBDA
        {
            const Square pieceSquare = (posIndex >> (6 * pieceIndex)) & 0x3F;
            pieceIndex++;

            if (pieceSquare.GetBitboard() & occupied)
            {
                positionValid = false;
                return;
            }

            occupied |= pieceSquare.GetBitboard();
            pos.SetPiece(pieceSquare, type, color);
        };

        for (uint32_t i = 0; i < matKey.numWhitePawns; ++i)     placePiece(Piece::Pawn, Color::White);
        for (uint32_t i = 0; i < matKey.numWhiteKnights; ++i)   placePiece(Piece::Knight, Color::White);
        for (uint32_t i = 0; i < matKey.numWhiteBishops; ++i)   placePiece(Piece::Bishop, Color::White);
        for (uint32_t i = 0; i < matKey.numWhiteRooks; ++i)     placePiece(Piece::Rook, Color::White);
        for (uint32_t i = 0; i < matKey.numWhiteQueens; ++i)    placePiece(Piece::Queen, Color::White);

        for (uint32_t i = 0; i < matKey.numBlackPawns; ++i)     placePiece(Piece::Pawn, Color::Black);
        for (uint32_t i = 0; i < matKey.numBlackKnights; ++i)   placePiece(Piece::Knight, Color::Black);
        for (uint32_t i = 0; i < matKey.numBlackBishops; ++i)   placePiece(Piece::Bishop, Color::Black);
        for (uint32_t i = 0; i < matKey.numBlackRooks; ++i)     placePiece(Piece::Rook, Color::Black);
        for (uint32_t i = 0; i < matKey.numBlackQueens; ++i)    placePiece(Piece::Queen, Color::Black);

        if (!positionValid)
        {
            continue;
        }

        if (!pos.IsValid(true)) continue;
        if (pos.IsInCheck(GetOppositeColor(sideToMove))) continue;

        // check only quiet position
        //MoveList moves;
        //pos.GenerateMoveList(moves, MOVE_GEN_ONLY_TACTICAL);
        //if (moves.Size() > 0)
        //{
        //    continue;
        //}

        int32_t wdl = 0;
        bool probeResult = ProbeTablebase_WDL(pos, &wdl);

        // make WDL score be white perspective
        if (sideToMove == Color::Black)
        {
            wdl = -wdl;
        }

        if (probeResult)
        {
            stats.count++;

            int32_t evalScore = 0;
            if (EvaluateEndgame(pos, evalScore))
            {
                if (wdl > 0) // win
                {
                    if (evalScore >= KnownWinValue)
                    {
                        stats.recognizedWins++;
                    }
                    else if (evalScore <= -KnownWinValue)
                    {
                        std::cout << "Incorrect win score: " << pos.ToFEN() << std::endl;
                        stats.incorrectWins++;
                    }
                    else
                    {
                        stats.notRecognizedWins++;
                    }
                }
                else if (wdl < 0) // loss
                {
                    if (evalScore <= -KnownWinValue)
                    {
                        stats.recognizedLosses++;
                    }
                    else if (evalScore >= KnownWinValue)
                    {
                        std::cout << "Incorrect loss score: " << pos.ToFEN() << std::endl;
                        stats.incorrectLosses++;
                    }
                    else
                    {
                        stats.notRecognizedLosses++;
                    }
                }
                else // draw
                {
                    if (evalScore == 0)
                    {
                        stats.recognizedDraws++;
                    }
                    else if (evalScore >= KnownWinValue || evalScore <= -KnownWinValue)
                    {
                        std::cout << "Incorrect draw score: " << pos.ToFEN() << std::endl;
                        stats.incorrectDraws++;
                    }
                    else
                    {
                        stats.notRecognizedDraws++;
                    }
                }
            }
            else
            {
                if (wdl > 0)
                {
                    stats.notRecognizedWins++;
                }
                else if (wdl < 0)
                {
                    stats.notRecognizedLosses++;
                }
                else
                {
                    stats.notRecognizedDraws++;
                }
            }
        }
    }
}

void ValidateEndgame_2v2(const MaterialKey matKey, const Color sideToMove)
{
    std::cout << "Side to move: " << (sideToMove == Color::White ? "WHITE" : "BLACK") << std::endl;

    EndgameValidationStats stats;

    for (uint32_t whiteKingSqIdx = 0; whiteKingSqIdx < 64; ++whiteKingSqIdx)
    {
        const Square whiteKingSq(whiteKingSqIdx);

        for (uint32_t blackKingSqIdx = 0; blackKingSqIdx < 64; ++blackKingSqIdx)
        {
            const Square blackKingSq(blackKingSqIdx);

            if (Square::Distance(whiteKingSq, blackKingSq) <= 1)
            {
                // kings cannot be touching
                continue;
            }

            ValidateEndgameForKingsPlacement(matKey, sideToMove, whiteKingSq, blackKingSq, stats);
        }
    }

    std::cout << "Successfully probed:   " << stats.count << std::endl << std::endl;

    std::cout << "Incorrect Wins:        " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.incorrectWins / (float)stats.count) << "% (" << stats.incorrectWins << ")" << std::endl;
    std::cout << "Incorrect Draws:       " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.incorrectDraws / (float)stats.count) << "% (" << stats.incorrectDraws << ")" << std::endl;
    std::cout << "Incorrect Losses:      " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.incorrectLosses / (float)stats.count) << "% (" << stats.incorrectLosses << ")" << std::endl;
    std::cout << "Correct Wins:          " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.recognizedWins / (float)stats.count) << "% (" << stats.recognizedWins << ")" << std::endl;
    std::cout << "Correct Draws:         " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.recognizedDraws / (float)stats.count) << "% (" << stats.recognizedDraws << ")" << std::endl;
    std::cout << "Correct Losses:        " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.recognizedLosses / (float)stats.count) << "% (" << stats.recognizedLosses << ")" << std::endl;
    std::cout << "Non-recognized Wins:   " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.notRecognizedWins / (float)stats.count) << "% (" << stats.notRecognizedWins << ")" << std::endl;
    std::cout << "Non-recognized Draws:  " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.notRecognizedDraws / (float)stats.count) << "% (" << stats.notRecognizedDraws << ")" << std::endl;
    std::cout << "Non-recognized Losses: " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.notRecognizedLosses / (float)stats.count) << "% (" << stats.notRecognizedLosses << ")" << std::endl;
}

void ValidateEndgame()
{
    MaterialKey key;
    key.numWhiteRooks = 1;
    key.numBlackPawns = 1;

    ValidateEndgame_2v2(key, Color::White);
    //ValidateEndgame_2v2(key, Color::Black);
}