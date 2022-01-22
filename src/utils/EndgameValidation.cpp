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

void ValidateEndgame_2v2()
{
    const uint32_t numWhiteKingPositions = 64;

    size_t successfullyProbed = 0;

    size_t incorrectWins = 0;
    size_t incorrectDraws = 0;
    size_t incorrectLosses = 0;

    size_t recognizedWins = 0;
    size_t recognizedDraws = 0;
    size_t recognizedLosses = 0;

    size_t notRecognizedWins = 0;
    size_t notRecognizedDraws = 0;
    size_t notRecognizedLosses = 0;

    const Color sideToMove = Color::White;

    for (uint32_t whiteKingSqIdx = 0; whiteKingSqIdx < numWhiteKingPositions; ++whiteKingSqIdx)
    {
        const Square whiteKingSq(whiteKingSqIdx);

        uint32_t maxNumPositions = 64 * 64 * 64;

        for (uint32_t posIndex = 0; posIndex < maxNumPositions; ++posIndex)
        {
            const Square blackKingSq = (posIndex) & 0x3F;
            const Square whitePieceSq = (posIndex >> 6) & 0x3F;
            const Square blackPieceSq = (posIndex >> 12) & 0x3F;

            const Bitboard blackKingAllowedSquares =
                ~whiteKingSq.GetBitboard() &
                ~Bitboard::GetKingAttacks(whiteKingSq);

            const Bitboard whitePieceAllowedSquares =
                ~whiteKingSq.GetBitboard() &
                ~blackKingSq.GetBitboard();

            const Bitboard blackPieceAllowedSquares =
                ~whiteKingSq.GetBitboard() &
                ~blackKingSq.GetBitboard() &
                ~whitePieceSq.GetBitboard();

            if ((blackKingSq.GetBitboard() & blackKingAllowedSquares) == 0) continue;
            if ((whitePieceSq.GetBitboard() & whitePieceAllowedSquares) == 0) continue;
            if ((blackPieceSq.GetBitboard() & blackPieceAllowedSquares) == 0) continue;

            Position pos;
            pos.SetSideToMove(sideToMove);
            pos.SetPiece(whiteKingSq, Piece::King, Color::White);
            pos.SetPiece(blackKingSq, Piece::King, Color::Black);
            pos.SetPiece(whitePieceSq, Piece::Rook, Color::White);
            pos.SetPiece(blackPieceSq, Piece::Rook, Color::Black);
            ASSERT(pos.IsValid());

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

            if (probeResult)
            {
                successfullyProbed++;

                int32_t evalScore = 0;
                if (EvaluateEndgame(pos, evalScore))
                {
                    if (wdl > 0) // win
                    {
                        if (evalScore >= KnownWinValue)
                        {
                            recognizedWins++;
                        }
                        else
                        {
                            std::cout << "Incorrect win score: " << pos.ToFEN() << std::endl;
                            incorrectWins++;
                        }
                    }
                    else if (wdl < 0) // loss
                    {
                        if (evalScore <= -KnownWinValue)
                        {
                            recognizedLosses++;
                        }
                        else
                        {
                            std::cout << "Incorrect loss score: " << pos.ToFEN() << std::endl;
                            incorrectLosses++;
                        }
                    }
                    else // draw
                    {
                        if (evalScore == 0)
                        {
                            recognizedDraws++;
                        }
                        else
                        {
                            std::cout << "Incorrect draw score: " << pos.ToFEN() << std::endl;
                            incorrectDraws++;
                        }
                    }
                }
                else
                {
                    if (wdl > 0)
                    {
                        notRecognizedWins++;
                        //std::cout << "Non recognized win score: " << pos.ToFEN() << std::endl;
                    }
                    else if (wdl < 0)
                    {
                        notRecognizedLosses++;
                    }
                    else
                    {
                        notRecognizedDraws++;
                    }
                }
            }
        }
    }

    std::cout << "Successfully probed:   " << successfullyProbed << std::endl;

    std::cout << "Incorrect Wins:        " << incorrectWins << std::endl;
    std::cout << "Incorrect Draws:       " << incorrectDraws << std::endl;
    std::cout << "Incorrect Losses:      " << incorrectLosses << std::endl;

    std::cout << "Correct Wins:          " << recognizedWins << std::endl;
    std::cout << "Correct Draws:         " << recognizedDraws << std::endl;
    std::cout << "Correct Losses:        " << recognizedLosses << std::endl;

    std::cout << "Non-recognized Wins:   " << notRecognizedWins << std::endl;
    std::cout << "Non-recognized Draws:  " << notRecognizedDraws << std::endl;
    std::cout << "Non-recognized Losses: " << notRecognizedLosses << std::endl;
}

void ValidateEndgame()
{
    ValidateEndgame_2v2();
}