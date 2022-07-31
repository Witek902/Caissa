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
#include "../backend/Waitable.hpp"

#include "ThreadPool.hpp"

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

    double totalErrorSqr = 0.0;

    uint64_t incorrectWins = 0;
    uint64_t incorrectDraws = 0;
    uint64_t incorrectLosses = 0;

    uint64_t recognizedWins = 0;
    uint64_t recognizedDraws = 0;
    uint64_t recognizedLosses = 0;

    uint64_t notRecognizedWins = 0;
    uint64_t notRecognizedDraws = 0;
    uint64_t notRecognizedLosses = 0;

    int64_t pieceSquareScores[64][12];
    uint64_t pieceSquareCounters[64][12];

    EndgameValidationStats()
    {
        memset(pieceSquareScores, 0, sizeof(pieceSquareScores));
        memset(pieceSquareCounters, 0, sizeof(pieceSquareCounters));
    }

    void Append(const EndgameValidationStats& other)
    {
        count += other.count;
        totalErrorSqr += other.totalErrorSqr;

        incorrectWins += other.incorrectWins;
        incorrectDraws += other.incorrectDraws;
        incorrectLosses += other.incorrectLosses;

        recognizedWins += other.recognizedWins;
        recognizedDraws += other.recognizedDraws;
        recognizedLosses += other.recognizedLosses;

        notRecognizedWins += other.notRecognizedWins;
        notRecognizedDraws += other.notRecognizedDraws;
        notRecognizedLosses += other.notRecognizedLosses;

        for (uint32_t square = 0; square < 64; square++)
        {
            for (uint32_t piece = 0; piece < 12; piece++)
            {
                pieceSquareScores[square][piece] += other.pieceSquareScores[square][piece];
                pieceSquareCounters[square][piece] += other.pieceSquareCounters[square][piece];
            }
        }
    }

    void PrintPieceSquareTable() const
    {
        for (uint32_t color = 0; color < 2; color++)
        {
            for (uint32_t pieceIdx = 0; pieceIdx < 6; pieceIdx++)
            {
                bool hasAnyScore = false;
                for (uint32_t square = 0; square < 64; square++)
                {
                    if (pieceSquareCounters[square][pieceIdx + color * 6] > 0)
                    {
                        hasAnyScore = true;
                        break;
                    }
                }

                if (!hasAnyScore)
                {
                    continue;
                }

                const Piece piece = (Piece)(pieceIdx + (uint32_t)Piece::Pawn);

                std::cout
                    << "static const int16_t "
                    << (color == 0 ? "White" : "Black")
                    << PieceToString(piece)
                    << "Psqt[] = {"
                    << std::endl;

                int32_t averageCP = 0;
                int32_t numValidSquares = 0;

                for (uint32_t rank = 0; rank < 8; ++rank)
                {
                    std::cout << "    ";
                    for (uint32_t file = 0; file < 8; file++)
                    {
                        const uint32_t square = 8 * rank + file;
                        const int64_t score = pieceSquareScores[square][pieceIdx + color * 6];
                        const int64_t counter = pieceSquareCounters[square][pieceIdx + color * 6];
                        
                        const float weight = counter > 0 ? WinProbabilityToPawns(0.5f + 0.5f * (float)score / (float)counter) : 0.0f;
                        const int32_t cp = (counter > 0 && score >= counter) ? 9999 : int32_t(roundf(100.0f * weight));

                        averageCP += counter > 0 ? cp : 0;
                        numValidSquares += counter > 0;

                        std::cout << std::right << std::setw(6) << cp << ", ";
                    }
                    std::cout << std::endl;
                }

                std::cout << "};" << std::endl;

                std::cout << "Average: " << (numValidSquares > 0 ? (averageCP / numValidSquares) : 0) << std::endl;
                std::cout << std::endl;
            }
        }
    }
};

struct EndgameValidationParam
{
    MaterialKey matKey;
    Color sideToMove = Color::White;
    Bitboard whitePawnsAllowedSquares   = Bitboard::Full();
    Bitboard whiteKnightsAllowedSquares = Bitboard::Full();
    Bitboard whiteBishopsAllowedSquares = Bitboard::Full();
    Bitboard whiteRooksAllowedSquares   = Bitboard::Full();
    Bitboard whiteQueensAllowedSquares  = Bitboard::Full();
    Bitboard blackPawnsAllowedSquares   = Bitboard::Full();
    Bitboard blackKnightsAllowedSquares = Bitboard::Full();
    Bitboard blackBishopsAllowedSquares = Bitboard::Full();
    Bitboard blackRooksAllowedSquares   = Bitboard::Full();
    Bitboard blackQueensAllowedSquares  = Bitboard::Full();
};

static void ValidateEndgameForKingsPlacement(const EndgameValidationParam& param, const Square whiteKingSq, const Square blackKingSq, EndgameValidationStats& stats)
{
    const uint32_t numPieces = param.matKey.CountAll();
    ASSERT(numPieces <= 10);

    uint64_t maxNumPositions = 1ull << (6 * numPieces);

    for (uint32_t posIndex = 0; posIndex < maxNumPositions; ++posIndex)
    {
        Position pos;
        pos.SetSideToMove(param.sideToMove);
        pos.SetPiece(whiteKingSq, Piece::King, Color::White);
        pos.SetPiece(blackKingSq, Piece::King, Color::Black);

        Bitboard occupied = whiteKingSq.GetBitboard() | blackKingSq.GetBitboard();

        bool positionValid = true;
        uint32_t pieceIndex = 0;

        const auto placePiece = [&](const Piece type, const Color color, const Bitboard allowedSquare) INLINE_LAMBDA
        {
            const Square pieceSquare = (posIndex >> (6 * pieceIndex)) & 0x3F;
            pieceIndex++;

            if (pieceSquare.GetBitboard() & (occupied | ~allowedSquare))
            {
                positionValid = false;
                return;
            }

            occupied |= pieceSquare.GetBitboard();
            pos.SetPiece(pieceSquare, type, color);
        };

        const auto updatePieceSquareCounters = [&](int32_t value) INLINE_LAMBDA
        {
            for (uint32_t i = 0; i < 6; ++i)
            {
                const Piece piece = (Piece)(i + (uint32_t)Piece::Pawn);
                pos.Whites().GetPieceBitBoard(piece).Iterate([&](uint32_t index)
                {
                    stats.pieceSquareCounters[index][i] += 1;
                    stats.pieceSquareScores[index][i] += value;
                });
                pos.Blacks().GetPieceBitBoard(piece).Iterate([&](uint32_t index)
                {
                    stats.pieceSquareCounters[index][i + 6] += 1;
                    stats.pieceSquareScores[index][i + 6] += value;
                });
            }
        };

        for (uint32_t i = 0; i < param.matKey.numWhitePawns; ++i)     placePiece(Piece::Pawn, Color::White, param.whitePawnsAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numWhiteKnights; ++i)   placePiece(Piece::Knight, Color::White, param.whiteKnightsAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numWhiteBishops; ++i)   placePiece(Piece::Bishop, Color::White, param.whiteBishopsAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numWhiteRooks; ++i)     placePiece(Piece::Rook, Color::White, param.whiteRooksAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numWhiteQueens; ++i)    placePiece(Piece::Queen, Color::White, param.whiteQueensAllowedSquares);

        for (uint32_t i = 0; i < param.matKey.numBlackPawns; ++i)     placePiece(Piece::Pawn, Color::Black, param.blackPawnsAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numBlackKnights; ++i)   placePiece(Piece::Knight, Color::Black, param.blackKnightsAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numBlackBishops; ++i)   placePiece(Piece::Bishop, Color::Black, param.blackBishopsAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numBlackRooks; ++i)     placePiece(Piece::Rook, Color::Black, param.blackRooksAllowedSquares);
        for (uint32_t i = 0; i < param.matKey.numBlackQueens; ++i)    placePiece(Piece::Queen, Color::Black, param.blackQueensAllowedSquares);

        //if (Square(FirstBitSet(pos.Whites().pawns)) != Square_b6)
        //{
        //    continue;
        //}

        if (!positionValid)
        {
            continue;
        }

        if (!pos.IsValid(true)) continue;
        if (pos.IsInCheck(GetOppositeColor(param.sideToMove))) continue;
        if (pos.IsInCheck(param.sideToMove)) continue;
        if (!pos.IsQuiet()) continue;

        int32_t wdl = 0;
        bool probeResult = ProbeTablebase_WDL(pos, &wdl);

        ASSERT(wdl >= -1 && wdl <= 1);

        // make WDL score be white perspective
        if (param.sideToMove == Color::Black)
        {
            wdl = -wdl;
        }

        const float trueScore = 0.5f + 0.5f * wdl;

        if (probeResult)
        {
            bool exactScoreRecognized = false;

            stats.count++;

            int32_t evalScore = 0;
            if (EvaluateEndgame(pos, evalScore))
            {
                const float error = trueScore - PawnToWinProbability(evalScore * 0.01f);
                stats.totalErrorSqr += error * error;

                if (wdl > 0) // win
                {
                    if (evalScore >= KnownWinValue)
                    {
                        stats.recognizedWins++;
                        exactScoreRecognized = true;
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
                        exactScoreRecognized = true;
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
                        exactScoreRecognized = true;
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

            // update PSQT only for non-recognized scores, so the PSQT evaluation includes only these positions
            if (!exactScoreRecognized)
            {
                updatePieceSquareCounters(wdl);
            }
        }
    }
}

static void ValidateEndgame_2v2(const EndgameValidationParam& param)
{
    using namespace threadpool;

    std::cout << "Side to move: " << (param.sideToMove == Color::White ? "WHITE" : "BLACK") << std::endl;

    EndgameValidationStats stats;
    std::mutex statsMutex;

    Waitable waitable;
    {
        TaskBuilder taskBuilder(waitable);

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

                taskBuilder.Task("ValidateEndgame", [&param, whiteKingSq, blackKingSq, &stats](const TaskContext&)
                {
                    EndgameValidationStats localStats;
                    ValidateEndgameForKingsPlacement(param, whiteKingSq, blackKingSq, localStats);
                    {
                        std::unique_lock<std::mutex>(statsMutex);
                        stats.Append(localStats);
                    }
                });
            }
        }
    }
    waitable.Wait();

    std::cout << "Successfully probed:   " << stats.count << std::endl << std::endl;
    std::cout << "Mean square error:     " << std::sqrt(stats.totalErrorSqr / stats.count) << std::endl << std::endl;

    std::cout << "Incorrect Wins:        " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.incorrectWins / (float)stats.count) << "% (" << stats.incorrectWins << ")" << std::endl;
    std::cout << "Incorrect Draws:       " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.incorrectDraws / (float)stats.count) << "% (" << stats.incorrectDraws << ")" << std::endl;
    std::cout << "Incorrect Losses:      " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.incorrectLosses / (float)stats.count) << "% (" << stats.incorrectLosses << ")" << std::endl;
    std::cout << "Correct Wins:          " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.recognizedWins / (float)stats.count) << "% (" << stats.recognizedWins << ")" << std::endl;
    std::cout << "Correct Draws:         " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.recognizedDraws / (float)stats.count) << "% (" << stats.recognizedDraws << ")" << std::endl;
    std::cout << "Correct Losses:        " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.recognizedLosses / (float)stats.count) << "% (" << stats.recognizedLosses << ")" << std::endl;
    std::cout << "Non-recognized Wins:   " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.notRecognizedWins / (float)stats.count) << "% (" << stats.notRecognizedWins << ")" << std::endl;
    std::cout << "Non-recognized Draws:  " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.notRecognizedDraws / (float)stats.count) << "% (" << stats.notRecognizedDraws << ")" << std::endl;
    std::cout << "Non-recognized Losses: " << std::right << std::fixed << std::setprecision(1) << (100.0f * (float)stats.notRecognizedLosses / (float)stats.count) << "% (" << stats.notRecognizedLosses << ")" << std::endl;

    std::cout << std::endl;

    stats.PrintPieceSquareTable();
}

void ValidateEndgame()
{
    EndgameValidationParam param;
    param.matKey.numWhiteBishops = 0;
    param.matKey.numWhiteRooks = 0;
    param.matKey.numWhitePawns = 2;
    param.matKey.numBlackRooks = 0;
    param.matKey.numBlackPawns = 0;
    param.matKey.numBlackKnights = 0;
    param.matKey.numBlackBishops = 0;

    //param.whitePawnsAllowedSquares = Square(0, 6).GetBitboard();// Bitboard::FileBitboard<0>() | Bitboard::FileBitboard<1>() | Bitboard::FileBitboard<2>() | Bitboard::FileBitboard<3>();

    ValidateEndgame_2v2(param);

    param.sideToMove = Color::Black;

    ValidateEndgame_2v2(param);
}