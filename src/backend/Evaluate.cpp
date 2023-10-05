#include "Evaluate.hpp"
#include "Move.hpp"
#include "Material.hpp"
#include "Endgame.hpp"
#include "PackedNeuralNetwork.hpp"
#include "NeuralNetworkEvaluator.hpp"
#include "Pawns.hpp"
#include "Search.hpp"

#include <fstream>
#include <memory>

#if defined(CAISSA_EVALFILE)

    // embed eval file into executable
    #define INCBIN_PREFIX
    #define INCBIN_STYLE INCBIN_STYLE_CAMEL
    #include "incbin.h"
    INCBIN(Embed, CAISSA_EVALFILE);

    const char* c_DefaultEvalFile = "<empty>";

#else // !defined(CAISSA_EVALFILE)

    // use eval file
    const char* c_DefaultEvalFile = "eval-21.pnn";

#endif // defined(CAISSA_EVALFILE)


#define S(mg, eg) PieceScore{ mg, eg }

namespace {

static constexpr int32_t c_evalSaturationTreshold   = 8000;

static constexpr PieceScore c_tempoBonus            = S(2, 2);

#ifdef USE_MOBILITY
static constexpr PieceScore c_knightMobilityBonus[9] = {
    S(-28,-112), S(-14, -39), S(-8,  -5), S(-2,  12), S(3,  22), S(5,  34), S(14,  32), S(21,  28), S(27,  17) };
static constexpr PieceScore c_bishopMobilityBonus[14] ={
    S( -29,-105), S( -22, -49), S( -10, -29), S(  -6,  -6), S(  -1,   2), S(   0,   8), S(   4,  16),
    S(  10,  22), S(  12,  20), S(  17,  22), S(  25,  14), S(  36,  18), S(  32,  20), S(  37,  17) };
static constexpr PieceScore c_rookMobilityBonus[15] = {
    S( -28, -75), S( -21, -39), S( -17, -21), S( -13, -14), S( -15,   4), S( -10,   7), S(  -4,  17),
    S(   1,  14), S(   3,  13), S(   8,  17), S(  15,  20), S(  27,  15), S(  36,  11), S(  41,   1), S(  75, -16) };
static constexpr PieceScore c_queenMobilityBonus[28] = {
    S( -34, -70), S( -23, -80), S( -16, -78), S( -13, -72), S( -10, -56), S(  -7, -44), S(  -7, -22),
    S(  -3, -21), S(  -5, -11), S(   0,   6), S(   1,  10), S(   2,  20), S(   0,  26), S(   0,  32),
    S(   2,  40), S(   5,  38), S(   4,  33), S(   8,  34), S(  19,  35), S(  28,  32), S(  34,  16),
    S(  45,  11), S(  40,  17), S(  35,  10), S(  31,   9), S(  28,   3), S(  25,   6), S(  20,   1) };
#endif // USE_MOBILITY


} // namespace

PackedNeuralNetworkPtr g_mainNeuralNetwork;

bool LoadMainNeuralNetwork(const char* path)
{
    PackedNeuralNetworkPtr network = std::make_unique<nn::PackedNeuralNetwork>();

    if (path == nullptr || strcmp(path, "") == 0 || strcmp(path, "<empty>") == 0)
    {
#if defined(CAISSA_EVALFILE)
        if (network->LoadFromMemory(EmbedData))
        {
            g_mainNeuralNetwork = std::move(network);
            std::cout << "info string Using embedded neural network" << std::endl;
            return true;
        }
#endif // defined(CAISSA_EVALFILE)

        std::cout << "info string disabled neural network evaluation" << std::endl;
        g_mainNeuralNetwork.reset();
        return true;
    }

    if (network->LoadFromFile(path))
    {
        g_mainNeuralNetwork = std::move(network);
        std::cout << "info string Loaded neural network: " << path << std::endl;
        return true;
    }

    // TODO use embedded net?

    g_mainNeuralNetwork.reset();
    return false;
}

static std::string GetDefaultEvalFilePath()
{
    std::string path = GetExecutablePath();

    if (!path.empty())
    {
        path = path.substr(0, path.find_last_of("/\\")); // remove exec name
        path += "/";
    }

    return path;
}

bool TryLoadingDefaultEvalFile()
{
#if defined(CAISSA_EVALFILE)

    // use embedded net
    return LoadMainNeuralNetwork(nullptr);

#else // !defined(CAISSA_EVALFILE)

    // check if there's eval file in same directory as executable
    {
        std::string path = GetDefaultEvalFilePath() + c_DefaultEvalFile;
        if (!path.empty())
        {
            bool fileExists = false;
            {
                std::ifstream f(path.c_str());
                fileExists = f.good();
            }

            if (fileExists && LoadMainNeuralNetwork(path.c_str()))
            {
                return true;
            }
        }
    }

    // try working directory
    {
        bool fileExists = false;
        {
            std::ifstream f(c_DefaultEvalFile);
            fileExists = f.good();
        }

        if (fileExists && LoadMainNeuralNetwork(c_DefaultEvalFile))
        {
            return true;
        }
    }

    std::cout << "info string Failed to load default neural network " << c_DefaultEvalFile << std::endl;
    return false;

#endif // defined(CAISSA_EVALFILE)
}

static int32_t InterpolateScore(const int32_t mgPhase, const TPieceScore<int32_t>& score)
{
    ASSERT(mgPhase >= 0 && mgPhase <= 64);
    const int32_t egPhase = 64 - mgPhase;
    return (score.mg * mgPhase + score.eg * egPhase) / 64;
}

bool CheckInsufficientMaterial(const Position& pos)
{
    const Bitboard queensRooksPawns =
        pos.Whites().queens | pos.Whites().rooks | pos.Whites().pawns |
        pos.Blacks().queens | pos.Blacks().rooks | pos.Blacks().pawns;

    if (queensRooksPawns != 0)
    {
        return false;
    }

    if (pos.Whites().knights == 0 && pos.Blacks().knights == 0)
    {
        // king and bishop vs. king
        if ((pos.Whites().bishops == 0 && pos.Blacks().bishops.Count() <= 1) ||
            (pos.Whites().bishops.Count() <= 1 && pos.Blacks().bishops == 0))
        {
            return true;
        }

        // king and bishop vs. king and bishop (bishops on the same color squares)
        if (pos.Whites().bishops.Count() == 1 && pos.Blacks().bishops.Count() == 1)
        {
            const bool whiteBishopOnLightSquare = (pos.Whites().bishops & Bitboard::LightSquares()) != 0;
            const bool blackBishopOnLightSquare = (pos.Blacks().bishops & Bitboard::LightSquares()) != 0;
            return whiteBishopOnLightSquare == blackBishopOnLightSquare;
        }
    }


    // king and knight vs. king
    if (pos.Whites().bishops == 0 && pos.Blacks().bishops == 0)
    {
        if ((pos.Whites().knights == 0 && pos.Blacks().knights.Count() <= 1) ||
            (pos.Whites().knights.Count() <= 1 && pos.Blacks().knights == 0))
        {
            return true;
        }
    }

    return false;
}

#ifdef USE_MOBILITY
static TPieceScore<int32_t> EvaluateMobility(const Position& pos)
{
    const Bitboard whitesOccupied = pos.Whites().Occupied();
    const Bitboard blacksOccupied = pos.Blacks().Occupied();
    const Bitboard allOccupied = whitesOccupied | blacksOccupied;

    const Bitboard whitePawnsAttacks = Bitboard::GetPawnAttacks<Color::White>(pos.Whites().pawns);
    const Bitboard blackPawnsAttacks = Bitboard::GetPawnAttacks<Color::Black>(pos.Blacks().pawns);

    const Bitboard whiteKnightsAttacks = Bitboard::GetKnightAttacks(pos.Whites().knights);
    const Bitboard blackKnightsAttacks = Bitboard::GetKnightAttacks(pos.Blacks().knights);

    const Bitboard whitesMinorsArea = ~whitesOccupied & ~blackPawnsAttacks;
    const Bitboard blacksMinorsArea = ~blacksOccupied & ~whitePawnsAttacks;

    const Bitboard whitesMajorsArea = whitesMinorsArea & ~blackKnightsAttacks;
    const Bitboard blacksMajorsArea = whitesMinorsArea & ~whiteKnightsAttacks;

    TPieceScore<int32_t> value = { 0, 0 };

    // white pieces
    {
        pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_knightMobilityBonus[(Bitboard::GetKnightAttacks(Square(square)) & whitesMinorsArea).Count()];
        });
        pos.Whites().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_bishopMobilityBonus[(Bitboard::GenerateBishopAttacks(Square(square), allOccupied) & whitesMinorsArea).Count()];
        });
        pos.Whites().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_rookMobilityBonus[(Bitboard::GenerateRookAttacks(Square(square), allOccupied) & whitesMajorsArea).Count()];
        });
        pos.Whites().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_queenMobilityBonus[(Bitboard::GenerateQueenAttacks(Square(square), allOccupied) & whitesMajorsArea).Count()];
        });
    }

    // black pieces
    {
        pos.Blacks().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_knightMobilityBonus[(Bitboard::GetKnightAttacks(Square(square)) & blacksMinorsArea).Count()];
        });
        pos.Blacks().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_bishopMobilityBonus[(Bitboard::GenerateBishopAttacks(Square(square), allOccupied) & blacksMinorsArea).Count()];
        });
        pos.Blacks().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_rookMobilityBonus[(Bitboard::GenerateRookAttacks(Square(square), allOccupied) & blacksMajorsArea).Count()];
        });
        pos.Blacks().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_queenMobilityBonus[(Bitboard::GenerateQueenAttacks(Square(square), allOccupied) & blacksMajorsArea).Count()];
        });
    }

    return value;
}
#endif // USE_MOBILITY

ScoreType Evaluate(const Position& pos)
{
    NodeInfo dummyNode = { pos };

    AccumulatorCache dummyCache;
    if (g_mainNeuralNetwork)
    {
        dummyCache.Init(g_mainNeuralNetwork.get());
    }

    return Evaluate(dummyNode, dummyCache);
}

ScoreType Evaluate(NodeInfo& node, AccumulatorCache& cache)
{
    const Position& pos = node.position;

    const int32_t whiteQueens   = pos.Whites().queens.Count();
    const int32_t whiteRooks    = pos.Whites().rooks.Count();
    const int32_t whiteBishops  = pos.Whites().bishops.Count();
    const int32_t whiteKnights  = pos.Whites().knights.Count();
    const int32_t whitePawns    = pos.Whites().pawns.Count();
    const int32_t blackQueens   = pos.Blacks().queens.Count();
    const int32_t blackRooks    = pos.Blacks().rooks.Count();
    const int32_t blackBishops  = pos.Blacks().bishops.Count();
    const int32_t blackKnights  = pos.Blacks().knights.Count();
    const int32_t blackPawns    = pos.Blacks().pawns.Count();

    const int32_t whitePieceCount = whiteQueens + whiteRooks + whiteBishops + whiteKnights + whitePawns;
    const int32_t blackPieceCount = blackQueens + blackRooks + blackBishops + blackKnights + blackPawns;

    int32_t scale = c_endgameScaleMax;

    // check endgame evaluation first
    if (whitePieceCount + blackPieceCount <= 6 || blackPieceCount == 0 || whitePieceCount == 0)
    {
        int32_t endgameScore;
        if (EvaluateEndgame(pos, endgameScore, scale))
        {
            ASSERT(endgameScore < TablebaseWinValue && endgameScore > -TablebaseWinValue);
            return (ScoreType)endgameScore;
        }
    }

    // 0 - endgame, 64 - opening
    const int32_t gamePhase = std::min(64,
        3 * (whiteKnights + blackKnights + whiteBishops + blackBishops) +
        5 * (whiteRooks   + blackRooks) +
        10 * (whiteQueens  + blackQueens));

    int32_t finalValue = 0;

    if (g_mainNeuralNetwork)
    {
        int32_t nnValue = NNEvaluator::Evaluate(*g_mainNeuralNetwork, node, cache);

        // convert to centipawn range
        nnValue = (nnValue * c_nnOutputToCentiPawns + nn::OutputScale / 2) / nn::OutputScale;

        // NN output is side-to-move relative
        if (pos.GetSideToMove() == Color::Black) nnValue = -nnValue;

        finalValue = nnValue;
    }
    else // fallback to simple evaluation
    {
        TPieceScore<int32_t> value = {};

        value += c_queenValue * (whiteQueens - blackQueens);
        value += c_rookValue * (whiteRooks - blackRooks);
        value += c_bishopValue * (whiteBishops - blackBishops);
        value += c_knightValue * (whiteKnights - blackKnights);
        value += c_pawnValue * (whitePawns - blackPawns);

        // tempo bonus
        value += (pos.GetSideToMove() == Color::White) ? c_tempoBonus : -c_tempoBonus;

#ifdef USE_MOBILITY
        value += EvaluateMobility(pos);
#endif // USE_MOBILITY

        // accumulate middle/end game scores
        finalValue = InterpolateScore(gamePhase, value);
    }

    // apply scaling based on game phase
    finalValue = finalValue * (96 + gamePhase) / 128;

    // saturate eval value so it doesn't exceed KnownWinValue
    if (finalValue > c_evalSaturationTreshold)
        finalValue = c_evalSaturationTreshold + (finalValue - c_evalSaturationTreshold) / 8;
    else if (finalValue < -c_evalSaturationTreshold)
        finalValue = -c_evalSaturationTreshold + (finalValue + c_evalSaturationTreshold) / 8;

    ASSERT(finalValue > -KnownWinValue && finalValue < KnownWinValue);

    return (ScoreType)(finalValue * scale / c_endgameScaleMax);
}

void EnsureAccumulatorUpdated(NodeInfo& node, AccumulatorCache& cache)
{
    if (g_mainNeuralNetwork)
    {
        NNEvaluator::EnsureAccumulatorUpdated(*g_mainNeuralNetwork, node, cache);
    }
}