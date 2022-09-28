#include "Evaluate.hpp"
#include "Move.hpp"
#include "Material.hpp"
#include "Endgame.hpp"
#include "PackedNeuralNetwork.hpp"
#include "PieceSquareTables.h"
#include "NeuralNetworkEvaluator.hpp"
#include "Pawns.hpp"
#include "Search.hpp"

#include <unordered_map>
#include <fstream>
#include <memory>

const char* c_DefaultEvalFile = "eval.pnn";
const char* c_DefaultEndgameEvalFile = "endgame.pnn";

#define S(mg, eg) PieceScore{ mg, eg }

static constexpr int32_t c_evalSaturationTreshold   = 4000;

alignas(CACHELINE_SIZE)
const PieceScore PSQT[6][Square::NumSquares] =
{
    {
        S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
        S( -35, -23), S(  -8,  -9), S( -19,  -9), S( -36, -13), S( -37, -15), S(  -8, -16), S(   2, -15), S( -33, -29),
        S( -29, -36), S(  -1, -26), S( -17, -34), S( -31, -33), S( -30, -33), S( -18, -33), S(   4, -31), S( -23, -38),
        S( -22, -26), S(  10, -25), S(  -1, -45), S(   3, -52), S(   2, -53), S(  -5, -41), S(  12, -31), S( -19, -33),
        S(  -4, -13), S(  17, -19), S(  16, -36), S(  44, -50), S(  38, -49), S(  29, -35), S(  25, -25), S(  -3, -24),
        S(   9,  21), S(  52,  26), S(  77,   1), S(  97, -12), S(  98, -10), S(  83,  -9), S(  61,  19), S(   5,  18),
        S( 157,  42), S( 138,  62), S( 171,  40), S( 189,  11), S( 193,  15), S( 171,  36), S( 130,  63), S( 154,  46),
        S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
    },
    {
        S( -44, -77), S( -37, -54), S( -38, -19), S( -35, -16), S( -28, -21), S( -40, -20), S( -34, -57), S( -51, -80),
        S( -32, -36), S( -38,   2), S( -26,   2), S( -20,   9), S( -19,  10), S( -23,   0), S( -32,  -2), S( -29, -36),
        S( -36,  -4), S( -20,  13), S(  -2,  23), S(   5,  40), S(   6,  37), S(  -8,  26), S( -15,  14), S( -33,  -6),
        S(  -3,  18), S(   8,  30), S(  19,  48), S(  13,  46), S(  11,  42), S(  10,  53), S(  11,  25), S( -11,  10),
        S(  20,  10), S(  17,  47), S(  42,  55), S(  46,  52), S(  37,  55), S(  49,  44), S(  13,  40), S(  17,  12),
        S(  18,  11), S(  57,  15), S(  85,  24), S(  93,  22), S( 100,  25), S(  93,  24), S(  65,  16), S(  22,  10),
        S(   3,  -9), S(  14,  15), S(  71,  17), S(  82,  18), S(  82,  23), S(  69,  11), S(  18,  17), S(  14,  -1),
        S( -81, -27), S(  -2,   6), S(   6,  13), S(  30,  12), S(  27,  15), S(   0,   9), S(  -6,   6), S( -66, -30),
    },
    {
        S( -23, -50), S( -16, -25), S( -35, -18), S( -38,  -5), S( -32,  -7), S( -38, -16), S( -24, -25), S( -19, -50),
        S(   3, -30), S(  -9, -11), S(  -1,  -5), S( -21,   3), S( -21,   1), S(  -4,  -7), S( -11, -17), S(   8, -33),
        S( -13, -12), S(   4,   6), S(  -8,  15), S(   3,  18), S(   3,  12), S(  -5,  15), S(   3,   6), S( -15,  -8),
        S(  -2,   1), S(  -3,  14), S(   3,  26), S(  17,  16), S(  22,   9), S(   4,  23), S(   4,  13), S(  -6,   3),
        S( -11,   0), S(   2,  28), S(  43,   7), S(  56,  12), S(  50,  17), S(  39,   9), S(  -1,  29), S(  -7,   0),
        S(  22,  -2), S(  50,   8), S(  52,  15), S(  67,   5), S(  62,   8), S(  56,  11), S(  52,   9), S(  22,  -3),
        S(  -4,  -4), S(  12,  13), S(  18,  12), S(  23,  20), S(  23,  21), S(  30,  14), S(   6,  15), S(  -2,  -4),
        S( -22,   0), S( -16,  12), S( -20,  14), S( -21,  26), S( -18,  22), S( -19,  14), S( -12,   8), S( -17,  -1),
    },
    {
        S( -40, -16), S( -31,  -7), S( -22,  -5), S(  -9,  -6), S(  -7, -15), S( -31,   1), S( -29,  -8), S( -38, -22),
        S( -41, -15), S( -33,  -6), S( -28,   1), S( -17,  -2), S( -17,   0), S( -25,  -4), S( -27,  -6), S( -44, -11),
        S( -41,   3), S( -24,  10), S( -36,  17), S( -21,  15), S( -18,  12), S( -30,  16), S( -21,   4), S( -41,   0),
        S( -36,  17), S( -13,  22), S( -23,  32), S(  -5,  25), S(  -7,  22), S( -18,  32), S( -13,  20), S( -33,  18),
        S(   3,  19), S(  22,  23), S(  21,  29), S(  42,  22), S(  37,  25), S(  27,  26), S(  19,  25), S(   7,  18),
        S(  33,  19), S(  61,  11), S(  70,  12), S(  87,   5), S(  93,   6), S(  68,  15), S(  65,  12), S(  29,  23),
        S(  60,  20), S(  50,  20), S(  90,  17), S( 107,   6), S( 100,   7), S(  98,  16), S(  56,  20), S(  51,  16),
        S(  78,   6), S(  74,   6), S(  72,  15), S(  87,   9), S(  83,   6), S(  78,   9), S(  79,   7), S(  80,   1),
    },
    {
        S( -51, -14), S( -33, -50), S( -29, -59), S( -27, -57), S( -22, -56), S( -27, -54), S( -32, -55), S( -50, -22),
        S( -14, -45), S(  -9, -55), S( -13, -60), S( -13, -45), S( -15, -39), S(  -3, -56), S( -11, -56), S( -15, -40),
        S(  -6, -36), S( -13, -16), S( -15,  -4), S( -23,   2), S( -18,   1), S( -13,  -8), S(  -1, -34), S( -10, -40),
        S( -19,  11), S(  -9,  11), S( -11,  26), S( -20,  39), S( -15,  32), S( -15,  24), S( -12,   3), S( -15,  -2),
        S(   2,   3), S( -11,  41), S(   0,  52), S(   8,  47), S(   9,  50), S(   8,  44), S(  -8,  34), S(  -4,   4),
        S(  18,   7), S(  24,  26), S(  19,  43), S(  31,  47), S(  38,  41), S(  25,  32), S(  31,  26), S(  16,   2),
        S(   8,   3), S( -17,  39), S(  22,  38), S(  24,  42), S(  23,  46), S(  22,  35), S(  -5,  23), S(  18, -10),
        S(  24,  -7), S(  31,   3), S(  38,  18), S(  26,  31), S(  28,  33), S(  25,  31), S(  17,  12), S(  29,  -4),
    },
    {
        S(  20, -82), S(  12, -57), S( -36, -46), S( -91, -32), S( -42, -61), S( -40, -41), S(   0, -50), S(  20, -82),
        S(   3, -40), S( -11, -19), S( -47,  -9), S( -78,  -8), S( -79,  -4), S( -52, -12), S( -20, -21), S(  -2, -36),
        S( -49, -10), S( -47,   3), S( -77,  18), S( -73,  23), S( -73,  22), S( -77,  18), S( -46,  -1), S( -49, -11),
        S( -81,  11), S( -52,  20), S( -38,  22), S( -35,  31), S( -33,  30), S( -31,  25), S( -56,  20), S( -82,   8),
        S( -63,  17), S(  17,  21), S(  26,  29), S(  56,  22), S(  57,  21), S(  29,  23), S(  22,  16), S( -61,  15),
        S(  28,   4), S(  90,  13), S( 100,  12), S(  94,  14), S(  96,  13), S(  97,  18), S(  89,  12), S(  26,   2),
        S(  30, -10), S(  67,  17), S(  73,  11), S(  60,   8), S(  62,   7), S(  74,  10), S(  66,  19), S(  32, -12),
        S(  -9, -37), S(  19,  11), S(  40,  -1), S(  57, -16), S(  51, -21), S(  54,  -9), S(  40,   7), S(  10, -46),
    }
};

static constexpr PieceScore c_tempoBonus = S(4, 4);

static constexpr PieceScore c_bishopPairBonus = S(28, 52);

const PieceScore c_ourPawnDistanceBonus[8]       = { S(  25,  11), S(  14,  18), S(   3,  11), S( -13,  14), S( -24,  16), S( -26,  16), S( -10,  16) };
const PieceScore c_ourKnightDistanceBonus[8]     = { S(  10,  19), S(  10,  20), S(   9,  13), S(   3,  11), S(   3,  -2), S(   0, -20), S(   6, -33) };
const PieceScore c_ourBishopDistanceBonus[8]     = { S(  13,   9), S(   8,   6), S(   9,   3), S(   7,  -1), S(   7,  -7), S(   7,  -9), S(  13, -17) };
const PieceScore c_ourRookDistanceBonus[8]       = { S( -30,  -3), S( -24,   0), S( -21, -12), S( -11, -17), S( -14, -21), S( -18, -17), S( -15, -25) };
const PieceScore c_ourQueenDistanceBonus[8]      = { S(   4,   1), S(  -1,   6), S(  -3,   2), S(  -4,  -3), S(  -3,  -5), S(   5,  -8), S(  17, -24) };
const PieceScore c_theirPawnDistanceBonus[8]     = { S(  37,  59), S( -12,  20), S(   0,  12), S(   6,  10), S(   8,   1), S(  17,  -7), S(  17, -14) };
const PieceScore c_theirKnightDistanceBonus[8]   = { S( -52,   6), S( -28,  -1), S( -10, -17), S(   3, -22), S(   5, -13), S(   8,  -6), S(  16,  11) };
const PieceScore c_theirBishopDistanceBonus[8]   = { S( -27,  21), S( -37,   6), S(   0,   0), S(   6,  -7), S(   5,  -8), S(   3,  -2), S(   1,   1) };
const PieceScore c_theirRookDistanceBonus[8]     = { S( -44,  27), S( -24,  20), S(  -1,  13), S(  13,  -2), S(  20,  -4), S(  27, -14), S(  30, -17) };
const PieceScore c_theirQueenDistanceBonus[8]    = { S(-120,   0), S( -60,   2), S( -10,  -9), S(  13, -13), S(  27,  -5), S(  29,  16), S(  33,  26) };

static constexpr PieceScore c_knightMobilityBonus[9] =
    { S(-28,-112), S(-14, -39), S(-8,  -5), S(-2,  12), S(3,  22), S(5,  34), S(14,  32), S(21,  28), S(27,  17) };

static constexpr PieceScore c_bishopMobilityBonus[14] =
    { S( -29,-105), S( -22, -49), S( -10, -29), S(  -6,  -6), S(  -1,   2), S(   0,   8), S(   4,  16),
      S(  10,  22), S(  12,  20), S(  17,  22), S(  25,  14), S(  36,  18), S(  32,  20), S(  37,  17) };

static constexpr PieceScore c_rookMobilityBonus[15] =
    { S( -28, -75), S( -21, -39), S( -17, -21), S( -13, -14), S( -15,   4), S( -10,   7), S(  -4,  17),
      S(   1,  14), S(   3,  13), S(   8,  17), S(  15,  20), S(  27,  15), S(  36,  11), S(  41,   1), S(  75, -16) };

static constexpr PieceScore c_queenMobilityBonus[28] =
    { S( -34, -70), S( -23, -80), S( -16, -78), S( -13, -72), S( -10, -56), S(  -7, -44), S(  -7, -22),
      S(  -3, -21), S(  -5, -11), S(   0,   6), S(   1,  10), S(   2,  20), S(   0,  26), S(   0,  32),
      S(   2,  40), S(   5,  38), S(   4,  33), S(   8,  34), S(  19,  35), S(  28,  32), S(  34,  16),
      S(  45,  11), S(  40,  17), S(  34,  10), S(  17,   9), S(  15,   3), S(   6,   6), S(   3,   1) };

static constexpr PieceScore c_passedPawnBonus[8]            = { S(0,0), S(-7,7), S(-17,10), S(-15,30), S(6,49), S(41,70), S(0,0), S(0,0) };

using PackedNeuralNetworkPtr = std::unique_ptr<nn::PackedNeuralNetwork>;
static PackedNeuralNetworkPtr g_mainNeuralNetwork;
static PackedNeuralNetworkPtr g_endgameNeuralNetwork;

bool LoadMainNeuralNetwork(const char* path)
{
    PackedNeuralNetworkPtr network = std::make_unique<nn::PackedNeuralNetwork>();
    if (network->Load(path))
    {
        g_mainNeuralNetwork = std::move(network);
        std::cout << "info string Loaded neural network: " << path << std::endl;
        return true;
    }

    g_mainNeuralNetwork.reset();
    return false;
}

bool LoadEndgameNeuralNetwork(const char* path)
{
    PackedNeuralNetworkPtr network = std::make_unique<nn::PackedNeuralNetwork>();
    if (network->Load(path))
    {
        g_endgameNeuralNetwork = std::move(network);
        std::cout << "info string Loaded endgame neural network: " << path << std::endl;
        return true;
    }

    g_endgameNeuralNetwork.reset();
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
}

bool TryLoadingDefaultEndgameEvalFile()
{
    // check if there's eval file in same directory as executable
    {
        std::string path = GetDefaultEvalFilePath() + c_DefaultEndgameEvalFile;
        if (!path.empty())
        {
            bool fileExists = false;
            {
                std::ifstream f(path.c_str());
                fileExists = f.good();
            }

            if (fileExists && LoadEndgameNeuralNetwork(path.c_str()))
            {
                return true;
            }
        }
    }

    // try working directory
    {
        bool fileExists = false;
        {
            std::ifstream f(c_DefaultEndgameEvalFile);
            fileExists = f.good();
        }

        if (fileExists && LoadEndgameNeuralNetwork(c_DefaultEndgameEvalFile))
        {
            return true;
        }
    }

    std::cout << "info string Failed to load default neural network " << c_DefaultEvalFile << std::endl;
    return false;
}

static int32_t InterpolateScore(const int32_t phase, const TPieceScore<int32_t>& score)
{
    const int32_t mgPhase = std::min(64, phase);
    const int32_t egPhase = 64 - mgPhase;

    ASSERT(mgPhase >= 0 && mgPhase <= 64);
    ASSERT(egPhase >= 0 && egPhase <= 64);

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

static TPieceScore<int32_t> EvaluatePassedPawns(const Bitboard ourPawns, const Bitboard theirPawns)
{
    TPieceScore<int32_t> score = { 0, 0 };

    ourPawns.Iterate([&](uint32_t square)
    {
        const uint32_t rank = square / 8;
        const uint32_t file = square % 8;

        if (rank < 6)
        {
            constexpr const Bitboard fileMask = Bitboard::FileBitboard<0>();

            Bitboard passedPawnMask = fileMask << (square + 8);

            // blocked pawn
            if (ourPawns & passedPawnMask)
            {
                return;
            }

            if (file > 0) passedPawnMask |= fileMask << (square + 7);
            if (file < 7) passedPawnMask |= fileMask << (square + 9);

            if (theirPawns & passedPawnMask)
            {
                return;
            }

            score += c_passedPawnBonus[rank];
        }
    });

    return score;
}

ScoreType Evaluate(const Position& pos, NodeInfo* nodeInfo, bool useNN)
{
    const MaterialKey materialKey = pos.GetMaterialKey();
    const uint32_t numPieces = pos.GetNumPieces();

    // check endgame evaluation first
    {
        int32_t endgameScore;
        if (EvaluateEndgame(pos, endgameScore))
        {
            ASSERT(endgameScore < TablebaseWinValue && endgameScore > -TablebaseWinValue);
            return (ScoreType)endgameScore;
        }
    }

    const Square whiteKingSq(FirstBitSet(pos.Whites().king));
    const Square blackKingSq(FirstBitSet(pos.Blacks().king));

    const Bitboard whitesOccupied = pos.Whites().Occupied();
    const Bitboard blacksOccupied = pos.Blacks().Occupied();
    const Bitboard allOccupied = whitesOccupied | blacksOccupied;

    const Bitboard whitePawnsAttacks = Bitboard::GetPawnAttacks<Color::White>(pos.Whites().pawns);
    const Bitboard blackPawnsAttacks = Bitboard::GetPawnAttacks<Color::Black>(pos.Blacks().pawns);

    const Bitboard whitesMobilityArea = ~whitesOccupied & ~blackPawnsAttacks;
    const Bitboard blacksMobilityArea = ~blacksOccupied & ~whitePawnsAttacks;

    TPieceScore<int32_t> value = { 0, 0 };

    const int32_t whiteQueens   = materialKey.numWhiteQueens;
    const int32_t whiteRooks    = materialKey.numWhiteRooks;
    const int32_t whiteBishops  = materialKey.numWhiteBishops;
    const int32_t whiteKnights  = materialKey.numWhiteKnights;
    const int32_t whitePawns    = materialKey.numWhitePawns;

    const int32_t blackQueens   = materialKey.numBlackQueens;
    const int32_t blackRooks    = materialKey.numBlackRooks;
    const int32_t blackBishops  = materialKey.numBlackBishops;
    const int32_t blackKnights  = materialKey.numBlackKnights;
    const int32_t blackPawns    = materialKey.numBlackPawns;

    // 0 - endgame, 64 - opening
    const int32_t gamePhase =
        1 * (whitePawns + blackPawns) +
        2 * (whiteKnights + blackKnights) +
        2 * (whiteBishops + blackBishops) +
        4 * (whiteRooks + blackRooks) +
        8 * (whiteQueens + blackQueens);

    int32_t queensDiff = whiteQueens - blackQueens;
    int32_t rooksDiff = whiteRooks - blackRooks;
    int32_t bishopsDiff = whiteBishops - blackBishops;
    int32_t knightsDiff = whiteKnights - blackKnights;
    int32_t pawnsDiff = whitePawns - blackPawns;

    // piece square tables
    value.mg += pos.GetPieceSquareValueMG();
    value.eg += pos.GetPieceSquareValueEG();

    value += c_queenValue * queensDiff;
    value += c_rookValue * rooksDiff;
    value += c_bishopValue * bishopsDiff;
    value += c_knightValue * knightsDiff;
    value += c_pawnValue * pawnsDiff;

    // tempo bonus
    if (pos.GetSideToMove() == Color::White)
    {
        value += c_tempoBonus;
    }
    else
    {
        value -= c_tempoBonus;
    }

    // bishop pair
    {
        if ((pos.Whites().bishops & Bitboard::LightSquares()) && (pos.Whites().bishops & Bitboard::DarkSquares())) value += c_bishopPairBonus;
        if ((pos.Blacks().bishops & Bitboard::LightSquares()) && (pos.Blacks().bishops & Bitboard::DarkSquares())) value -= c_bishopPairBonus;
    }

    //// passed pawns
    //{
    //    value += EvaluatePassedPawns(pos.Whites().pawns, pos.Blacks().pawns);
    //    value -= EvaluatePassedPawns(pos.Blacks().pawns.MirroredVertically(), pos.Whites().pawns.MirroredVertically());
    //}

    /*
    // white pieces
    {
        pos.Whites().pawns.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_ourPawnDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_theirPawnDistanceBonus[Square::Distance(blackKingSq, Square(square))];
        });
        pos.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_ourKnightDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_theirKnightDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_knightMobilityBonus[(Bitboard::GetKnightAttacks(Square(square)) & whitesMobilityArea).Count()];
        });
        pos.Whites().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_ourBishopDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_theirBishopDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_bishopMobilityBonus[(Bitboard::GenerateBishopAttacks(Square(square), allOccupied) & whitesMobilityArea).Count()];
        });
        pos.Whites().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_ourRookDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_theirRookDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_rookMobilityBonus[(Bitboard::GenerateRookAttacks(Square(square), allOccupied) & whitesMobilityArea).Count()];
        });
        pos.Whites().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value += c_ourQueenDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_theirQueenDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_queenMobilityBonus[(Bitboard::GenerateQueenAttacks(Square(square), allOccupied) & whitesMobilityArea).Count()];
        });
    }

    // black pieces
    {
        pos.Blacks().pawns.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_ourPawnDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_theirPawnDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
        });
        pos.Blacks().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_ourKnightDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_theirKnightDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_knightMobilityBonus[(Bitboard::GetKnightAttacks(Square(square)) & blacksMobilityArea).Count()];
        });
        pos.Blacks().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_ourBishopDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_theirBishopDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_bishopMobilityBonus[(Bitboard::GenerateBishopAttacks(Square(square), allOccupied) & blacksMobilityArea).Count()];
        });
        pos.Blacks().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_ourRookDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_theirRookDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_rookMobilityBonus[(Bitboard::GenerateRookAttacks(Square(square), allOccupied) & blacksMobilityArea).Count()];
        });
        pos.Blacks().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            value -= c_ourQueenDistanceBonus[Square::Distance(blackKingSq, Square(square))];
            value += c_theirQueenDistanceBonus[Square::Distance(whiteKingSq, Square(square))];
            value -= c_queenMobilityBonus[(Bitboard::GenerateQueenAttacks(Square(square), allOccupied) & blacksMobilityArea).Count()];
        });
    }
    */

    // accumulate middle/end game scores
    int32_t finalValue = InterpolateScore(gamePhase, value);

    if (useNN)
    {
        const nn::PackedNeuralNetwork* networkToUse = nullptr;
        bool useIncrementalUpdate = false;
        if (numPieces >= 4 && numPieces <= 5)
        {
            networkToUse = g_endgameNeuralNetwork.get();
        }
        if (!networkToUse)
        {
            networkToUse = g_mainNeuralNetwork.get();
            useIncrementalUpdate = true;
        }

        // use neural network for balanced positions
        if (networkToUse && std::abs(finalValue) < c_nnTresholdMax)
        {
            int32_t nnValue = (nodeInfo && useIncrementalUpdate) ?
                NNEvaluator::Evaluate(*networkToUse, *nodeInfo, NetworkInputMapping::Full_Symmetrical) :
                NNEvaluator::Evaluate(*networkToUse, pos, NetworkInputMapping::Full_Symmetrical);

            // convert to centipawn range
            nnValue = (nnValue * c_nnOutputToCentiPawns + nn::OutputScale / 2) / nn::OutputScale;

            // NN output is side-to-move relative
            if (pos.GetSideToMove() == Color::Black) nnValue = -nnValue;

            constexpr int32_t nnBlendRange = c_nnTresholdMax - c_nnTresholdMin;
            const int32_t nnFactor = std::max(0, std::abs(finalValue) - c_nnTresholdMin);
            ASSERT(nnFactor <= nnBlendRange);
            finalValue = (nnFactor * finalValue + nnValue * (nnBlendRange - nnFactor)) / nnBlendRange;
        }
    }

    // saturate eval value so it doesn't exceed KnownWinValue
    if (finalValue > c_evalSaturationTreshold)
    {
        finalValue = c_evalSaturationTreshold + (finalValue - c_evalSaturationTreshold) / 4;
    }

    ASSERT(finalValue > -KnownWinValue && finalValue < KnownWinValue);

    // scale down when approaching 50-move draw
    finalValue = finalValue * (128 - std::max(0, (int32_t)pos.GetHalfMoveCount() - 4)) / 128;

    ASSERT(finalValue > -KnownWinValue && finalValue < KnownWinValue);

    return (ScoreType)finalValue;
}
