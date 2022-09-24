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

static constexpr PieceScore c_tempoBonus = S(10, 5);

static constexpr PieceScore c_ourPawnDistanceBonus[8]       = { S(0,0), S(16,8),   S(7,5),     S(0,0),     S(-7,-6), S(-10,-2), S(-12,-4), S(-16,-4)};
static constexpr PieceScore c_ourKnightDistanceBonus[8]     = { S(0,0), S(4,22),   S(1,15),    S(-3,11),   S(-4,-1), S(-2,-7),  S(-1,-16), S(3,-24) };
static constexpr PieceScore c_ourBishopDistanceBonus[8]     = { S(0,0), S(2,6),    S(-3,4),    S(-4,5),    S(-4,-2), S(-3,-3),  S(1,-1),   S(11,-6) };
static constexpr PieceScore c_ourRookDistanceBonus[8]       = { S(0,0), S(-10,1),  S(-6,4),    S(-4,0),    S(3,-1),  S(2,-2),   S(6,2),    S(7,-4)  };
static constexpr PieceScore c_ourQueenDistanceBonus[8]      = { S(0,0), S(2,-8),   S(-12,1),   S(-11,-4),  S(-7,-5), S(1,-3),   S(10,4),   S(24,5)  };

static constexpr PieceScore c_theirPawnDistanceBonus[8]     = { S(0,0), S(30,39),  S(-11,10),  S(-8,-3),   S(-7,-6), S(-3,-7),  S(3,-9),   S(-1,-15) };
static constexpr PieceScore c_theirKnightDistanceBonus[8]   = { S(0,0), S(-17,0),  S(-13,-2),  S(-7,-8),   S(1,-5),  S(5,-2),   S(12,3),   S(16,16)  };
static constexpr PieceScore c_theirBishopDistanceBonus[8]   = { S(0,0), S(5,11),   S(-15,0),   S(-2,-2),   S(2,4),   S(2,-2),   S(3,-2),   S(7,-1)   };
static constexpr PieceScore c_theirRookDistanceBonus[8]     = { S(0,0), S(-7,2),   S(-4,1),    S(-1,1),    S(0,1),   S(1,-4),   S(4,-1),   S(7,-3)   };
static constexpr PieceScore c_theirQueenDistanceBonus[8]    = { S(0,0), S(-99,-99),S(-45,-29), S(-18,-16), S(1,-5),  S(13,6),   S(22,16),  S(30,28)  };

static constexpr PieceScore c_knightMobilityBonus[9] =
    { S(-35,-62), S(-14,-24), S(-3,-3), S(3,6), S(10,11), S(16,16), S(23,19), S(27,21), S(29,23) };

static constexpr PieceScore c_bishopMobilityBonus[14] =
    { S(-25,-41), S(-15, -4), S(-3, 7),  S(7, 19), S(14, 27), S(20, 34),
      S( 24, 38), S( 26, 40), S(29, 42), S(31,44), S(33, 45), S(35, 46),
      S( 36, 47), S( 37, 48) };

static constexpr PieceScore c_rookMobilityBonus[15] =
    { S(-10,-38), S(-6,-7), S(-3, 14), S(1, 26), S(5, 36), S(12,40),
      S( 16, 43), S(20,45), S(25, 48), S(29,51), S(33,53), S(37,55),
      S( 40, 57), S(42,60), S(43, 63) };

static constexpr PieceScore c_queenMobilityBonus[28] =
    { S(-22,-65), S(-17,-37), S(-10, -1), S( -5, 35), S( 0, 50), S( 4, 62),
      S( 6,  74), S( 10, 84), S( 16, 87), S( 19, 95), S(21,102), S(22,105),
      S( 24,108), S( 26,113), S( 28,115), S( 30,117), S(33,119), S(37,121),
      S( 39,123), S( 42,125), S( 45,127), S( 47,129), S(49,131), S(51,133),
      S( 53,136), S( 55,139), S( 57,142), S( 60,144) };

static constexpr PieceScore c_passedPawnBonus[8]            = { S(0,0), S(0,5), S(-5,6), S(1,20), S(22,40), S(52,70), S(0,0), S(0,0) };

alignas(CACHELINE_SIZE)
const PieceScore PSQT[6][Square::NumSquares] =
{
    {
        S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
        S( -48, -54), S( -23, -21), S( -35, -25), S( -56, -40), S( -55, -36), S( -29, -20), S( -19, -16), S( -45, -49),
        S( -45, -51), S( -19, -28), S( -44, -36), S( -53, -43), S( -54, -48), S( -41, -35), S( -18, -29), S( -41, -48),
        S( -43, -43), S( -14, -25), S( -27, -45), S( -27, -52), S( -25, -54), S( -30, -39), S( -16, -27), S( -40, -47),
        S( -23, -15), S(  -7, -14), S( -13, -24), S(   5, -31), S(   9, -29), S(  -5, -22), S(   2, -14), S( -21, -17),
        S(  -9,  28), S(  19,  47), S(  32,  24), S(  39,  16), S(  34,  26), S(  23,  31), S(  37,  53), S(  -3,  20),
        S(  78, 120), S(  85, 126), S(  99,  97), S(  97,  83), S(  95,  73), S(  98,  99), S( 100, 113), S(  85, 127),
        S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
    },
    {
        S( -79, -57), S( -37, -37), S( -38, -27), S( -35, -18), S( -42, -22), S( -38, -22), S( -38, -34), S( -65, -54),
        S( -32, -39), S( -35, -20), S( -24, -12), S( -19,  -4), S( -19,   0), S( -16, -11), S( -28, -28), S( -36, -43),
        S( -36, -18), S( -16,   5), S(  -6,   9), S(   5,  23), S(   9,  18), S(  -7,  15), S( -14,   3), S( -32, -22),
        S(  -9,  -3), S(   8,  12), S(  16,  23), S(  13,  28), S(   9,  35), S(  11,  28), S(  20,  17), S( -12,   2),
        S(  11,  -6), S(  20,  20), S(  36,  40), S(  35,  33), S(  33,  37), S(  36,  35), S(  14,  19), S(  20,  -1),
        S(   7,   0), S(  41,  16), S(  62,  37), S(  51,  35), S(  61,  35), S(  59,  23), S(  42,  15), S(   9,  -8),
        S(  -5, -15), S(  15,  -5), S(  47,  22), S(  48,  34), S(  37,  22), S(  53,  18), S(  15,   0), S(  -3, -19),
        S( -96, -53), S(  11,  -5), S(   2,  -7), S(   5,   6), S(  16,   0), S(  -2, -25), S(  15,  -8), S( -75, -42),
    },
    {
        S( -27, -26), S(  -9,  -1), S( -32, -15), S( -35,  -9), S( -39, -11), S( -35, -21), S(  -1, -15), S( -32, -25),
        S(   0, -21), S( -13,  -3), S(  -8,   0), S( -21,  -4), S( -21,  -7), S(  -8,   0), S( -15,  -7), S(   1, -12),
        S( -10, -16), S(  -2,   5), S(  -7,   1), S(   3,  10), S(   1,   9), S( -14,   3), S(   2,  -5), S( -10,  -6),
        S(   3,  -7), S(  -3,   6), S(   1,  22), S(  11,  21), S(  10,  18), S(   0,  11), S(  -1,   1), S(   5, -12),
        S(  -9,  -9), S(   7,   6), S(  16,   7), S(  33,  23), S(  28,  30), S(  26,  14), S(   1,  14), S( -15,  -7),
        S(   9,  -1), S(  17,   7), S(  35,  19), S(  32,  10), S(  38,   3), S(  32,  17), S(  28,  10), S(  11,   0),
        S( -21, -18), S(  -2,  -1), S(  22,  13), S(  18,  11), S(  12,  13), S(  13,   0), S(  15,   7), S( -15, -13),
        S( -13, -17), S(  15,  -1), S( -10,  -7), S(   3,   9), S(   4,   8), S( -27, -17), S(   9,   3), S(  -9, -20),
    },
    {
        S( -41, -27), S( -30, -23), S( -34, -22), S( -21, -14), S( -18,  -8), S( -30, -24), S( -31, -21), S( -43, -35),
        S( -56, -32), S( -33, -20), S( -25, -27), S( -28, -24), S( -29, -14), S( -34, -25), S( -34, -33), S( -47, -32),
        S( -34, -26), S( -32, -19), S( -40, -14), S( -26, -15), S( -25,  -9), S( -32,  -2), S( -25, -33), S( -39, -21),
        S( -28, -18), S( -16,   0), S( -23,   2), S( -12,  -2), S(  -8,  -1), S( -18,   0), S( -10,  -3), S( -30, -16),
        S(   0,   6), S(  10,  12), S(  17,  10), S(  15,   7), S(  19,  12), S(  11,  12), S(  11,  11), S(   7,  -5),
        S(  12,  19), S(  32,  10), S(  32,  20), S(  40,  24), S(  56,  24), S(  37,  18), S(  33,  15), S(   6,  13),
        S(  32,  17), S(  32,  22), S(  36,  27), S(  45,  33), S(  53,  31), S(  46,  34), S(  27,  21), S(  30,  18),
        S(  40,   7), S(  33,  11), S(  34,  23), S(  35,  30), S(  38,  32), S(  42,  17), S(  43,  16), S(  29,   8),
    },
    {
        S( -15, -33), S( -29, -49), S( -34, -45), S( -29, -32), S( -31, -37), S( -35, -44), S( -24, -48), S( -16, -45),
        S( -17, -45), S( -21, -35), S( -17, -30), S( -20, -22), S( -17, -25), S( -18, -22), S( -13, -31), S( -18, -48),
        S( -19, -39), S( -10,  -9), S( -10,  -3), S( -18,  -6), S( -15, -10), S( -15, -13), S(  -9, -14), S( -12, -30),
        S( -24, -15), S(  -7,  -2), S(  -9,  18), S( -12,  16), S(  -6,  17), S(  -7,  14), S(  -2,   3), S( -13, -11),
        S(  -6,  -4), S(   1,  11), S(   8,  17), S(  23,  39), S(  19,  42), S(  18,  23), S(  -2,  11), S(   0,  -7),
        S(   6,  -5), S(  19,  19), S(   9,  39), S(  28,  49), S(  30,  38), S(  31,  36), S(  24,  24), S(   6,   3),
        S(  -1, -14), S( -12,  13), S(  34,  37), S(  33,  42), S(  25,  38), S(  33,  38), S(   3,   1), S(  12,   0),
        S(  29,   7), S(  16,  22), S(  35,  22), S(  31,  36), S(  23,  31), S(  24,  40), S(  24,   8), S(  27,  12),
    },
    {
        S(   4, -81), S(  21, -65), S(   5, -70), S( -35, -78), S( -12, -87), S(  -1, -67), S(  17, -60), S(   5, -71),
        S(  10, -55), S(   5, -51), S(   7, -52), S( -20, -48), S( -15, -50), S(   3, -49), S(  16, -45), S(  10, -50),
        S(   0, -52), S(  -5, -40), S( -10, -38), S(  13, -21), S( -15, -28), S(  -5, -27), S(   0, -35), S(   0, -48),
        S( -15, -33), S( -10, -10), S( -10,  -7), S( -10,   0), S( -15,  -4), S( -10,   0), S( -10,  -6), S( -15, -33),
        S( -25,  -4), S( -20,  32), S( -20,  18), S( -20,  24), S( -20,  24), S( -20,  28), S( -20,  31), S( -25,   6),
        S( -35,  26), S( -30,  68), S( -30,  59), S( -30,  61), S( -30,  61), S( -30,  63), S( -30,  70), S( -35,  23),
        S( -45,  26), S( -40,  83), S( -40,  71), S( -40,  64), S( -40,  61), S( -40,  72), S( -40,  64), S( -45,  24),
        S( -55, -84), S( -50,  64), S( -50,  96), S( -50,  74), S( -50,  76), S( -50,  86), S( -50,  70), S( -55, -73),
    }
};

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
    // 32 at the beginning, 0 at the end
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

    // TODO for some reason this makes engine weaker...
    /*
    // passed pawns
    {
        value += EvaluatePassedPawns(pos.Whites().pawns, pos.Blacks().pawns);
        value -= EvaluatePassedPawns(pos.Blacks().pawns.MirroredVertically(), pos.Whites().pawns.MirroredVertically());
    }
    */

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

    // tempo bonus
    if (pos.GetSideToMove() == Color::White)
    {
        value += c_tempoBonus;
    }
    else
    {
        value -= c_tempoBonus;
    }

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
