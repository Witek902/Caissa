#include "Evaluate.hpp"
#include "Move.hpp"
#include "Material.hpp"
#include "Endgame.hpp"
#include "PackedNeuralNetwork.hpp"
#include "PieceSquareTables.h"

#include "nnue-probe/nnue.h"

#include <unordered_map>

#define S(mg, eg) PieceScore{ mg, eg }

static constexpr int32_t c_evalSaturationTreshold   = 6000;

static constexpr int32_t c_castlingRightsBonus  = 5;
static constexpr int32_t c_doubledPawnPenalty   = 0;
static constexpr int32_t c_noPawnPenalty        = 120;
static constexpr int32_t c_passedPawnBonus      = 0;

const PieceScore PawnPSQT[Square::NumSquares] =
{
    S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
    S(  98, 178), S( 134, 173), S(  61, 158), S(  95, 134), S(  68, 147), S( 126, 132), S(  34, 165), S( -11, 187),
    S(  -6,  94), S(   7, 100), S(  26,  85), S(  31,  67), S(  65,  56), S(  56,  53), S(  25,  82), S( -20,  84),
    S( -14,  32), S(  13,  24), S(   6,  13), S(  21,   5), S(  23,  -2), S(  12,   4), S(  17,  17), S( -23,  17),
    S( -27,  13), S(  -2,   9), S(  -5,  -3), S(  12,  -7), S(  17,  -7), S(   6,  -8), S(  10,   3), S( -25,  -1),
    S( -26,   4), S(  -4,   7), S(  -4,  -6), S( -10,   1), S(   3,   0), S(   3,  -5), S(  33,  -1), S( -12,  -8),
    S( -35,  13), S(  -1,   8), S( -20,   8), S( -23,  10), S( -15,  13), S(  24,   0), S(  38,   2), S( -22,  -7),
    S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
};

const PieceScore KnightPSQT[Square::NumSquares] =
{
    S(-167, -58), S( -89, -38), S( -34, -13), S( -49, -28), S(  61, -31), S( -97, -27), S( -15, -63), S(-107, -99),
    S( -73, -25), S( -41,  -8), S(  72, -25), S(  36,  -2), S(  23,  -9), S(  62, -25), S(   7, -24), S( -17, -52),
    S( -47, -24), S(  60, -20), S(  37,  10), S(  65,   9), S(  84,  -1), S( 129,  -9), S(  73, -19), S(  44, -41),
    S(  -9, -17), S(  17,   3), S(  19,  22), S(  53,  22), S(  37,  22), S(  69,  11), S(  18,   8), S(  22, -18),
    S( -13, -18), S(   4,  -6), S(  16,  16), S(  13,  25), S(  28,  16), S(  19,  17), S(  21,   4), S(  -8, -18),
    S( -23, -23), S(  -9,  -3), S(  12,  -1), S(  10,  15), S(  19,  10), S(  17,  -3), S(  25, -20), S( -16, -22),
    S( -29, -42), S( -53, -20), S( -12, -10), S(  -3,  -5), S(  -1,  -2), S(  18, -20), S( -14, -23), S( -19, -44),
    S(-105, -29), S( -21, -51), S( -58, -23), S( -33, -15), S( -17, -22), S( -28, -18), S( -19, -50), S( -23, -64),
};

const PieceScore BishopPSQT[Square::NumSquares] =
{
    S( -29, -14), S(   4, -21), S( -82, -11), S( -37,  -8), S( -25,  -7), S( -42,  -9), S(   7, -17), S(  -8, -24),
    S( -26,  -8), S(  16,  -4), S( -18,   7), S( -13, -12), S(  30,  -3), S(  59, -13), S(  18,  -4), S( -47, -14),
    S( -16,   2), S(  37,  -8), S(  43,   0), S(  40,  -1), S(  35,  -2), S(  50,   6), S(  37,   0), S(  -2,   4),
    S(  -4,  -3), S(   5,   9), S(  19,  12), S(  50,   9), S(  37,  14), S(  37,  10), S(   7,   3), S(  -2,   2),
    S(  -6,  -6), S(  13,   3), S(  13,  13), S(  26,  19), S(  34,   7), S(  12,  10), S(  10,  -3), S(   4,  -9),
    S(   0, -12), S(  15,  -3), S(  15,   8), S(  15,  10), S(  14,  13), S(  27,   3), S(  18,  -7), S(  10, -15),
    S(   4, -14), S(  15, -18), S(  16,  -7), S(   0,  -1), S(   7,   4), S(  21,  -9), S(  33, -15), S(   1, -27),
    S( -33, -23), S(  -3,  -9), S( -14, -23), S( -21,  -5), S( -13,  -9), S( -12, -16), S( -39,  -5), S( -21, -17),
};

const PieceScore RookPSQT[Square::NumSquares] =
{
    S(  32,  13), S(  42,  10), S(  32,  18), S(  51,  15), S(  63,  12), S(   9,  12), S(  31,   8), S(  43,   5),
    S(  27,  11), S(  32,  13), S(  58,  13), S(  62,  11), S(  80,  -3), S(  67,   3), S(  26,   8), S(  44,   3),
    S(  -5,   7), S(  19,   7), S(  26,   7), S(  36,   5), S(  17,   4), S(  45,  -3), S(  61,  -5), S(  16,  -3),
    S( -24,   4), S( -11,   3), S(   7,  13), S(  26,   1), S(  24,   2), S(  35,   1), S(  -8,  -1), S( -20,   2),
    S( -36,   3), S( -26,   5), S( -12,   8), S(  -1,   4), S(   9,  -5), S(  -7,  -6), S(   6,  -8), S( -23, -11),
    S( -45,  -4), S( -25,   0), S( -16,  -5), S( -17,  -1), S(   3,  -7), S(   0, -12), S(  -5,  -8), S( -33, -16),
    S( -44,  -6), S( -16,  -6), S( -20,   0), S(  -9,   2), S(  -1,  -9), S(  11,  -9), S(  -6, -11), S( -71,  -3),
    S( -19,  -9), S( -13,   2), S(   1,   3), S(  17,  -1), S(  16,  -5), S(   7, -13), S( -37,   4), S( -26, -20),
};

const PieceScore QueenPSQT[Square::NumSquares] =
{
    S( -28,  -9), S(   0,  22), S(  29,  22), S(  12,  27), S(  59,  27), S(  44,  19), S(  43,  10), S(  45,  20),
    S( -24, -17), S( -39,  20), S(  -5,  32), S(   1,  41), S( -16,  58), S(  57,  25), S(  28,  30), S(  54,   0),
    S( -13, -20), S( -17,   6), S(   7,   9), S(   8,  49), S(  29,  47), S(  56,  35), S(  47,  19), S(  57,   9),
    S( -27,   3), S( -27,  22), S( -16,  24), S( -16,  45), S(  -1,  57), S(  17,  40), S(  -2,  57), S(   1,  36),
    S(  -9, -18), S( -26,  28), S(  -9,  19), S( -10,  47), S(  -2,  31), S(  -4,  34), S(   3,  39), S(  -3,  23),
    S( -14, -16), S(   2, -27), S( -11,  15), S(  -2,   6), S(  -5,   9), S(   2,  17), S(  14,  10), S(   5,   5),
    S( -35, -22), S(  -8, -23), S(  11, -30), S(   2, -16), S(   8, -16), S(  15, -23), S(  -3, -36), S(   1, -32),
    S(  -1, -33), S( -18, -28), S(  -9, -22), S(  10, -43), S( -15,  -5), S( -25, -32), S( -31, -20), S( -50, -41),
};

const PieceScore KingPSQT[Square::NumSquares] =
{
    S( -65, -74), S(  23, -35), S(  16, -18), S( -15, -18), S( -56, -11), S( -34,  15), S(   2,   4), S(  13, -17),
    S(  29, -12), S(  -1,  17), S( -20,  14), S(  -7,  17), S(  -8,  17), S(  -4,  38), S( -38,  23), S( -29,  11),
    S(  -9,  10), S(  24,  17), S(   2,  23), S( -16,  15), S( -20,  20), S(   6,  45), S(  22,  44), S( -22,  13),
    S( -17,  -8), S( -20,  22), S( -12,  24), S( -27,  27), S( -30,  26), S( -25,  33), S( -14,  26), S( -36,   3),
    S( -49, -18), S(  -1,  -4), S( -27,  21), S( -39,  24), S( -46,  27), S( -44,  23), S( -33,   9), S( -51, -11),
    S( -14, -19), S( -14,  -3), S( -22,  11), S( -46,  21), S( -44,  23), S( -30,  16), S( -15,   7), S( -27,  -9),
    S(   1, -27), S(   7, -11), S(  -8,   4), S( -64,  13), S( -43,  14), S( -16,   4), S(   9,  -5), S(   8, -17),
    S( -15, -53), S(  36, -34), S(  12, -21), S( -54, -11), S(   8, -28), S( -28, -14), S(  24, -24), S(  14, -43),
};

using PackedNeuralNetworkPtr = std::unique_ptr<nn::PackedNeuralNetwork>;
static std::unordered_map<MaterialKey, PackedNeuralNetworkPtr> g_neuralNetworks;

static void LoadNetworkForMaterialKey(const MaterialKey key)
{
    const std::string path = "../../networks/" + key.ToString() + ".pnn";

    PackedNeuralNetworkPtr network = std::make_unique<nn::PackedNeuralNetwork>();
    if (network->Load(path.c_str()))
    {
        g_neuralNetworks[key] = std::move(network);
    }
}

void InitEvaluation()
{
    //LoadNetworkForMaterialKey({ 1,0,0,1,0,1,0,0,1,0 });
    //LoadNetworkForMaterialKey({ 1,0,0,0,0,1,0,0,0,0 });
    //LoadNetworkForMaterialKey({ 1,0,0,1,0,0,0,0,1,0 });
}

static uint32_t FlipRank(uint32_t square)
{
    uint32_t rank = square / 8;
    uint32_t file = square % 8;
    square = 8u * (7u - rank) + file;
    return square;
}

INLINE static void EvalWhitePieceSquareTable(const Bitboard bitboard, const PieceScore* scores, int32_t& outScoreMG, int32_t& outScoreEG)
{
    bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        square = FlipRank(square);
        ASSERT(square < 64);
        outScoreMG += scores[square].mg;
        outScoreEG += scores[square].eg;
    });
}

INLINE static void EvalBlackPieceSquareTable(const Bitboard bitboard, const PieceScore* scores, int32_t& outScoreMG, int32_t& outScoreEG)
{
    bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        ASSERT(square < 64);
        outScoreMG -= scores[square].mg;
        outScoreEG -= scores[square].eg;
    });
}

static int32_t InterpolateScore(const Position& pos, int32_t mgScore, int32_t egScore)
{
    // 32 at the beginning, 0 at the end
    const int32_t mgPhase = std::min(32u, (pos.Whites().Occupied() | pos.Blacks().Occupied()).Count());
    const int32_t egPhase = 32 - mgPhase;

    ASSERT(mgPhase >= 0 && mgPhase <= 32);
    ASSERT(egPhase >= 0 && egPhase <= 32);

    return (mgScore * mgPhase + egScore * egPhase) / 32;
}

bool CheckInsufficientMaterial(const Position& position)
{
    const Bitboard queensRooksPawns =
        position.Whites().queens | position.Whites().rooks | position.Whites().pawns |
        position.Blacks().queens | position.Blacks().rooks | position.Blacks().pawns;

    if (queensRooksPawns != 0)
    {
        return false;
    }

    if (position.Whites().knights == 0 && position.Blacks().knights == 0)
    {
        // king and bishop vs. king
        if ((position.Whites().bishops == 0 && position.Blacks().bishops.Count() <= 1) ||
            (position.Whites().bishops.Count() <= 1 && position.Blacks().bishops == 0))
        {
            return true;
        }

        // king and bishop vs. king and bishop (bishops on the same color squares)
        if (position.Whites().bishops.Count() == 1 && position.Blacks().bishops.Count() == 1)
        {
            const bool whiteBishopOnLightSquare = (position.Whites().bishops & Bitboard::LightSquares()) != 0;
            const bool blackBishopOnLightSquare = (position.Blacks().bishops & Bitboard::LightSquares()) != 0;
            return whiteBishopOnLightSquare == blackBishopOnLightSquare;
        }
    }


    // king and knight vs. king
    if (position.Whites().bishops == 0 && position.Blacks().bishops == 0)
    {
        if ((position.Whites().knights == 0 && position.Blacks().knights.Count() <= 1) ||
            (position.Whites().knights.Count() <= 1 && position.Blacks().knights == 0))
        {
            return true;
        }
    }

    return false;
}

static int32_t CountPassedPawns(const Bitboard ourPawns, const Bitboard theirPawns)
{
    int32_t count = 0;

    ourPawns.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        const uint32_t rank = square / 8;
        const uint32_t file = square % 8;
        
        if (rank >= 6)
        {
            // pawn is ready to promotion - consider is as passed
            count++;
        }
        else
        {
            constexpr const Bitboard fileMask = Bitboard::FileBitboard<0>();

            Bitboard passedPawnMask = fileMask << (square + 8);
            if (file > 0) passedPawnMask |= fileMask << (square + 7);
            if (file < 7) passedPawnMask |= fileMask << (square + 9);

            if ((theirPawns & passedPawnMask) == 0)
            {
                count++;
            }
        }
    });

    return count;
}

static int32_t EvaluateStockfishNNUE(const Position& position, NNUEdata** nnueData)
{
    int32_t pieces[64 + 1];
    int32_t squares[64 + 1];

    size_t index = 2;

    pieces[0] = pieces::wking;
    squares[0] = FirstBitSet(position.Whites().king);

    pieces[1] = pieces::bking;
    squares[1] = FirstBitSet(position.Blacks().king);

    position.Whites().pawns.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::wpawn;
        squares[index] = square;
        index++;
    });

    position.Blacks().pawns.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::bpawn;
        squares[index] = square;
        index++;
    });

    position.Whites().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::wknight;
        squares[index] = square;
        index++;
    });

    position.Blacks().knights.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::bknight;
        squares[index] = square;
        index++;
    });

    position.Whites().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::wbishop;
        squares[index] = square;
        index++;
    });

    position.Blacks().bishops.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::bbishop;
        squares[index] = square;
        index++;
    });

    position.Whites().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::wrook;
        squares[index] = square;
        index++;
    });

    position.Blacks().rooks.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::brook;
        squares[index] = square;
        index++;
    });

    position.Whites().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::wqueen;
        squares[index] = square;
        index++;
    });

    position.Blacks().queens.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        pieces[index] = pieces::bqueen;
        squares[index] = square;
        index++;
    });

    pieces[index] = 0;
    squares[index] = 0;

    int32_t score = nnueData ?
        nnue_evaluate_incremental(position.GetSideToMove() == Color::White ? 0 : 1, pieces, squares, nnueData) :
        nnue_evaluate(position.GetSideToMove() == Color::White ? 0 : 1, pieces, squares);

    if (position.GetSideToMove() == Color::Black)
    {
        score = -score;
    }

    return score;
}

static int32_t EvaluateNeuralNetwork(nn::PackedNeuralNetwork& network, const Position& position)
{
    uint16_t features[64];
    const uint32_t numFeatures = position.ToPackedFeaturesVector(features);

    float nnOutput = (float)network.Run(features, numFeatures) / (float)nn::OutputScale;
    nnOutput = std::clamp(nnOutput, 0.001f, 0.999f);
    nnOutput = 100.0f * WinProbabilityToPawns(nnOutput);

    return std::clamp<int32_t>((int32_t)nnOutput, -MaxNNScore, MaxNNScore);
}

ScoreType Evaluate(const Position& position, NNUEdata** nnueData)
{
    const MaterialKey materialKey = position.GetMaterialKey();

    // check endgame evaluation first
    {
        int32_t endgameScore;
        if (EvaluateEndgame(position, endgameScore))
        {
            ASSERT(endgameScore < TablebaseWinValue && endgameScore > -TablebaseWinValue);
            return (ScoreType)endgameScore;
        }
    }

    /*
    {
        auto iter = g_neuralNetworks.find(materialKey);
        if (iter != g_neuralNetworks.end())
        {
            int32_t nnScore = EvaluateNeuralNetwork(*(iter->second), position);
            ASSERT(nnScore < TablebaseWinValue&& nnScore > -TablebaseWinValue);
            return (ScoreType)nnScore;
        }

        if (!materialKey.IsSymetric())
        {
            iter = g_neuralNetworks.find(materialKey.SwappedColors());
            if (iter != g_neuralNetworks.end())
            {
                int32_t nnScore = -EvaluateNeuralNetwork(*(iter->second), position.SwappedColors());
                ASSERT(nnScore < TablebaseWinValue&& nnScore > -TablebaseWinValue);
                return (ScoreType)nnScore;
            }
        }
    }
    */

    int32_t value = 0;
    int32_t valueMG = 0;
    int32_t valueEG = 0;

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

    if (whitePawns == 0)
    {
        value -= c_noPawnPenalty;
    }

    if (blackPawns == 0)
    {
        value += c_noPawnPenalty;
    }

    int32_t queensDiff = whiteQueens - blackQueens;
    int32_t rooksDiff = whiteRooks - blackRooks;
    int32_t bishopsDiff = whiteBishops - blackBishops;
    int32_t knightsDiff = whiteKnights - blackKnights;
    int32_t pawnsDiff = whitePawns - blackPawns;

    valueMG += c_queenValue.mg * queensDiff;
    valueMG += c_rookValue.mg * rooksDiff;
    valueMG += c_bishopValue.mg * bishopsDiff;
    valueMG += c_knightValue.mg * knightsDiff;
    valueMG += c_pawnValue.mg * pawnsDiff;

    valueEG += c_queenValue.eg * queensDiff;
    valueEG += c_rookValue.eg * rooksDiff;
    valueEG += c_bishopValue.eg * bishopsDiff;
    valueEG += c_knightValue.eg * knightsDiff;
    valueEG += c_pawnValue.eg * pawnsDiff;

    value += c_castlingRightsBonus * ((int32_t)PopCount(position.GetWhitesCastlingRights()) - (int32_t)PopCount(position.GetBlacksCastlingRights()));

    // piece square tables
    {
        EvalWhitePieceSquareTable(position.Whites().pawns,      PawnPSQT,   valueMG, valueEG);
        EvalWhitePieceSquareTable(position.Whites().knights,    KnightPSQT, valueMG, valueEG);
        EvalWhitePieceSquareTable(position.Whites().bishops,    BishopPSQT, valueMG, valueEG);
        EvalWhitePieceSquareTable(position.Whites().rooks,      RookPSQT,   valueMG, valueEG);
        EvalWhitePieceSquareTable(position.Whites().queens,     QueenPSQT,  valueMG, valueEG);
        EvalWhitePieceSquareTable(position.Whites().king,       KingPSQT,   valueMG, valueEG);

        EvalBlackPieceSquareTable(position.Blacks().pawns,      PawnPSQT,   valueMG, valueEG);
        EvalBlackPieceSquareTable(position.Blacks().knights,    KnightPSQT, valueMG, valueEG);
        EvalBlackPieceSquareTable(position.Blacks().bishops,    BishopPSQT, valueMG, valueEG);
        EvalBlackPieceSquareTable(position.Blacks().rooks,      RookPSQT,   valueMG, valueEG);
        EvalBlackPieceSquareTable(position.Blacks().queens,     QueenPSQT,  valueMG, valueEG);
        EvalBlackPieceSquareTable(position.Blacks().king,       KingPSQT,   valueMG, valueEG);
    }

    // white doubled pawns
    {
        const Bitboard pawns = position.Whites().pawns;

        int32_t numDoubledPawns = 0;
        for (uint32_t file = 0; file < 8u; ++file)
        {
            int32_t numPawnsInFile = (pawns & Bitboard::FileBitboard(file)).Count();
            numDoubledPawns += std::max(0, numPawnsInFile - 1);
        }

        value -= c_doubledPawnPenalty * numDoubledPawns;
    }

    // black doubled pawns
    {
        const Bitboard pawns = position.Blacks().pawns;

        int32_t numDoubledPawns = 0;
        for (uint32_t file = 0; file < 8u; ++file)
        {
            int32_t numPawnsInFile = (pawns & Bitboard::FileBitboard(file)).Count();
            numDoubledPawns += std::max(0, numPawnsInFile - 1);
        }

        value += c_doubledPawnPenalty * numDoubledPawns;
    }

    // white passed pawns
    {
        const int32_t numPassedWhitePawns = CountPassedPawns(position.Whites().pawns, position.Blacks().pawns);
        const int32_t numPassedBlackPawns = CountPassedPawns(position.Blacks().pawns.MirroredVertically(), position.Whites().pawns.MirroredVertically());

        value += (numPassedWhitePawns - numPassedBlackPawns) * c_passedPawnBonus;
    }

    // tempo bonus
    if (position.GetSideToMove() == Color::White)
    {
        value += position.Whites().Occupied().Count();
    }
    else
    {
        value += position.Blacks().Occupied().Count();
    }

    // accumulate middle/end game scores
    value += InterpolateScore(position, valueMG, valueEG);

    // saturate eval value so it doesn't exceed KnownWinValue
    if (value > c_evalSaturationTreshold)
    {
        value = c_evalSaturationTreshold + (value - c_evalSaturationTreshold) / 4;
    }

    ASSERT(value > -KnownWinValue && value < KnownWinValue);

    constexpr int32_t nnueTreshold = 512;

    // use NNUE for balanced positions
    if (nnue_is_valid() && value < nnueTreshold && value > -nnueTreshold)
    {
        const int32_t nnueValue = EvaluateStockfishNNUE(position, nnueData);
        const int32_t nnueFactor = std::abs(value);
        value = (nnueFactor * value + nnueValue * (nnueTreshold - 1 - nnueFactor)) / nnueTreshold;
    }

    // scale down when approaching 50-move draw
    value = value * (128 - std::max(0, (int32_t)position.GetHalfMoveCount() - 4)) / 128;

    ASSERT(value > -KnownWinValue && value < KnownWinValue);

    return (ScoreType)value;
}
