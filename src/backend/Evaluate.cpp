#include "Evaluate.hpp"
#include "Move.hpp"
#include "Material.hpp"
#include "Endgame.hpp"
#include "PackedNeuralNetwork.hpp"
#include "PieceSquareTables.h"
#include "NeuralNetworkEvaluator.hpp"
#include "Pawns.hpp"

#include <unordered_map>
#include <fstream>
#include <memory>

const char* c_DefaultEvalFile = "eval-2.pnn";

#define S(mg, eg) PieceScore{ mg, eg }

static constexpr int32_t c_evalSaturationTreshold   = 4000;

static constexpr int32_t c_castlingRightsBonus  = 5;
static constexpr int32_t c_doubledPawnPenalty   = 0;
static constexpr int32_t c_passedPawnBonus      = 0;

const PieceScore PawnPSQT[Square::NumSquares] =
{
    S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
    S( -48, -54), S( -23, -21), S( -35, -25), S( -56, -40), S( -55, -36), S( -29, -20), S( -19, -16), S( -45, -49),
    S( -45, -51), S( -19, -28), S( -44, -36), S( -53, -43), S( -54, -48), S( -41, -35), S( -18, -29), S( -41, -48),
    S( -43, -43), S( -14, -25), S( -27, -45), S( -27, -52), S( -25, -54), S( -30, -39), S( -16, -27), S( -40, -47),
    S( -23, -15), S(  -7, -14), S( -13, -24), S(   5, -31), S(   9, -29), S(  -5, -22), S(   2, -14), S( -21, -17),
    S(  -9,  28), S(  19,  47), S(  32,  24), S(  39,  16), S(  34,  26), S(  23,  31), S(  37,  53), S(  -3,  20),
    S(  78, 120), S(  85, 126), S(  99,  97), S(  97,  83), S(  95,  73), S(  98,  99), S( 100, 113), S(  85, 127),
    S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0), S(   0,   0),
};

const PieceScore KnightPSQT[Square::NumSquares] =
{
    S( -79, -57), S( -37, -37), S( -38, -27), S( -35, -18), S( -42, -22), S( -38, -22), S( -38, -34), S( -65, -54),
    S( -32, -39), S( -35, -20), S( -24, -12), S( -19,  -4), S( -19,   0), S( -16, -11), S( -28, -28), S( -36, -43),
    S( -36, -18), S( -16,   5), S(  -6,   9), S(   5,  23), S(   9,  18), S(  -7,  15), S( -14,   3), S( -32, -22),
    S(  -9,  -3), S(   8,  12), S(  16,  23), S(  13,  28), S(   9,  35), S(  11,  28), S(  20,  17), S( -12,   2),
    S(  11,  -6), S(  20,  20), S(  36,  40), S(  35,  33), S(  33,  37), S(  36,  35), S(  14,  19), S(  20,  -1),
    S(   7,   0), S(  41,  16), S(  62,  37), S(  51,  35), S(  61,  35), S(  59,  23), S(  42,  15), S(   9,  -8),
    S(  -5, -15), S(  15,  -5), S(  47,  22), S(  48,  34), S(  37,  22), S(  53,  18), S(  15,   0), S(  -3, -19),
    S( -96, -53), S(  11,  -5), S(   2,  -7), S(   5,   6), S(  16,   0), S(  -2, -25), S(  15,  -8), S( -75, -42),
};

const PieceScore BishopPSQT[Square::NumSquares] =
{
    S( -27, -26), S(  -9,  -1), S( -32, -15), S( -35,  -9), S( -39, -11), S( -35, -21), S(  -1, -15), S( -32, -25),
    S(   0, -21), S( -13,  -3), S(  -8,   0), S( -21,  -4), S( -21,  -7), S(  -8,   0), S( -15,  -7), S(   1, -12),
    S( -10, -16), S(  -2,   5), S(  -7,   1), S(   3,  10), S(   1,   9), S( -14,   3), S(   2,  -5), S( -10,  -6),
    S(   3,  -7), S(  -3,   6), S(   1,  22), S(  11,  21), S(  10,  18), S(   0,  11), S(  -1,   1), S(   5, -12),
    S(  -9,  -9), S(   7,   6), S(  16,   7), S(  33,  23), S(  28,  30), S(  26,  14), S(   1,  14), S( -15,  -7),
    S(   9,  -1), S(  17,   7), S(  35,  19), S(  32,  10), S(  38,   3), S(  32,  17), S(  28,  10), S(  11,   0),
    S( -21, -18), S(  -2,  -1), S(  22,  13), S(  18,  11), S(  12,  13), S(  13,   0), S(  15,   7), S( -15, -13),
    S( -13, -17), S(  15,  -1), S( -10,  -7), S(   3,   9), S(   4,   8), S( -27, -17), S(   9,   3), S(  -9, -20),
};

const PieceScore RookPSQT[Square::NumSquares] =
{
    S( -41, -27), S( -30, -23), S( -34, -22), S( -21, -14), S( -18,  -8), S( -30, -24), S( -31, -21), S( -43, -35),
    S( -56, -32), S( -33, -20), S( -25, -27), S( -28, -24), S( -29, -14), S( -34, -25), S( -34, -33), S( -47, -32),
    S( -34, -26), S( -32, -19), S( -40, -14), S( -26, -15), S( -25,  -9), S( -32,  -2), S( -25, -33), S( -39, -21),
    S( -28, -18), S( -16,   0), S( -23,   2), S( -12,  -2), S(  -8,  -1), S( -18,   0), S( -10,  -3), S( -30, -16),
    S(   0,   6), S(  10,  12), S(  17,  10), S(  15,   7), S(  19,  12), S(  11,  12), S(  11,  11), S(   7,  -5),
    S(  12,  19), S(  32,  10), S(  32,  20), S(  40,  24), S(  56,  24), S(  37,  18), S(  33,  15), S(   6,  13),
    S(  32,  17), S(  32,  22), S(  36,  27), S(  45,  33), S(  53,  31), S(  46,  34), S(  27,  21), S(  30,  18),
    S(  40,   7), S(  33,  11), S(  34,  23), S(  35,  30), S(  38,  32), S(  42,  17), S(  43,  16), S(  29,   8),
};

const PieceScore QueenPSQT[Square::NumSquares] =
{
    S( -15, -33), S( -29, -49), S( -34, -45), S( -29, -32), S( -31, -37), S( -35, -44), S( -24, -48), S( -16, -45),
    S( -17, -45), S( -21, -35), S( -17, -30), S( -20, -22), S( -17, -25), S( -18, -22), S( -13, -31), S( -18, -48),
    S( -19, -39), S( -10,  -9), S( -10,  -3), S( -18,  -6), S( -15, -10), S( -15, -13), S(  -9, -14), S( -12, -30),
    S( -24, -15), S(  -7,  -2), S(  -9,  18), S( -12,  16), S(  -6,  17), S(  -7,  14), S(  -2,   3), S( -13, -11),
    S(  -6,  -4), S(   1,  11), S(   8,  17), S(  23,  39), S(  19,  42), S(  18,  23), S(  -2,  11), S(   0,  -7),
    S(   6,  -5), S(  19,  19), S(   9,  39), S(  28,  49), S(  30,  38), S(  31,  36), S(  24,  24), S(   6,   3),
    S(  -1, -14), S( -12,  13), S(  34,  37), S(  33,  42), S(  25,  38), S(  33,  38), S(   3,   1), S(  12,   0),
    S(  29,   7), S(  16,  22), S(  35,  22), S(  31,  36), S(  23,  31), S(  24,  40), S(  24,   8), S(  27,  12),
};

const PieceScore KingPSQT[Square::NumSquares] =
{
    S(   4, -81), S(  21, -65), S(   5, -70), S( -35, -78), S( -12, -87), S(  -1, -67), S(  17, -60), S(   5, -71),
    S(  10, -55), S(   5, -51), S(   7, -52), S( -20, -48), S( -15, -50), S(   3, -49), S(  16, -45), S(  10, -50),
    S(   0, -52), S(  -5, -40), S( -10, -38), S(  13, -21), S( -15, -28), S(  -5, -27), S(   0, -35), S(   0, -48),
    S( -15, -33), S( -10, -10), S( -10,  -7), S( -10,   0), S( -15,  -4), S( -10,   0), S( -10,  -6), S( -15, -33),
    S( -25,  -4), S( -20,  32), S( -20,  18), S( -20,  24), S( -20,  24), S( -20,  28), S( -20,  31), S( -25,   6),
    S( -35,  26), S( -30,  68), S( -30,  59), S( -30,  61), S( -30,  61), S( -30,  63), S( -30,  70), S( -35,  23),
    S( -45,  26), S( -40,  83), S( -40,  71), S( -40,  64), S( -40,  61), S( -40,  72), S( -40,  64), S( -45,  24),
    S( -55, -84), S( -50,  64), S( -50,  96), S( -50,  74), S( -50,  76), S( -50,  86), S( -50,  70), S( -55, -73),
};

using PackedNeuralNetworkPtr = std::unique_ptr<nn::PackedNeuralNetwork>;
static std::unordered_map<MaterialKey, PackedNeuralNetworkPtr> g_neuralNetworks;
PackedNeuralNetworkPtr g_mainNeuralNetwork;

static void LoadNetworkForMaterialKey(const MaterialKey key)
{
    const std::string path = "../../data/networks/" + key.ToString() + ".pnn";

    PackedNeuralNetworkPtr network = std::make_unique<nn::PackedNeuralNetwork>();
    if (network->Load(path.c_str()))
    {
        std::cout << "info string Loaded neural network: " << path.c_str() << std::endl;
        g_neuralNetworks[key] = std::move(network);
    }
}

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

static std::string GetDefaultEvalFilePath()
{
    std::string path = GetExecutablePath();

    if (!path.empty())
    {
        path = path.substr(0, path.find_last_of("/\\")); // remove exec name
        path += "/";
        path += c_DefaultEvalFile;
    }

    return path;
}

bool TryLoadingDefaultEvalFile()
{
    // check if there's eval file in same directory as executable
    {
        std::string path = GetDefaultEvalFilePath();
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

void InitEvaluation()
{
    //LoadNetworkForMaterialKey({ 1,0,0,1,0, 0,0,0,1,0 });
    //LoadNetworkForMaterialKey({ 0,0,0,1,0, 1,0,0,1,0 });
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
        ASSERT(square < 64);
        outScoreMG += scores[square].mg;
        outScoreEG += scores[square].eg;
    });
}

INLINE static void EvalBlackPieceSquareTable(const Bitboard bitboard, const PieceScore* scores, int32_t& outScoreMG, int32_t& outScoreEG)
{
    bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        square = FlipRank(square);
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

ScoreType Evaluate(const Position& position, NodeInfo* nodeInfo, bool useNN)
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

    if (!g_neuralNetworks.empty())
    {
        if (position.GetSideToMove() == Color::White)
        {
            const auto iter = g_neuralNetworks.find(materialKey);
            if (iter != g_neuralNetworks.end())
            {
                const PackedNeuralNetworkPtr& network = iter->second;
                int32_t nnScore = NNEvaluator::Evaluate(*network, position, NetworkInputMapping::MaterialPacked_Symmetrical);

                // convert to centipawn range
                nnScore = (nnScore * c_nnOutputToCentiPawns + nn::OutputScale / 2) / nn::OutputScale;

                ASSERT(nnScore < TablebaseWinValue&& nnScore > -TablebaseWinValue);
                return (ScoreType)nnScore;
            }
        }
        else
        {
            const MaterialKey swappedMaterialKey = materialKey.SwappedColors();
            const auto iter = g_neuralNetworks.find(swappedMaterialKey);
            if (iter != g_neuralNetworks.end())
            {
                const PackedNeuralNetworkPtr& network = iter->second;
                int32_t nnScore = NNEvaluator::Evaluate(*network, position.SwappedColors(), NetworkInputMapping::MaterialPacked_Symmetrical);

                // convert to centipawn range
                nnScore = -((nnScore * c_nnOutputToCentiPawns + nn::OutputScale / 2) / nn::OutputScale);

                ASSERT(nnScore < TablebaseWinValue && nnScore > -TablebaseWinValue);
                return (ScoreType)nnScore;
            }
        }
    }

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

    // use neural network for balanced positions
    if (useNN && g_mainNeuralNetwork && std::abs(value) < c_nnTresholdMax)
    {
        int32_t nnValue = nodeInfo ?
            NNEvaluator::Evaluate(*g_mainNeuralNetwork, *nodeInfo, NetworkInputMapping::Full_Symmetrical) :
            NNEvaluator::Evaluate(*g_mainNeuralNetwork, position, NetworkInputMapping::Full_Symmetrical);

        // convert to centipawn range
        nnValue = (nnValue * c_nnOutputToCentiPawns + nn::OutputScale / 2) / nn::OutputScale;

        // NN output is side-to-move relative
        if (position.GetSideToMove() == Color::Black) nnValue = -nnValue;

        constexpr int32_t nnBlendRange = c_nnTresholdMax - c_nnTresholdMin;
        const int32_t nnFactor = std::max(0, std::abs(value) - c_nnTresholdMin);
        ASSERT(nnFactor <= nnBlendRange);
        value = (nnFactor * value + nnValue * (nnBlendRange - nnFactor)) / nnBlendRange;
    }

    // saturate eval value so it doesn't exceed KnownWinValue
    if (value > c_evalSaturationTreshold)
    {
        value = c_evalSaturationTreshold + (value - c_evalSaturationTreshold) / 4;
    }

    ASSERT(value > -KnownWinValue && value < KnownWinValue);

    // scale down when approaching 50-move draw
    value = value * (128 - std::max(0, (int32_t)position.GetHalfMoveCount() - 4)) / 128;

    ASSERT(value > -KnownWinValue && value < KnownWinValue);

    return (ScoreType)value;
}
