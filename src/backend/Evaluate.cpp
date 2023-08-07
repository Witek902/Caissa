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

const char* c_DefaultEvalFile = "eval-19.pnn";
#ifdef USE_ENDGAME_NEURAL_NETWORK
const char* c_DefaultEndgameEvalFile = "endgame-2.pnn";
#endif // USE_ENDGAME_NEURAL_NETWORK

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

using PackedNeuralNetworkPtr = std::unique_ptr<nn::PackedNeuralNetwork>;
static PackedNeuralNetworkPtr g_mainNeuralNetwork;
#ifdef USE_ENDGAME_NEURAL_NETWORK
static PackedNeuralNetworkPtr g_endgameNeuralNetwork;
#endif // USE_ENDGAME_NEURAL_NETWORK

} // namespace


bool LoadMainNeuralNetwork(const char* path)
{
    if (strcmp(path, "") == 0)
    {
        std::cout << "info string disabled neural network evaluation" << std::endl;
        g_mainNeuralNetwork.reset();
        return true;
    }

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

#ifdef USE_ENDGAME_NEURAL_NETWORK
bool LoadEndgameNeuralNetwork(const char* path)
{
    if (strcmp(path, "") == 0)
    {
        std::cout << "info string disabled neural network endgame evaluation" << std::endl;
        g_endgameNeuralNetwork.reset();
        return true;
    }

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
#endif // USE_ENDGAME_NEURAL_NETWORK

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

#ifdef USE_ENDGAME_NEURAL_NETWORK
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
#endif // USE_ENDGAME_NEURAL_NETWORK

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

#ifdef EVAL_USE_PSQT

const TPieceScore<int32_t> ComputePSQT(const Position& pos)
{
    const Square whiteKingSq = pos.Whites().GetKingSquare();
    const Square blackKingSq = pos.Blacks().GetKingSquare();

    const Square whiteKingSqFlipped = whiteKingSq.File() >= 4 ? whiteKingSq.FlippedFile() : whiteKingSq;
    const Square blackKingSqFlipped = blackKingSq.File() >= 4 ? blackKingSq.FlippedRank().FlippedFile() : blackKingSq.FlippedRank();

    const uint32_t whiteKingSquareIndex = 4 * whiteKingSqFlipped.Rank() + whiteKingSqFlipped.File();
    const uint32_t blackKingSquareIndex = 4 * blackKingSqFlipped.Rank() + blackKingSqFlipped.File();

    const uint32_t whiteSqMask = whiteKingSq.File() >= 4 ? 0b000111 : 0; // mirror horizontally depending on king position
    const uint32_t blackSqMask = blackKingSq.File() >= 4 ? 0b111111 : 0b111000; // mirror vertically + mirror horizontally depending on king position

    const KingsPerspectivePSQT& whitesPSQT = PSQT[whiteKingSquareIndex];
    const KingsPerspectivePSQT& blacksPSQT = PSQT[blackKingSquareIndex];

    TPieceScore<int32_t> value = { 0, 0 };

    pos.Whites().pawns.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[0][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[1][2 * (square ^ blackSqMask)]); });
    pos.Whites().knights.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[2][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[3][2 * (square ^ blackSqMask)]); });
    pos.Whites().bishops.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[4][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[5][2 * (square ^ blackSqMask)]); });
    pos.Whites().rooks.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[6][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[7][2 * (square ^ blackSqMask)]); });
    pos.Whites().queens.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[8][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[9][2 * (square ^ blackSqMask)]); });

    pos.Blacks().pawns.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[1][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[0][2 * (square ^ blackSqMask)]); });
    pos.Blacks().knights.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[3][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[2][2 * (square ^ blackSqMask)]); });
    pos.Blacks().bishops.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[5][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[4][2 * (square ^ blackSqMask)]); });
    pos.Blacks().rooks.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[7][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[6][2 * (square ^ blackSqMask)]); });
    pos.Blacks().queens.Iterate([&](const uint32_t square) INLINE_LAMBDA{
        value += PieceScore(&whitesPSQT[9][2 * (square ^ whiteSqMask)]);
        value -= PieceScore(&blacksPSQT[8][2 * (square ^ blackSqMask)]); });

    return value;
}

void ComputeIncrementalPSQT(TPieceScore<int32_t>& score, const Position& pos, const DirtyPiece* dirtyPieces, uint32_t numDirtyPieces)
{
    const Square whiteKingSq = pos.Whites().GetKingSquare();
    const Square blackKingSq = pos.Blacks().GetKingSquare();

    const Square whiteKingSqFlipped = whiteKingSq.File() >= 4 ? whiteKingSq.FlippedFile() : whiteKingSq;
    const Square blackKingSqFlipped = blackKingSq.File() >= 4 ? blackKingSq.FlippedRank().FlippedFile() : blackKingSq.FlippedRank();

    const uint32_t whiteKingSquareIndex = 4 * whiteKingSqFlipped.Rank() + whiteKingSqFlipped.File();
    const uint32_t blackKingSquareIndex = 4 * blackKingSqFlipped.Rank() + blackKingSqFlipped.File();

    const uint32_t whiteSqMask = whiteKingSq.File() >= 4 ? 0b000111 : 0; // mirror horizontally depending on king position
    const uint32_t blackSqMask = blackKingSq.File() >= 4 ? 0b111111 : 0b111000; // mirror vertically + mirror horizontally depending on king position

    const KingsPerspectivePSQT& whitesPSQT = PSQT[whiteKingSquareIndex];
    const KingsPerspectivePSQT& blacksPSQT = PSQT[blackKingSquareIndex];

    for (uint32_t i = 0; i < numDirtyPieces; ++i)
    {
        const DirtyPiece& dirtyPiece = dirtyPieces[i];

        // any king movement invalidates PSQT as the values are king-relative
        // it should be checked before
        ASSERT(dirtyPiece.piece != Piece::King);

        const uint32_t pieceIndex = (uint32_t)dirtyPiece.piece - (uint32_t)Piece::Pawn;
        ASSERT(pieceIndex < 5);

        const uint32_t whitePieceIndex = 2u * pieceIndex + (0 ^ (uint32_t)dirtyPiece.color);
        const uint32_t blackPieceIndex = 2u * pieceIndex + (1 ^ (uint32_t)dirtyPiece.color);

        if (dirtyPiece.toSquare.IsValid()) // add piece
        {
            score += PieceScore(&whitesPSQT[whitePieceIndex][2u * (dirtyPiece.toSquare.Index() ^ whiteSqMask)]);
            score -= PieceScore(&blacksPSQT[blackPieceIndex][2u * (dirtyPiece.toSquare.Index() ^ blackSqMask)]);
        }
        if (dirtyPiece.fromSquare.IsValid()) // remove piece
        {
            score -= PieceScore(&whitesPSQT[whitePieceIndex][2u * (dirtyPiece.fromSquare.Index() ^ whiteSqMask)]);
            score += PieceScore(&blacksPSQT[blackPieceIndex][2u * (dirtyPiece.fromSquare.Index() ^ blackSqMask)]);
        }
    }

    // compare with non-incremental version
    ASSERT(score == ComputePSQT(pos));
}

#endif // EVAL_USE_PSQT

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

ScoreType Evaluate(const Position& pos, NodeInfo* nodeInfo, bool useNN)
{
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
            (whitePawns   + blackPawns) +
        2 * (whiteKnights + blackKnights) +
        2 * (whiteBishops + blackBishops) +
        4 * (whiteRooks   + blackRooks) +
        8 * (whiteQueens  + blackQueens));

    int32_t finalValue = 0;

    if (useNN && g_mainNeuralNetwork)
    {
        int32_t nnValue = nodeInfo ?
            NNEvaluator::Evaluate(*g_mainNeuralNetwork, *nodeInfo) :
            NNEvaluator::Evaluate(*g_mainNeuralNetwork, pos);

        // convert to centipawn range
        nnValue = (nnValue * c_nnOutputToCentiPawns + nn::OutputScale / 2) / nn::OutputScale;

        // NN output is side-to-move relative
        if (pos.GetSideToMove() == Color::Black) nnValue = -nnValue;

        finalValue = nnValue;
    }
    else // fallback to simple evaluation
    {
        TPieceScore<int32_t> value;

#ifdef EVAL_USE_PSQT
        if (nodeInfo)
        {
            value = nodeInfo->psqtScore;
            ASSERT(value.mg != INT32_MIN && value.eg != INT32_MIN);
        }
        else
        {
            value = ComputePSQT(pos);
        }
#else // !EVAL_USE_PSQT
        value = {};
#endif // EVAL_USE_PSQT

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
    finalValue = finalValue * (192 + gamePhase) / 256;

    // saturate eval value so it doesn't exceed KnownWinValue
    if (finalValue > c_evalSaturationTreshold)
        finalValue = c_evalSaturationTreshold + (finalValue - c_evalSaturationTreshold) / 8;
    else if (finalValue < -c_evalSaturationTreshold)
        finalValue = -c_evalSaturationTreshold + (finalValue + c_evalSaturationTreshold) / 8;

    ASSERT(finalValue > -KnownWinValue && finalValue < KnownWinValue);

    return (ScoreType)(finalValue * scale / c_endgameScaleMax);
}

void EnsureAccumulatorUpdated(NodeInfo& node)
{
    if (g_mainNeuralNetwork)
    {
        NNEvaluator::EnsureAccumulatorUpdated(*g_mainNeuralNetwork, node);
    }
}