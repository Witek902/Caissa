#include "Evaluate.hpp"
#include "Move.hpp"
#include "Material.hpp"
#include "Endgame.hpp"
#include "PackedNeuralNetwork.hpp"
#include "NeuralNetworkEvaluator.hpp"
#include "Pawns.hpp"
#include "Search.hpp"

#include <unordered_map>
#include <fstream>
#include <memory>

const char* c_DefaultEvalFile = "eval-AE45D47A.pnn";
const char* c_DefaultEndgameEvalFile = "endgame-9AE58CC2.pnn";

#define S(mg, eg) PieceScore{ mg, eg }

namespace {

static constexpr int32_t c_evalSaturationTreshold   = 10000;

static constexpr PieceScore c_tempoBonus            = S(4, 4);
static constexpr PieceScore c_castlingRightsBonus   = S(39, 9);
static constexpr PieceScore c_bishopPairBonus       = S(34, 49);

using PackedNeuralNetworkPtr = std::unique_ptr<nn::PackedNeuralNetwork>;
static PackedNeuralNetworkPtr g_mainNeuralNetwork;
static PackedNeuralNetworkPtr g_endgameNeuralNetwork;

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

    // piece square tables probing
    {
        const Square whiteKingSqFlipped = whiteKingSq.File() >= 4 ? whiteKingSq.FlippedFile() : whiteKingSq;
        const Square blackKingSqFlipped = blackKingSq.File() >= 4 ? blackKingSq.FlippedRank().FlippedFile() : blackKingSq.FlippedRank();

        const uint32_t whiteKingSquareIndex = 4 * whiteKingSqFlipped.Rank() + whiteKingSqFlipped.File();
        const uint32_t blackKingSquareIndex = 4 * blackKingSqFlipped.Rank() + blackKingSqFlipped.File();

        const uint32_t whiteSqMask = whiteKingSq.File() >= 4 ? 0b000111 : 0; // mirror horizontally depending on king position
        const uint32_t blackSqMask = blackKingSq.File() >= 4 ? 0b111111 : 0b111000; // mirror vertically + mirror horizontally depending on king position

        const KingsPerspectivePSQT& whitesPSQT = PSQT[whiteKingSquareIndex];
        const KingsPerspectivePSQT& blacksPSQT = PSQT[blackKingSquareIndex];

        pos.Whites().pawns.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[0][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[1][2 * (square ^ blackSqMask)]); });
        pos.Whites().knights.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[2][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[3][2 * (square ^ blackSqMask)]); });
        pos.Whites().bishops.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[4][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[5][2 * (square ^ blackSqMask)]); });
        pos.Whites().rooks.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[6][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[7][2 * (square ^ blackSqMask)]); });
        pos.Whites().queens.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[8][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[9][2 * (square ^ blackSqMask)]); });
        
        pos.Blacks().pawns.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[1][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[0][2 * (square ^ blackSqMask)]); });
        pos.Blacks().knights.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[3][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[2][2 * (square ^ blackSqMask)]); });
        pos.Blacks().bishops.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[5][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[4][2 * (square ^ blackSqMask)]); });
        pos.Blacks().rooks.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[7][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[6][2 * (square ^ blackSqMask)]); });
        pos.Blacks().queens.Iterate([&](const uint32_t square) INLINE_LAMBDA {
            value += PieceScore(&whitesPSQT[9][2 * (square ^ whiteSqMask)]);
            value -= PieceScore(&blacksPSQT[8][2 * (square ^ blackSqMask)]); });
    }

    value += c_queenValue * (whiteQueens - blackQueens);
    value += c_rookValue * (whiteRooks - blackRooks);
    value += c_bishopValue * (whiteBishops - blackBishops);
    value += c_knightValue * (whiteKnights - blackKnights);
    value += c_pawnValue * (whitePawns - blackPawns);

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
    if ((pos.Whites().bishops & Bitboard::LightSquares()) && (pos.Whites().bishops & Bitboard::DarkSquares())) value += c_bishopPairBonus;
    if ((pos.Blacks().bishops & Bitboard::LightSquares()) && (pos.Blacks().bishops & Bitboard::DarkSquares())) value -= c_bishopPairBonus;

    // castling rights
    value += c_castlingRightsBonus * ((int8_t)PopCount(pos.GetWhitesCastlingRights()) - (int8_t)PopCount(pos.GetBlacksCastlingRights()));

    // 0 - endgame, 64 - opening
    const int32_t gamePhase =
        1 * (whitePawns + blackPawns) +
        2 * (whiteKnights + blackKnights) +
        2 * (whiteBishops + blackBishops) +
        4 * (whiteRooks + blackRooks) +
        8 * (whiteQueens + blackQueens);

    // accumulate middle/end game scores
    int32_t finalValue = InterpolateScore(gamePhase, value);

    if (useNN)
    {
        const nn::PackedNeuralNetwork* networkToUse = nullptr;
        bool useIncrementalUpdate = false;
        if (numPieces >= 4 && numPieces <= 6)
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

            const int32_t nnBlendRange = c_nnTresholdMax - c_nnTresholdMin;
            const int32_t nnFactor = std::max(0, std::abs(finalValue) - c_nnTresholdMin);
            ASSERT(nnFactor <= nnBlendRange);
            finalValue = (nnFactor * finalValue + nnValue * (nnBlendRange - nnFactor)) / nnBlendRange;
        }
    }

    // saturate eval value so it doesn't exceed KnownWinValue
    if (finalValue > c_evalSaturationTreshold)
    {
        finalValue = c_evalSaturationTreshold + (finalValue - c_evalSaturationTreshold) / 8;
    }

    ASSERT(finalValue > -KnownWinValue && finalValue < KnownWinValue);

    return (ScoreType)finalValue;
}
