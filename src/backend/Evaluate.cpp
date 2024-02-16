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
    const char* c_DefaultEvalFile = "eval-33.pnn";

#endif // defined(CAISSA_EVALFILE)

namespace {

static constexpr int32_t c_evalSaturationTreshold   = 8000;

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
    if (whitePieceCount + blackPieceCount <= 6 || blackPieceCount == 0 || whitePieceCount == 0) [[unlikely]]
    {
        int32_t endgameScore;
        if (EvaluateEndgame(pos, endgameScore, scale))
        {
            ASSERT(endgameScore < TablebaseWinValue && endgameScore > -TablebaseWinValue);
            if (pos.GetSideToMove() == Black) endgameScore = -endgameScore;
            return (ScoreType)endgameScore;
        }
    }

    // 0 - endgame, 64 - opening
    const int32_t gamePhase = std::min(64,
        3 * (whiteKnights + blackKnights + whiteBishops + blackBishops) +
        5 * (whiteRooks   + blackRooks) +
        10 * (whiteQueens  + blackQueens));

    const auto materialValue =
        c_pawnValue * (whitePawns - blackPawns) +
        c_knightValue * (whiteKnights - blackKnights) +
        c_bishopValue * (whiteBishops - blackBishops) +
        c_rookValue * (whiteRooks - blackRooks) +
        c_queenValue * (whiteQueens - blackQueens);
    const int32_t simpleEval = (materialValue.mg * gamePhase + materialValue.eg * (64 - gamePhase)) / 64;

    const int32_t nnValue = NNEvaluator::Evaluate(*g_mainNeuralNetwork, node, cache);

    // convert to centipawn range
    constexpr int32_t nnOutputDiv = nn::OutputScale * nn::WeightScale / c_nnOutputToCentiPawns;
    int32_t finalValue = DivRoundNearest(nnValue, nnOutputDiv);

    // scale down when difference between simple eval and NN eval is large
    finalValue -= finalValue * std::min(2048, std::abs(simpleEval - finalValue)) / 8192;

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