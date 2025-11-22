#include "Evaluate.hpp"
#include "Endgame.hpp"
#include "Search.hpp"

#include <fstream>

#if defined(CAISSA_EVALFILE)

    // embed eval file into executable
    #define INCBIN_PREFIX
    #define INCBIN_STYLE INCBIN_STYLE_CAMEL
    #include "incbin.h"
        INCBIN(Embed, CAISSA_EVALFILE);

    const char* c_DefaultEvalFile = "<empty>";

#else // !defined(CAISSA_EVALFILE)

    // use eval file
    const char* c_DefaultEvalFile = "eval-67.pnn";

#endif // defined(CAISSA_EVALFILE)

namespace {

static constexpr int32_t c_evalSaturationTreshold   = 8000;
static constexpr ScoreType c_castlingRightsBonus = 5;

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

    const int32_t queens = (pos.Whites().queens | pos.Blacks().queens).Count();
    const int32_t rooks = (pos.Whites().rooks | pos.Blacks().rooks).Count();
    const int32_t bishopsAndKnights = (pos.Whites().bishops | pos.Blacks().bishops | pos.Whites().knights | pos.Blacks().knights).Count();
    const int32_t pawns = (pos.Whites().pawns | pos.Blacks().pawns).Count();

    const int32_t pieceCount = queens + rooks + bishopsAndKnights + pawns;

    // check endgame evaluation first
    if (pieceCount <= 6) [[unlikely]]
    {
        int32_t endgameScore;
        if (EvaluateEndgame(pos, endgameScore))
        {
            ASSERT(endgameScore < TablebaseWinValue && endgameScore > -TablebaseWinValue);
            if (pos.GetSideToMove() == Black) endgameScore = -endgameScore;
            return (ScoreType)endgameScore;
        }
    }

    int32_t value = NNEvaluator::Evaluate(*g_mainNeuralNetwork, node, cache);

    // convert to centipawn range
    value /= nn::OutputScale * nn::WeightScale / c_nnOutputToCentiPawns;

    // apply scaling based on game phase (0 - endgame, 24 - opening)
    const int32_t gamePhase = bishopsAndKnights + 2 * rooks + 4 * queens;
    value = value * (52 + gamePhase) / 64;

    // saturate eval value so it doesn't exceed KnownWinValue
    if (value > c_evalSaturationTreshold)
        value = c_evalSaturationTreshold + (value - c_evalSaturationTreshold) / 8;
    else if (value < -c_evalSaturationTreshold)
        value = -c_evalSaturationTreshold + (value + c_evalSaturationTreshold) / 8;

    ASSERT(value > -KnownWinValue && value < KnownWinValue);

    return (ScoreType)value;
}

void EnsureAccumulatorUpdated(NodeInfo& node, AccumulatorCache& cache)
{
    NNEvaluator::EnsureAccumulatorUpdated(*g_mainNeuralNetwork, node, cache);
}