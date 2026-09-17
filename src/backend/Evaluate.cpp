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
    const char* c_DefaultEvalFile = "eval-ml-8-152B-permuted.pnn";

#endif // defined(CAISSA_EVALFILE)

namespace {

static constexpr int32_t c_evalSaturationTreshold   = 8000;
static constexpr ScoreType c_castlingRightsBonus = 5;

} // namespace

const nn::PackedNeuralNetwork* g_mainNeuralNetwork = nullptr;
static bool g_usingEmbeddedNeuralNetwork = false;

#ifdef ENABLE_NET_L3_TUNING

// Last-layer weights and biases of output buckets 1..NumVariants-1 exposed as tunable parameters.
// They are copied into the (writable) network at search start, so the evaluation code reads the
// network exactly as in a regular build. Bucket 0 is skipped: it covers only up to 3 non-king pieces.
static constexpr uint32_t c_firstTunableVariant = 1;
static int32_t s_l3WeightShadow[nn::NumVariants][nn::L2Size];
static int32_t s_l3BiasShadow[nn::NumVariants];
static nn::PackedNeuralNetwork* g_tunableNeuralNetwork = nullptr;
static bool s_netTunablesRegistered = false;

static void RegisterNeuralNetTunables(nn::PackedNeuralNetwork* network)
{
    g_tunableNeuralNetwork = network;

    for (uint32_t v = c_firstTunableVariant; v < nn::NumVariants; ++v)
    {
        const nn::PackedNeuralNetwork::OutputSubnetVariant& subnet = network->outputSubnetVariants[v];
        for (uint32_t i = 0; i < nn::L2Size; ++i)
            s_l3WeightShadow[v][i] = subnet.l3Weights[i];
        s_l3BiasShadow[v] = subnet.l3Bias;
    }

    if (s_netTunablesRegistered)
        return;
    s_netTunablesRegistered = true;

    // weights: 1 unit is 1/2064 of a float weight (median |w| ~1000); biases: ~1500 units per centipawn
    constexpr int32_t c_weightRange = 1000;
    constexpr int32_t c_biasRange = 30000;
    constexpr int32_t c_weightMin = std::numeric_limits<nn::LastLayerWeightType>::min();
    constexpr int32_t c_weightMax = std::numeric_limits<nn::LastLayerWeightType>::max();

    for (uint32_t v = c_firstTunableVariant; v < nn::NumVariants; ++v)
    {
        const std::string prefix = "NN_L3_B" + std::to_string(v) + "_";
        for (uint32_t i = 0; i < nn::L2Size; ++i)
        {
            int32_t& value = s_l3WeightShadow[v][i];
            const std::string name = prefix + "W" + (i < 10 ? "0" : "") + std::to_string(i);
            g_TunableParameters.emplace_back(name, value, std::max(c_weightMin, value - c_weightRange), std::min(c_weightMax, value + c_weightRange));
        }
        int32_t& bias = s_l3BiasShadow[v];
        g_TunableParameters.emplace_back(prefix + "Bias", bias, bias - c_biasRange, bias + c_biasRange);
    }
}

void ApplyNeuralNetTunables()
{
    if (!g_tunableNeuralNetwork)
        return;

    for (uint32_t v = c_firstTunableVariant; v < nn::NumVariants; ++v)
    {
        nn::PackedNeuralNetwork::OutputSubnetVariant& subnet = g_tunableNeuralNetwork->outputSubnetVariants[v];
        for (uint32_t i = 0; i < nn::L2Size; ++i)
            subnet.l3Weights[i] = static_cast<nn::LastLayerWeightType>(s_l3WeightShadow[v][i]);
        subnet.l3Bias = s_l3BiasShadow[v];
    }
}

#endif // ENABLE_NET_L3_TUNING

bool LoadMainNeuralNetwork(const char* path)
{
    if (!g_usingEmbeddedNeuralNetwork)
    {
        // release previous network
        delete g_mainNeuralNetwork;
        g_mainNeuralNetwork = nullptr;
#ifdef ENABLE_NET_L3_TUNING
        g_tunableNeuralNetwork = nullptr;
#endif // ENABLE_NET_L3_TUNING
    }

    if (path == nullptr || strcmp(path, "") == 0 || strcmp(path, "<empty>") == 0)
    {
#if defined(CAISSA_EVALFILE)
        const auto* embeddedNetwork = reinterpret_cast<const nn::PackedNeuralNetwork*>(EmbedData);
        // the embedded data is read-only, so an older format cannot be converted in place
        if (embeddedNetwork->header.version != nn::CurrentVersion)
        {
            std::cout << "info string Embedded neural network has unsupported version " << embeddedNetwork->header.version << std::endl;
            return false;
        }
#ifdef ENABLE_NET_L3_TUNING
        // writable copy, so the tunable last-layer values can be applied
        auto* networkCopy = new nn::PackedNeuralNetwork();
        memcpy(networkCopy, embeddedNetwork, sizeof(nn::PackedNeuralNetwork));
        g_mainNeuralNetwork = networkCopy;
        g_usingEmbeddedNeuralNetwork = false;
        RegisterNeuralNetTunables(networkCopy);
#else
        g_mainNeuralNetwork = embeddedNetwork;
        g_usingEmbeddedNeuralNetwork = true;
#endif // ENABLE_NET_L3_TUNING
        std::cout << "info string Using embedded neural network" << std::endl;
        return true;
#else
        std::cout << "info string disabled neural network evaluation" << std::endl;
        g_mainNeuralNetwork = nullptr;
        g_usingEmbeddedNeuralNetwork = false;
        return true;
#endif // defined(CAISSA_EVALFILE)
    }

    auto* newNetwork = new nn::PackedNeuralNetwork();
    if (newNetwork->LoadFromFile(path))
    {
        g_mainNeuralNetwork = newNetwork;
        g_usingEmbeddedNeuralNetwork = false;
#ifdef ENABLE_NET_L3_TUNING
        RegisterNeuralNetTunables(newNetwork);
#endif // ENABLE_NET_L3_TUNING
        std::cout << "info string Loaded neural network: " << path << std::endl;
        return true;
    }
    else
    {
        delete newNetwork;
    }

    // TODO use embedded net?
    
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
        dummyCache.Init(g_mainNeuralNetwork);
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

    // apply castling rights bonus
    {
        ScoreType bonus = 0;
        if (pos.Whites().GetKingSquare() != Square_e1) bonus += c_castlingRightsBonus * (ScoreType)PopCount(pos.GetWhitesCastlingRights());
        if (pos.Blacks().GetKingSquare() != Square_e8) bonus -= c_castlingRightsBonus * (ScoreType)PopCount(pos.GetBlacksCastlingRights());
        value += pos.GetSideToMove() == White ? bonus : -bonus;
    }

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