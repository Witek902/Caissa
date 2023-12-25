#include "Evaluate.hpp"
#include "Move.hpp"
#include "Material.hpp"
#include "Endgame.hpp"
#include "PackedNeuralNetwork.hpp"
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
    const char* c_DefaultEvalFile = "eval-32.pnn";

#endif // defined(CAISSA_EVALFILE)


#define S(mg, eg) PieceScore{ mg, eg }

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

// enable validation of NN output (check if incremental updates work correctly)
//#define VALIDATE_NETWORK_OUTPUT

#ifdef NN_ACCUMULATOR_STATS

static std::atomic<uint64_t> s_NumAccumulatorUpdates = 0;
static std::atomic<uint64_t> s_NumAccumulatorRefreshes = 0;

void NNEvaluator::GetStats(uint64_t& outNumUpdates, uint64_t& outNumRefreshes)
{
    outNumUpdates = s_NumAccumulatorUpdates;
    outNumRefreshes = s_NumAccumulatorRefreshes;
}
void NNEvaluator::ResetStats()
{
    s_NumAccumulatorUpdates = 0;
    s_NumAccumulatorRefreshes = 0;
}

#endif // NN_ACCUMULATOR_STATS

void nn::AccumulatorCache::Init(const nn::PackedNeuralNetwork* net)
{
    if (currentNet != net)
    {
        for (uint32_t c = 0; c < 2; ++c)
        {
            for (uint32_t b = 0; b < 2 * nn::NumKingBuckets; ++b)
            {
                memcpy(kingBuckets[c][b].accum.values, net->GetAccumulatorBiases(), sizeof(nn::AccumulatorType) * nn::AccumulatorSize);
                memset(kingBuckets[c][b].pieces, 0, sizeof(kingBuckets[c][b].pieces));
            }
        }
        currentNet = net;
    }
}

template<bool IncludePieceFeatures>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective)
{
    uint32_t numFeatures = 0;

    const auto& whites = pos.GetSide(perspective);
    const auto& blacks = pos.GetSide(GetOppositeColor(perspective));

    Square kingSquare = whites.GetKingSquare();

    uint32_t bitFlipMask = 0;

    if (kingSquare.File() >= 4)
    {
        // flip file
        kingSquare = kingSquare.FlippedFile();
        bitFlipMask = 0b000111;
    }

    if (perspective == Color::Black)
    {
        // flip rank
        kingSquare = kingSquare.FlippedRank();
        bitFlipMask |= 0b111000;
    }

    const uint32_t kingBucket = nn::KingBucketIndex[kingSquare.Index()];
    ASSERT(kingBucket < nn::NumKingBuckets);

    uint32_t inputOffset = kingBucket * 12 * 64;

    const auto writeKingRelativePieceFeatures = [&](const Bitboard bitboard, const uint32_t bitFlipMask) INLINE_LAMBDA
    {
        bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            outFeatures[numFeatures++] = (uint16_t)(inputOffset + (square ^ bitFlipMask));
        });
        inputOffset += 64;
    };

    writeKingRelativePieceFeatures(whites.pawns, bitFlipMask);
    writeKingRelativePieceFeatures(whites.knights, bitFlipMask);
    writeKingRelativePieceFeatures(whites.bishops, bitFlipMask);
    writeKingRelativePieceFeatures(whites.rooks, bitFlipMask);
    writeKingRelativePieceFeatures(whites.queens, bitFlipMask);
    writeKingRelativePieceFeatures(whites.king, bitFlipMask);

    writeKingRelativePieceFeatures(blacks.pawns, bitFlipMask);
    writeKingRelativePieceFeatures(blacks.knights, bitFlipMask);
    writeKingRelativePieceFeatures(blacks.bishops, bitFlipMask);
    writeKingRelativePieceFeatures(blacks.rooks, bitFlipMask);
    writeKingRelativePieceFeatures(blacks.queens, bitFlipMask);
    writeKingRelativePieceFeatures(blacks.king, bitFlipMask);

    if constexpr (IncludePieceFeatures)
    {
        inputOffset = nn::NumKingBuckets * 12 * 64;

        const auto writePieceFeatures = [&](const Bitboard bitboard, const uint32_t bitFlipMask) INLINE_LAMBDA
        {
            bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
            {
                outFeatures[numFeatures++] = (uint16_t)(inputOffset + (square ^ bitFlipMask));
            });
            inputOffset += 64;
        };
        writePieceFeatures(whites.pawns, bitFlipMask);
        writePieceFeatures(whites.knights, bitFlipMask);
        writePieceFeatures(whites.bishops, bitFlipMask);
        writePieceFeatures(whites.rooks, bitFlipMask);
        writePieceFeatures(whites.queens, bitFlipMask);
        writePieceFeatures(whites.king, bitFlipMask);

        writePieceFeatures(blacks.pawns, bitFlipMask);
        writePieceFeatures(blacks.knights, bitFlipMask);
        writePieceFeatures(blacks.bishops, bitFlipMask);
        writePieceFeatures(blacks.rooks, bitFlipMask);
        writePieceFeatures(blacks.queens, bitFlipMask);
        writePieceFeatures(blacks.king, bitFlipMask);

        ASSERT(inputOffset == nn::NumKingBuckets * 12 * 64 + 12 * 64);
    }

    return numFeatures;
}

template uint32_t PositionToFeaturesVector<true>(const Position& pos, uint16_t* outFeatures, const Color perspective);
template uint32_t PositionToFeaturesVector<false>(const Position& pos, uint16_t* outFeatures, const Color perspective);

template<Color perspective>
INLINE static uint32_t DirtyPieceToFeatureIndex(const Piece piece, const Color pieceColor, Square square, const Position& pos)
{
    // this must match PositionToFeaturesVector !!!

    Square kingSquare = pos.GetSide(perspective).GetKingSquare();

    // flip the according to the perspective
    if constexpr (perspective == Color::Black)
    {
        square = square.FlippedRank();
        kingSquare = kingSquare.FlippedRank();
    }

    // flip the according to the king placement
    if (kingSquare.File() >= 4)
    {
        square = square.FlippedFile();
        kingSquare = kingSquare.FlippedFile();
    }

    const uint32_t kingBucket = nn::KingBucketIndex[kingSquare.Index()];
    ASSERT(kingBucket < nn::NumKingBuckets);

    uint32_t index =
        kingBucket * 12 * 64 +
        ((uint32_t)piece - (uint32_t)Piece::Pawn) * 64 +
        square.Index();

    if (pieceColor != perspective)
    {
        index += 6 * 64;
    }

    ASSERT(index < nn::NumKingBuckets * 12 * 64);

    return index;
}

static int32_t Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos)
{
    constexpr uint32_t maxFeatures = 64;

    uint16_t ourFeatures[maxFeatures];
    const uint32_t numOurFeatures = PositionToFeaturesVector(pos, ourFeatures, pos.GetSideToMove());
    ASSERT(numOurFeatures <= maxFeatures);

    uint16_t theirFeatures[maxFeatures];
    const uint32_t numTheirFeatures = PositionToFeaturesVector(pos, theirFeatures, GetOppositeColor(pos.GetSideToMove()));
    ASSERT(numTheirFeatures <= maxFeatures);

    return network.Run(ourFeatures, numOurFeatures, theirFeatures, numTheirFeatures, GetNetworkVariant(pos));
}

template<Color perspective>
INLINE static void UpdateAccumulator(const nn::PackedNeuralNetwork& network, const NodeInfo* prevAccumNode, NodeInfo& node, nn::AccumulatorCache::KingBucket& cache)
{
    ASSERT(prevAccumNode != &node);
    nn::Accumulator& accumulator = node.accumulator[(uint32_t)perspective];
    ASSERT(node.nnContext.accumDirty[(uint32_t)perspective]);

    constexpr uint32_t maxChangedFeatures = 64;
    uint32_t numAddedFeatures = 0;
    uint32_t numRemovedFeatures = 0;
    uint16_t addedFeatures[maxChangedFeatures];
    uint16_t removedFeatures[maxChangedFeatures];

    if (prevAccumNode)
    {
        ASSERT(!prevAccumNode->nnContext.accumDirty[(uint32_t)perspective]);

        // build a list of features to be updated
        for (const NodeInfo* nodePtr = &node; nodePtr != prevAccumNode; --nodePtr)
        {
            const NNEvaluatorContext& nnContext = nodePtr->nnContext;

            for (uint32_t i = 0; i < nnContext.numDirtyPieces; ++i)
            {
                const DirtyPiece& dirtyPiece = nnContext.dirtyPieces[i];

                if (dirtyPiece.toSquare.IsValid() && dirtyPiece.fromSquare.IsValid())
                {
                    // TODO use cached accumulator diff for piece move
                }

                if (dirtyPiece.toSquare.IsValid())
                {
                    ASSERT(numAddedFeatures < maxChangedFeatures);
                    const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex<perspective>(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.toSquare, node.position);
                    addedFeatures[numAddedFeatures++] = featureIdx;
                }
                if (dirtyPiece.fromSquare.IsValid())
                {
                    ASSERT(numRemovedFeatures < maxChangedFeatures);
                    const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex<perspective>(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.fromSquare, node.position);
                    removedFeatures[numRemovedFeatures++] = featureIdx;
                }
            }

            if (nodePtr->height == 0)
            {
                // reached end of stack
                break;
            }
        }

        // if same feature is present on both lists, it cancels out
        for (uint32_t i = 0; i < numAddedFeatures; ++i)
        {
            for (uint32_t j = 0; j < numRemovedFeatures; ++j)
            {
                if (addedFeatures[i] == removedFeatures[j])
                {
                    addedFeatures[i--] = addedFeatures[--numAddedFeatures];
                    removedFeatures[j--] = removedFeatures[--numRemovedFeatures];
                    break;
                }
            }
        }

#ifdef VALIDATE_NETWORK_OUTPUT
        {
            const uint32_t maxFeatures = 64;
            uint16_t referenceFeatures[maxFeatures];
            const uint32_t numReferenceFeatures = PositionToFeaturesVector(node.position, referenceFeatures, perspective);

            for (uint32_t i = 0; i < numAddedFeatures; ++i)
            {
                bool found = false;
                for (uint32_t j = 0; j < numReferenceFeatures; ++j)
                {
                    if (addedFeatures[i] == referenceFeatures[j]) found = true;
                }
                ASSERT(found);
            }
            for (uint32_t i = 0; i < numRemovedFeatures; ++i)
            {
                for (uint32_t j = 0; j < numReferenceFeatures; ++j)
                {
                    ASSERT(removedFeatures[i] != referenceFeatures[j]);
                }
            }
        }
#endif // VALIDATE_NETWORK_OUTPUT

#ifdef NN_ACCUMULATOR_STATS
        s_NumAccumulatorUpdates++;
#endif // NN_ACCUMULATOR_STATS

        if (numAddedFeatures == 0 && numRemovedFeatures == 0)
        {
            accumulator = prevAccumNode->accumulator[(uint32_t)perspective];
        }
        else
        {
            accumulator.Update(
                prevAccumNode->accumulator[(uint32_t)perspective],
                network.GetAccumulatorWeights(),
                numAddedFeatures, addedFeatures,
                numRemovedFeatures, removedFeatures);
        }
    }
    else // refresh accumulator
    {
        for (uint32_t c = 0; c < 2; ++c)
        {
            const Position& pos = node.position;
            const Bitboard* bitboards = &pos.GetSide((Color)c).pawns;
            for (uint32_t p = 0; p < 6; ++p)
            {
                const Bitboard prev = cache.pieces[c][p];
                const Bitboard curr = bitboards[p];
                const Piece piece = (Piece)(p + (uint32_t)Piece::Pawn);

                // additions
                (curr & ~prev).Iterate([&](const Square sq) INLINE_LAMBDA
                    {
                        ASSERT(numAddedFeatures < maxChangedFeatures);
                        addedFeatures[numAddedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<perspective>(piece, (Color)c, sq, pos);
                    });

                // removals
                (prev & ~curr).Iterate([&](const Square sq) INLINE_LAMBDA
                    {
                        ASSERT(numRemovedFeatures < maxChangedFeatures);
                        removedFeatures[numRemovedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<perspective>(piece, (Color)c, sq, pos);
                    });

                cache.pieces[c][p] = curr;
            }
        }

        cache.accum.Update(
            cache.accum,
            network.GetAccumulatorWeights(),
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);

        accumulator = cache.accum;
    }

    // mark accumulator as computed
    node.nnContext.accumDirty[(uint32_t)perspective] = false;
}

template<Color perspective>
INLINE static void RefreshAccumulator(const nn::PackedNeuralNetwork& network, NodeInfo& node, nn::AccumulatorCache& cache)
{
    const Position& pos = node.position;

    uint32_t kingSide, kingBucket;
    if constexpr (perspective == Color::White)
        GetKingSideAndBucket(pos.Whites().GetKingSquare(), kingSide, kingBucket);
    else
        GetKingSideAndBucket(pos.Blacks().GetKingSquare().FlippedRank(), kingSide, kingBucket);

    nn::AccumulatorCache::KingBucket& kingBucketCache = cache.kingBuckets[(uint32_t)perspective][kingBucket + kingSide * nn::NumKingBuckets];

    // find closest parent node that has valid accumulator
    const NodeInfo* prevAccumNode = nullptr;
    for (const NodeInfo* nodePtr = &node; ; --nodePtr)
    {
        uint32_t newKingSide, newKingBucket;
        if constexpr (perspective == Color::White)
            GetKingSideAndBucket(static_cast<const Position&>(nodePtr->position).Whites().GetKingSquare(), newKingSide, newKingBucket);
        else
            GetKingSideAndBucket(static_cast<const Position&>(nodePtr->position).Blacks().GetKingSquare().FlippedRank(), newKingSide, newKingBucket);

        if (newKingSide != kingSide || newKingBucket != kingBucket)
        {
            // king moved, accumulator needs to be refreshed
            break;
        }

        if (!nodePtr->nnContext.accumDirty[(uint32_t)perspective])
        {
            // found parent node with valid accumulator
            prevAccumNode = nodePtr;
            break;
        }

        if (nodePtr->height == 0)
        {
            // reached end of stack
            break;
        }
    }

    NodeInfo* parentInfo = &node - 1;

    if (prevAccumNode == &node)
    {
        // do nothing - accumulator is already up to date (was cached)
    }
    else if (node.height > 0 && prevAccumNode &&
        parentInfo != prevAccumNode &&
        parentInfo->nnContext.accumDirty[(uint32_t)perspective])
    {
        // two-stage update:
        // if parent node has invalid accumulator, update it first
        // this way, sibling nodes can reuse parent's accumulator
        UpdateAccumulator<perspective>(network, prevAccumNode, *parentInfo, kingBucketCache);
        UpdateAccumulator<perspective>(network, parentInfo, node, kingBucketCache);
    }
    else
    {
        UpdateAccumulator<perspective>(network, prevAccumNode, node, kingBucketCache);
    }
}

INLINE static int32_t Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node, nn::AccumulatorCache& cache)
{
#ifndef VALIDATE_NETWORK_OUTPUT
    if (node.nnContext.nnScore != InvalidValue)
    {
        return node.nnContext.nnScore;
    }
#endif // VALIDATE_NETWORK_OUTPUT

    RefreshAccumulator<Color::White>(network, node, cache);
    RefreshAccumulator<Color::Black>(network, node, cache);

    const nn::Accumulator& ourAccumulator = node.accumulator[(uint32_t)node.position.GetSideToMove()];
    const nn::Accumulator& theirAccumulator = node.accumulator[(uint32_t)node.position.GetSideToMove() ^ 1u];
    const int32_t nnOutput = network.Run(ourAccumulator, theirAccumulator, GetNetworkVariant(node.position));

#ifdef VALIDATE_NETWORK_OUTPUT
    {
        const int32_t nnOutputReference = Evaluate(network, node.position);
        ASSERT(nnOutput == nnOutputReference);
    }
    if (node.nnContext.nnScore != InvalidValue)
    {
        ASSERT(node.nnContext.nnScore == nnOutput);
    }
#endif // VALIDATE_NETWORK_OUTPUT

    // cache NN output
    node.nnContext.nnScore = nnOutput;

    return nnOutput;
}

ScoreType Evaluate(const Position& pos)
{
    NodeInfo dummyNode = { pos };

    nn::AccumulatorCache dummyCache;
    if (g_mainNeuralNetwork)
    {
        dummyCache.Init(g_mainNeuralNetwork.get());
    }

    return Evaluate(dummyNode, dummyCache);
}

ScoreType Evaluate(NodeInfo& node, nn::AccumulatorCache& cache)
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
        int32_t nnValue = Evaluate(*g_mainNeuralNetwork, node, cache);

        // convert to centipawn range
        constexpr int32_t nnOutputDiv = nn::OutputScale * nn::WeightScale / c_nnOutputToCentiPawns;
        nnValue = DivRoundNearest(nnValue, nnOutputDiv);

        // NN output is side-to-move relative
        if (pos.GetSideToMove() == Color::Black) nnValue = -nnValue;

        finalValue = nnValue;
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

void EnsureAccumulatorUpdated(NodeInfo& node, nn::AccumulatorCache& cache)
{
    if (g_mainNeuralNetwork)
    {
        RefreshAccumulator<Color::White>(*g_mainNeuralNetwork, node, cache);
        RefreshAccumulator<Color::Black>(*g_mainNeuralNetwork, node, cache);
    }
}