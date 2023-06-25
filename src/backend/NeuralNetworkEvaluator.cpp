#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

// enable validation of NN output (check if incremental updates work correctly)
//#define VALIDATE_NETWORK_OUTPUT

// (EXPERIMENTAL) enable accumulator cache
// - doesn't work with multithreading
// - looks like accessing cache lots of cache misses
// #define USE_ACCUMULATOR_CACHE

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

static void GetKingSideAndBucket(Square kingSquare, uint32_t& side, uint32_t& bucket)
{
    ASSERT(kingSquare.IsValid());

    if (kingSquare.File() >= 4)
    {
        kingSquare = kingSquare.FlippedFile();
        side = 1;
    }
    else
    {
        side = 0;
    }

    const uint32_t kingIndex = 4 * kingSquare.Rank() + kingSquare.File();
    ASSERT(kingIndex < 32);

    bucket = nn::KingBucketIndex[kingIndex];
    ASSERT(bucket < nn::NumKingBuckets);
}

template<bool IncludePieceFeatures>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective)
{
    uint32_t numFeatures = 0;

    const auto& whites = pos.GetSide(perspective);
    const auto& blacks = pos.GetSide(GetOppositeColor(perspective));

    Square whiteKingSquare = whites.GetKingSquare();
    Square blackKingSquare = blacks.GetKingSquare();

    uint32_t bitFlipMask = 0;

    if (whiteKingSquare.File() >= 4)
    {
        // flip file
        whiteKingSquare = whiteKingSquare.FlippedFile();
        blackKingSquare = blackKingSquare.FlippedFile();
        bitFlipMask = 0b000111;
    }

    if (perspective == Color::Black)
    {
        // flip rank
        whiteKingSquare = whiteKingSquare.FlippedRank();
        blackKingSquare = blackKingSquare.FlippedRank();
        bitFlipMask |= 0b111000;
    }

    const uint32_t kingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
    ASSERT(kingIndex < 32);

    const uint32_t kingBucket = nn::KingBucketIndex[kingIndex];
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
INLINE static uint32_t DirtyPieceToFeatureIndex(const Piece piece, const Color pieceColor, const Square square, const Position& pos)
{
    // this must match PositionToFeaturesVector !!!

    Square ourKingSquare = pos.GetSide(perspective).GetKingSquare();

    Square relativeSquare = square;
    {
        // flip the according to the perspective
        if constexpr (perspective == Color::Black)
        {
            relativeSquare = relativeSquare.FlippedRank();
            ourKingSquare = ourKingSquare.FlippedRank();
        }

        // flip the according to the king placement
        if (ourKingSquare.File() >= 4)
        {
            relativeSquare = relativeSquare.FlippedFile();
            ourKingSquare = ourKingSquare.FlippedFile();
        }
    }

    const uint32_t kingIndex = 4 * ourKingSquare.Rank() + ourKingSquare.File();
    ASSERT(kingIndex < 32);

    const uint32_t kingBucket = nn::KingBucketIndex[kingIndex];
    ASSERT(kingBucket < nn::NumKingBuckets);

    uint32_t index =
        kingBucket * 12 * 64 +
        ((uint32_t)piece - (uint32_t)Piece::Pawn) * 64 +
        relativeSquare.Index();

    if (pieceColor != perspective)
    {
        index += 6 * 64;
    }

    ASSERT(index < nn::NumKingBuckets * 12 * 64);

    return index;
}

uint32_t GetNetworkVariant(const Position& pos)
{
    const uint32_t numPieceCountBuckets = 8;
    const uint32_t pieceCountBucket = std::min(pos.GetNumPiecesExcludingKing() / 4u, numPieceCountBuckets - 1u);
    const uint32_t queenPresenceBucket = pos.Whites().queens || pos.Blacks().queens;
    return queenPresenceBucket * numPieceCountBuckets + pieceCountBucket;
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos)
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


#ifdef USE_ACCUMULATOR_CACHE

struct alignas(CACHELINE_SIZE) AccumulatorCacheEntry
{
    bool isValid = false;
    Color perspective = Color::White;
    uint64_t posHash;
    SidePosition posWhite;
    SidePosition posBlack;
    nn::Accumulator accumulator;
};

static constexpr uint32_t c_AccumulatorCacheSize = 8 * 1024;
static AccumulatorCacheEntry c_AccumulatorCache[c_AccumulatorCacheSize];

static bool ReadAccumulatorCache(const Position& pos, const Color perspective, nn::Accumulator& outAccumulator)
{
    const uint64_t posHash = pos.GetHash_NoSideToMove();
    const uint32_t index = posHash % c_AccumulatorCacheSize;
    const AccumulatorCacheEntry& entry = c_AccumulatorCache[index];

    // must have valid entry with matching piece placement and side to move
    if (!entry.isValid ||
        entry.perspective != perspective ||
        entry.posHash != posHash ||
        entry.posWhite != pos.Whites() ||
        entry.posBlack != pos.Blacks())
    {
        return false;
    }

    outAccumulator = entry.accumulator;
    return true;
}

static void WriteAccumulatorCache(const Position& pos, const Color perspective, const nn::Accumulator& accumulator)
{
    const uint64_t posHash = pos.GetHash_NoSideToMove();
    const uint32_t index = posHash % c_AccumulatorCacheSize;
    AccumulatorCacheEntry& entry = c_AccumulatorCache[index];

    // don't overwrite same entry
    if (entry.isValid &&
        entry.perspective == perspective &&
        entry.posWhite == pos.Whites() &&
        entry.posBlack == pos.Blacks())
    {
        return;
    }

    entry.isValid = true;
    entry.perspective = perspective;
    entry.posHash = posHash;
    entry.posWhite = pos.Whites();
    entry.posBlack = pos.Blacks();
    entry.accumulator = accumulator;
}

#endif // USE_ACCUMULATOR_CACHE

template<Color perspective>
INLINE static void UpdateAccumulator(const nn::PackedNeuralNetwork& network, const NodeInfo* prevAccumNode, const NodeInfo& node)
{
    ASSERT(prevAccumNode != &node);
    nn::Accumulator& accumulator = node.nnContext->accumulator[(uint32_t)perspective];
    ASSERT(node.nnContext->accumDirty[(uint32_t)perspective]);

    if (prevAccumNode)
    {
        ASSERT(prevAccumNode->nnContext);
        ASSERT(!prevAccumNode->nnContext->accumDirty[(uint32_t)perspective]);

        constexpr uint32_t maxChangedFeatures = 64;
        uint32_t numAddedFeatures = 0;
        uint32_t numRemovedFeatures = 0;
        uint16_t addedFeatures[maxChangedFeatures];
        uint16_t removedFeatures[maxChangedFeatures];

        // build a list of features to be updated
        for (const NodeInfo* nodePtr = &node; nodePtr != prevAccumNode; nodePtr = nodePtr->parentNode)
        {
            NNEvaluatorContext& nnContext = *(nodePtr->nnContext);

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
            accumulator = prevAccumNode->nnContext->accumulator[(uint32_t)perspective];
        }
        else
        {
            accumulator.Update(
                prevAccumNode->nnContext->accumulator[(uint32_t)perspective],
                network.GetAccumulatorWeights(),
                numAddedFeatures, addedFeatures,
                numRemovedFeatures, removedFeatures);
        }
    }
    else // refresh accumulator
    {
        const uint32_t maxFeatures = 64;
        uint16_t features[maxFeatures];
        const uint32_t numFeatures = PositionToFeaturesVector(node.position, features, perspective);
        ASSERT(numFeatures <= maxFeatures);

#ifdef NN_ACCUMULATOR_STATS
        s_NumAccumulatorRefreshes++;
#endif // NN_ACCUMULATOR_STATS

        accumulator.Refresh(
            network.GetAccumulatorWeights(), network.GetAccumulatorBiases(),
            numFeatures, features);
    }

    // mark accumulator as computed
    node.nnContext->accumDirty[(uint32_t)perspective] = false;

#ifdef USE_ACCUMULATOR_CACHE
    // cache accumulator values in PV nodes
    if (node.IsPV())
    {
        WriteAccumulatorCache(node.position, perspective, accumulator);
    }
#endif // USE_ACCUMULATOR_CACHE
}

template<Color perspective>
INLINE static void RefreshAccumulator(const nn::PackedNeuralNetwork& network, NodeInfo& node)
{
    const uint32_t refreshCost = node.position.GetNumPieces();

    uint32_t kingSide, kingBucket;

    if constexpr (perspective == Color::White)
        GetKingSideAndBucket(node.position.Whites().GetKingSquare(), kingSide, kingBucket);
    else
        GetKingSideAndBucket(node.position.Blacks().GetKingSquare().FlippedRank(), kingSide, kingBucket);

    // find closest parent node that has valid accumulator
    uint32_t updateCost = 0;
    const NodeInfo* prevAccumNode = nullptr;
    for (const NodeInfo* nodePtr = &node; nodePtr != nullptr; nodePtr = nodePtr->parentNode)
    {
        ASSERT(nodePtr->nnContext);

        updateCost += nodePtr->nnContext->numDirtyPieces;
        if (updateCost > refreshCost)
        {
            // update cost higher than refresh cost, incremental update not worth it
            break;
        }

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

        if (!nodePtr->nnContext->accumDirty[(uint32_t)perspective])
        {
            // found parent node with valid accumulator
            prevAccumNode = nodePtr;
            break;
        }

#ifdef USE_ACCUMULATOR_CACHE
        // check if accumulator was cached
        if (nodePtr->height < 8 &&
            ReadAccumulatorCache(nodePtr->position,
                perspective,
                nodePtr->nnContext->accumulator[(uint32_t)perspective]))
        {
            // found parent node with valid (cached) accumulator
            nodePtr->nnContext->accumDirty[(uint32_t)perspective] = false;
            prevAccumNode = nodePtr;
            break;
        }
#endif // USE_ACCUMULATOR_CACHE
    }

    if (prevAccumNode == &node)
    {
        // do nothing - accumulator is already up to date (was cached)
    }
    else if (node.parentNode && prevAccumNode &&
        node.parentNode != prevAccumNode &&
        node.parentNode->nnContext->accumDirty[(uint32_t)perspective])
    {
        // two-stage update:
        // if parent node has invalid accumulator, update it first
        // this way, sibling nodes can reuse parent's accumulator
        UpdateAccumulator<perspective>(network, prevAccumNode, *node.parentNode);
        UpdateAccumulator<perspective>(network, node.parentNode, node);
    }
    else
    {
        UpdateAccumulator<perspective>(network, prevAccumNode, node);
    }
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node)
{
    ASSERT(node.nnContext);

#ifndef VALIDATE_NETWORK_OUTPUT
    if (node.nnContext->nnScore != InvalidValue)
    {
        return node.nnContext->nnScore;
    }
#endif // VALIDATE_NETWORK_OUTPUT

    RefreshAccumulator<Color::White>(network, node);
    RefreshAccumulator<Color::Black>(network, node);

    const nn::Accumulator& ourAccumulator = node.nnContext->accumulator[(uint32_t)node.position.GetSideToMove()];
    const nn::Accumulator& theirAccumulator = node.nnContext->accumulator[(uint32_t)node.position.GetSideToMove() ^ 1u];
    const int32_t nnOutput = network.Run(ourAccumulator, theirAccumulator, GetNetworkVariant(node.position));

#ifdef VALIDATE_NETWORK_OUTPUT
    {
        const int32_t nnOutputReference = Evaluate(network, node.position);
        ASSERT(nnOutput == nnOutputReference);
    }
    if (node.nnContext->nnScore != InvalidValue)
    {
        ASSERT(node.nnContext->nnScore == nnOutput);
    }
#endif // VALIDATE_NETWORK_OUTPUT

    // cache NN output
    node.nnContext->nnScore = nnOutput;

    return nnOutput;
}
