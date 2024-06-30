#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

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

void AccumulatorCache::Init(const nn::PackedNeuralNetwork* net)
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
    const auto& blacks = pos.GetSide(perspective ^ 1);

    Square kingSquare = whites.GetKingSquare();

    uint32_t bitFlipMask = 0;

    if (kingSquare.File() >= 4)
    {
        // flip file
        kingSquare = kingSquare.FlippedFile();
        bitFlipMask = 0b000111;
    }

    if (perspective == Black)
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
    if constexpr (perspective == Black)
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

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos)
{
    constexpr uint32_t maxFeatures = 64;

    uint16_t ourFeatures[maxFeatures];
    const uint32_t numOurFeatures = PositionToFeaturesVector(pos, ourFeatures, pos.GetSideToMove());
    ASSERT(numOurFeatures <= maxFeatures);

    uint16_t theirFeatures[maxFeatures];
    const uint32_t numTheirFeatures = PositionToFeaturesVector(pos, theirFeatures, pos.GetSideToMove() ^ 1);
    ASSERT(numTheirFeatures <= maxFeatures);

    return network.Run(ourFeatures, numOurFeatures, theirFeatures, numTheirFeatures, GetNetworkVariant(pos));
}

template<Color color>
INLINE static void UpdateAccumulator(const nn::PackedNeuralNetwork& network, const NodeInfo* prevAccumNode, NodeInfo& node)
{
    ASSERT(prevAccumNode);
    ASSERT(prevAccumNode != &node);
    ASSERT(!prevAccumNode->nnContext.accumDirty[color]);
    ASSERT(node.nnContext.accumDirty[color]);

    constexpr uint32_t maxChangedFeatures = 64;
    uint32_t numAddedFeatures = 0;
    uint32_t numRemovedFeatures = 0;
    uint16_t addedFeatures[maxChangedFeatures];
    uint16_t removedFeatures[maxChangedFeatures];

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
                const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex<color>(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.toSquare, node.position);
                addedFeatures[numAddedFeatures++] = featureIdx;
            }
            if (dirtyPiece.fromSquare.IsValid())
            {
                ASSERT(numRemovedFeatures < maxChangedFeatures);
                const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex<color>(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.fromSquare, node.position);
                removedFeatures[numRemovedFeatures++] = featureIdx;
            }
        }

        if (nodePtr->ply == 0)
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
        const uint32_t numReferenceFeatures = PositionToFeaturesVector(node.position, referenceFeatures, color);

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
        // accumulator is unchanged, just point to the previous accumulator
        node.accumulatorPtr[color] = prevAccumNode->accumulatorPtr[color];
    }
    else
    {
        node.accumulatorPtr[color] = &node.accumulatorData[color];
        node.accumulatorData[color].Update(
            *(prevAccumNode->accumulatorPtr[color]),
            network.GetAccumulatorWeights(),
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);
    }

    // mark accumulator as computed
    node.nnContext.accumDirty[color] = false;
}

template<Color color>
INLINE static void RefreshAccumulator(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache::KingBucket& cache)
{
    constexpr uint32_t maxChangedFeatures = 64;
    uint32_t numAddedFeatures = 0;
    uint32_t numRemovedFeatures = 0;
    uint16_t addedFeatures[maxChangedFeatures];
    uint16_t removedFeatures[maxChangedFeatures];

    for (Color c = 0; c < 2; ++c)
    {
        const Position& pos = node.position;
        const Bitboard* bitboards = &pos.GetSide(c).pawns;
        for (uint32_t p = 0; p < 6; ++p)
        {
            const Bitboard prev = cache.pieces[c][p];
            const Bitboard curr = bitboards[p];
            const Piece piece = (Piece)(p + (uint32_t)Piece::Pawn);

            // additions
            (curr & ~prev).Iterate([&](const Square sq) INLINE_LAMBDA
            {
                ASSERT(numAddedFeatures < maxChangedFeatures);
                addedFeatures[numAddedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<color>(piece, c, sq, pos);
            });

            // removals
            (prev & ~curr).Iterate([&](const Square sq) INLINE_LAMBDA
            {
                ASSERT(numRemovedFeatures < maxChangedFeatures);
                removedFeatures[numRemovedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<color>(piece, c, sq, pos);
            });

            cache.pieces[c][p] = curr;
        }
    }

    cache.accum.Update(
        cache.accum, network.GetAccumulatorWeights(),
        numAddedFeatures, addedFeatures,
        numRemovedFeatures, removedFeatures);

    node.accumulatorPtr[color] = &node.accumulatorData[color];
    node.accumulatorData[color] = cache.accum;

    // mark accumulator as computed
    node.nnContext.accumDirty[color] = false;
}

template<Color color>
INLINE static void EnsureAccumulatorUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& head, AccumulatorCache& cache)
{
    if (!head.nnContext.accumDirty[color])
    {
        // accumulator is already updated
        return;
    }

    uint32_t kingSide, kingBucket;
    if constexpr (color == White)
        GetKingSideAndBucket(head.position.Whites().GetKingSquare(), kingSide, kingBucket);
    else
        GetKingSideAndBucket(head.position.Blacks().GetKingSquare().FlippedRank(), kingSide, kingBucket);

    AccumulatorCache::KingBucket& kingBucketCache = cache.kingBuckets[color][kingBucket + kingSide * nn::NumKingBuckets];

    // root node must be always refreshed
    // TODO refresh at the beginning of the search
    if (head.ply == 0) [[unlikely]]
    {
        RefreshAccumulator<color>(network, head, kingBucketCache);
        return;
    }

    // find closest parent node that has valid accumulator
    NodeInfo* iter = &head;
    for (;;)
    {
        iter--;

        uint32_t newKingSide, newKingBucket;
        if constexpr (color == White)
            GetKingSideAndBucket(static_cast<const Position&>(iter->position).Whites().GetKingSquare(), newKingSide, newKingBucket);
        else
            GetKingSideAndBucket(static_cast<const Position&>(iter->position).Blacks().GetKingSquare().FlippedRank(), newKingSide, newKingBucket);

        if (newKingSide != kingSide || newKingBucket != kingBucket)
        {
            // king moved, accumulator needs to be refreshed
            RefreshAccumulator<color>(network, head, kingBucketCache);
            return;
        }

        if (!iter->nnContext.accumDirty[color])
        {
            // found parent node with valid accumulator - update all accumulators on the stack
            NodeInfo* lastUpdated = iter;
            while (lastUpdated != &head)
            {
                UpdateAccumulator<color>(network, lastUpdated, *(lastUpdated + 1));
                lastUpdated++;
            }
            return;
        }
    }
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache)
{
#ifndef VALIDATE_NETWORK_OUTPUT
    if (node.nnContext.nnScore != InvalidValue)
    {
        return node.nnContext.nnScore;
    }
#endif // VALIDATE_NETWORK_OUTPUT

    EnsureAccumulatorUpdated<White>(network, node, cache);
    EnsureAccumulatorUpdated<Black>(network, node, cache);

    const nn::Accumulator& ourAccumulator = *node.accumulatorPtr[(uint32_t)node.position.GetSideToMove()];
    const nn::Accumulator& theirAccumulator = *node.accumulatorPtr[(uint32_t)node.position.GetSideToMove() ^ 1u];
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

void NNEvaluator::EnsureAccumulatorsUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache)
{
    EnsureAccumulatorUpdated<White>(network, node, cache);
    EnsureAccumulatorUpdated<Black>(network, node, cache);
}
