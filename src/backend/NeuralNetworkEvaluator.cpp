#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

// enable validation of NN output (check if incremental updates work correctly)
// #define VALIDATE_NETWORK_OUTPUT

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


INLINE static uint32_t DirtyPieceToFeatureIndex(const Piece piece, const Color pieceColor, const Square square, const Position& pos, const Color perspective)
{
    // this must match Position::ToSparseFeaturesVector !!!

    Square relativeSquare = square;
    {
        // flip the according to the perspective
        if (perspective == Color::Black) relativeSquare = relativeSquare.FlippedRank();

        // flip the according to the king placement
        const Bitboard leftFilesMask = 0x0F0F0F0F0F0F0F0Full;
        if ((pos.GetCurrentSide().king & leftFilesMask) == 0) relativeSquare = relativeSquare.FlippedFile();
    }

    uint32_t index;
    if (piece == Piece::Pawn)
    {
        ASSERT(relativeSquare.Rank() > 0 && relativeSquare.Rank() < 7);
        index = relativeSquare.Index() - 8;
    }
    else if (piece == Piece::King && pieceColor == perspective)
    {
        // king of the side-to-move is a special case - it can be only present on A-D files
        ASSERT(relativeSquare.File() < 4);
        const uint32_t kingSquareIndex = 4 * relativeSquare.Rank() + relativeSquare.File();
        ASSERT(kingSquareIndex < 32);

        index = 48 + 4 * 64; // skip other pieces
        index += kingSquareIndex;
    }
    else
    {
        index = 48; // skip pawns
        index += ((uint32_t)piece - (uint32_t)Piece::Knight) * 64;
        index += relativeSquare.Index();
    }

    // opposite-side pieces features are in second half
    if (pieceColor != perspective) index += 32 + 48 + 4 * 64;

    ASSERT(index < (32 + 64 + 2 * (4 * 64 + 48)));

    return index;
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos)
{
    Position positionCopy = pos;
    if (pos.GetSideToMove() == Color::Black) positionCopy = pos.SwappedColors();

    const uint32_t maxFeatures = 64;
    uint16_t features[maxFeatures];
    const uint32_t numFeatures = positionCopy.ToFeaturesVector(features, NetworkInputMapping::Full_Symmetrical);
    ASSERT(numFeatures <= maxFeatures);

    const uint32_t variant = 0;
    //const uint32_t variant = std::min(7u, (pos.GetNumPieces() - 1u) / 4u);

    return network.Run(features, numFeatures, variant);
}

INLINE static void AppendFeatureIndex(uint16_t featureIndex, uint16_t addedFeatures[], uint32_t& numAddedFeatures, uint16_t removedFeatures[], uint32_t& numRemovedFeatures)
{
    for (uint32_t j = 0; j < numRemovedFeatures; ++j)
    {
        // if a feature to add is on list to remove, the addition and removal cancel each other
        if (featureIndex == removedFeatures[j])
        {
            removedFeatures[j] = removedFeatures[--numRemovedFeatures];
            return;
        }
    }
    
    addedFeatures[numAddedFeatures++] = featureIndex;
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

NO_INLINE static bool ReadAccumulatorCache(const Position& pos, const Color perspective, nn::Accumulator& outAccumulator)
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

static void UpdateAccumulator(const nn::PackedNeuralNetwork& network, const NodeInfo* prevAccumNode, const NodeInfo& node, const Color perspective)
{
    ASSERT(prevAccumNode != &node);
    nn::Accumulator& accumulator = node.nnContext->accumulator[(uint32_t)perspective];

    if (prevAccumNode)
    {
        ASSERT(prevAccumNode->nnContext);
        ASSERT(!prevAccumNode->nnContext->accumDirty[(uint32_t)perspective]);

        uint32_t numAddedFeatures = 0;
        uint32_t numRemovedFeatures = 0;
        uint16_t addedFeatures[64];
        uint16_t removedFeatures[64];

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
                    const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.toSquare, node.position, perspective);
                    AppendFeatureIndex(featureIdx, addedFeatures, numAddedFeatures, removedFeatures, numRemovedFeatures);
                }
                if (dirtyPiece.fromSquare.IsValid())
                {
                    const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.fromSquare, node.position, perspective);
                    AppendFeatureIndex(featureIdx, removedFeatures, numRemovedFeatures, addedFeatures, numAddedFeatures);
                }
            }
        }

#ifdef VALIDATE_NETWORK_OUTPUT
        {
            Position positionCopy = node.position;
            if (perspective == Color::Black) positionCopy = node.position.SwappedColors();
            const uint32_t maxFeatures = 64;
            uint16_t referenceFeatures[maxFeatures];
            const uint32_t numReferenceFeatures = positionCopy.ToFeaturesVector(referenceFeatures, NetworkInputMapping::Full_Symmetrical);

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

        accumulator.Update(
            prevAccumNode->nnContext->accumulator[(uint32_t)perspective],
            network.GetAccumulatorWeights(),
            network.GetNumInputs(), network.GetLayerSize(1),
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);
    }
    else // refresh accumulator
    {
        Position positionCopy = node.position;
        if (perspective == Color::Black) positionCopy = node.position.SwappedColors();

        const uint32_t maxFeatures = 64;
        uint16_t features[maxFeatures];
        const uint32_t numFeatures = positionCopy.ToFeaturesVector(features, NetworkInputMapping::Full_Symmetrical);
        ASSERT(numFeatures <= maxFeatures);

#ifdef NN_ACCUMULATOR_STATS
        s_NumAccumulatorRefreshes++;
#endif // NN_ACCUMULATOR_STATS

        accumulator.Refresh(
            network.GetAccumulatorWeights(), network.GetAccumulatorBiases(),
            network.GetNumInputs(), network.GetLayerSize(1),
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

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node)
{
    ASSERT(node.nnContext);

#ifndef VALIDATE_NETWORK_OUTPUT
    if (node.nnContext->nnScore != InvalidValue)
    {
        return node.nnContext->nnScore;
    }
#endif // VALIDATE_NETWORK_OUTPUT

    const Color perspective = node.position.GetSideToMove();
    nn::Accumulator& accumulator = node.nnContext->accumulator[(uint32_t)perspective];

    uint32_t updateCost = 0;
    const uint32_t refreshCost = node.position.GetNumPieces();

    const Bitboard leftFilesMask = 0x0F0F0F0F0F0F0F0Full;
    const bool currKingSide = (static_cast<const Position&>(node.position).GetCurrentSide().king & leftFilesMask) != 0;

    // find closest parent node that has valid accumulator
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

        // if king moved across left and right files boundary, then we need to refresh the accumulator
        const bool prevKingSide = (static_cast<const Position&>(nodePtr->position).GetCurrentSide().king & leftFilesMask) != 0;
        if (currKingSide != prevKingSide)
        {
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
    else if (node.parentNode &&
        node.parentNode != prevAccumNode &&
        node.parentNode->nnContext->accumDirty[(uint32_t)perspective])
    {
        // two-stage update:
        // if parent node has invalid accumulator, update it first
        // this way, sibling nodes can reuse parent's accumulator
        UpdateAccumulator(network, prevAccumNode, *node.parentNode, perspective);
        UpdateAccumulator(network, node.parentNode, node, perspective);
    }
    else
    {
        UpdateAccumulator(network, prevAccumNode, node, perspective);
    }

    const uint32_t variant = 0;
    //const uint32_t variant = std::min(7u, (node.position.GetNumPieces() - 1u) / 4u);

    const int32_t nnOutput = network.Run(accumulator, variant);

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
