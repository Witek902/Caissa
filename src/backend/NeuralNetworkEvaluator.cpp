#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

// enable validation of NN output (check if incremental updates work correctly)
// #define VALIDATE_NETWORK_OUTPUT

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

void NNEvaluator::EnsureAccumulatorUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& node)
{
    RefreshAccumulator<Color::White>(network, node);
    RefreshAccumulator<Color::Black>(network, node);
}
