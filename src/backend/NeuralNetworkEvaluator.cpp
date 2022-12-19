#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

// #define VALIDATE_NETWORK_OUTPUT

INLINE static uint32_t DirtyPieceToFeatureIndex(const DirtyPiece piece, const Position& pos)
{
    // this must match Position::ToSparseFeaturesVector !!!

    const Color perspective = pos.GetSideToMove();

    Square relativeSquare = piece.square;
    {
        // flip the according to the perspective
        if (perspective == Color::Black) relativeSquare = relativeSquare.FlippedRank();

        // flip the according to the king placement
        const Bitboard leftFilesMask = 0x0F0F0F0F0F0F0F0Full;
        if ((pos.GetCurrentSide().king & leftFilesMask) == 0) relativeSquare = relativeSquare.FlippedFile();
    }

    uint32_t index;
    if (piece.piece == Piece::Pawn)
    {
        ASSERT(relativeSquare.Rank() > 0 && relativeSquare.Rank() < 7);
        index = relativeSquare.Index() - 8;
    }
    else if (piece.piece == Piece::King && piece.color == perspective)
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
        index += ((uint32_t)piece.piece - (uint32_t)Piece::Knight) * 64;
        index += relativeSquare.Index();
    }

    // opposite-side pieces features are in second half
    if (piece.color != perspective) index += 32 + 48 + 4 * 64;

    ASSERT(index < (32 + 64 + 2 * (4 * 64 + 48)));

    return index;
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos, NetworkInputMapping mapping)
{
    Position positionCopy = pos;
    if (pos.GetSideToMove() == Color::Black) positionCopy = pos.SwappedColors();

    const uint32_t maxFeatures = 64;
    uint16_t features[maxFeatures];
    const uint32_t numFeatures = positionCopy.ToFeaturesVector(features, mapping);
    ASSERT(numFeatures <= maxFeatures);

    return network.Run(features, numFeatures);
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

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node, NetworkInputMapping mapping)
{
    ASSERT(node.nnContext);

#ifndef VALIDATE_NETWORK_OUTPUT
	if (node.nnContext->nnScore != InvalidValue)
	{
		return node.nnContext->nnScore;
	}
#endif // VALIDATE_NETWORK_OUTPUT

    const uint32_t perspective = (uint32_t)node.position.GetSideToMove();
    nn::Accumulator& accumulator = node.nnContext->accumulator[perspective];

    uint32_t updateCost = 0;
    const uint32_t refreshCost = node.position.GetNumPieces();

    const Bitboard leftFilesMask = 0x0F0F0F0F0F0F0F0Full;
    const bool currKingSide = (static_cast<const Position&>(node.position).GetCurrentSide().king & leftFilesMask) != 0;

    const NodeInfo* prevAccumNode = nullptr;
    for (const NodeInfo* nodePtr = &node; nodePtr != nullptr; nodePtr = nodePtr->parentNode)
    {
        ASSERT(nodePtr->nnContext);

        updateCost += (nodePtr->nnContext->numAddedPieces + nodePtr->nnContext->numRemovedPieces);
        if (updateCost > refreshCost)
        {
            // update cost higher than refresh cost, incremetal update not worth it
            break;
        }

        // if king moved accros left and right files boundary, then we need to refresh the accumulator
        const bool prevKingSide = (static_cast<const Position&>(nodePtr->position).GetCurrentSide().king & leftFilesMask) != 0;
        if (currKingSide != prevKingSide)
        {
            break;
        }

        if (!nodePtr->nnContext->accumDirty[perspective])
        {
            // found parent node with valid accumulator
            prevAccumNode = nodePtr;
            break;
        }
    }

    if (prevAccumNode)
    {
        uint32_t numAddedFeatures = 0;
        uint32_t numRemovedFeatures = 0;
        uint16_t addedFeatures[64];
        uint16_t removedFeatures[64];

        // build a list of features to be updated
        for (const NodeInfo* nodePtr = &node; nodePtr != prevAccumNode; nodePtr = nodePtr->parentNode)
        {
            NNEvaluatorContext& nnContext = *(nodePtr->nnContext);

            for (uint32_t i = 0; i < nnContext.numAddedPieces; ++i)
            {
                const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex(nnContext.addedPieces[i], node.position);
                AppendFeatureIndex(featureIdx, addedFeatures, numAddedFeatures, removedFeatures, numRemovedFeatures);
            }

            for (uint32_t i = 0; i < nnContext.numRemovedPieces; ++i)
            {
                const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex(nnContext.removedPieces[i], node.position);
                AppendFeatureIndex(featureIdx, removedFeatures, numRemovedFeatures, addedFeatures, numAddedFeatures);
            }
        }

#ifdef VALIDATE_NETWORK_OUTPUT
        Position positionCopy = node.position;
        if (node.position.GetSideToMove() == Color::Black) positionCopy = node.position.SwappedColors();
        const uint32_t maxFeatures = 64;
        uint16_t referenceFeatures[maxFeatures];
        const uint32_t numReferenceFeatures = positionCopy.ToFeaturesVector(referenceFeatures, mapping);

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
#endif // VALIDATE_NETWORK_OUTPUT

        accumulator.Update(
            prevAccumNode->nnContext->accumulator[perspective],
            network.GetAccumulatorWeights(),
            network.GetNumInputs(), network.GetLayerSize(1),
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);
    }
    else // refresh accumulator
    {
        Position positionCopy = node.position;
        if (node.position.GetSideToMove() == Color::Black) positionCopy = node.position.SwappedColors();

        const uint32_t maxFeatures = 64;
        uint16_t features[maxFeatures];
        const uint32_t numFeatures = positionCopy.ToFeaturesVector(features, mapping);
        ASSERT(numFeatures <= maxFeatures);

        accumulator.Refresh(
            network.GetAccumulatorWeights(), network.GetAccumulatorBiases(),
            network.GetNumInputs(), network.GetLayerSize(1),
            numFeatures, features);
    }

    const int32_t nnOutput = network.Run(accumulator);

#ifdef VALIDATE_NETWORK_OUTPUT
    {
        const int32_t nnOutputReference = Evaluate(network, node.position, mapping);
        ASSERT(nnOutput == nnOutputReference);
    }
	if (node.nnContext->nnScore != InvalidValue)
	{
        ASSERT(node.nnContext->nnScore == nnOutput);
	}
#endif // VALIDATE_NETWORK_OUTPUT

    // mark accumulator as computed
    node.nnContext->accumDirty[perspective] = false;

    // cache NN output
    node.nnContext->nnScore = nnOutput;

    return nnOutput;
}
