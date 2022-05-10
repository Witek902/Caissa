#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"


INLINE static uint16_t DirtyPieceToFeatureIndex(const DirtyPiece piece, Color perspective)
{
    // this must match Position::ToSparseFeaturesVector !!!

    // flip the square according to the perspective
    const Square relativeSquare = perspective == Color::White ? piece.square : piece.square.FlippedRank();

    uint16_t index;
    if (piece.piece == Piece::Pawn)
    {
        ASSERT(relativeSquare.Rank() > 0 && relativeSquare.Rank() < 7);
        index = relativeSquare.Index() - 8;
    }
    else
    {
        index = 48; // skip pawns
        index += ((uint32_t)piece.piece - (uint32_t)Piece::Knight) * 64;
        index += relativeSquare.Index();
    }

    // opposite-side pieces features are in second half
    if (piece.color != perspective) index += 48 + 5 * 64;

    ASSERT(index < 2 * 5 * 64 + 2 * 48);

    return index;
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos)
{
    Position positionCopy = pos;
    if (pos.GetSideToMove() == Color::Black) positionCopy = pos.SwappedColors();

    const uint32_t maxFeatures = 64;
    uint16_t features[maxFeatures];
    const uint32_t numFeatures = positionCopy.ToSparseFeaturesVector(features);
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

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node)
{
    const uint32_t perspective = (uint32_t)node.position.GetSideToMove();
    nn::Accumulator& accumulator = node.nnContext.accumulator[perspective];

    uint32_t updateCost = 0;
	const uint32_t refreshCost = node.position.GetNumPieces();

    const NodeInfo* prevAccumNode = nullptr;
    for (const NodeInfo* nodePtr = &node; nodePtr != nullptr; nodePtr = nodePtr->parentNode)
    {
        updateCost += (nodePtr->nnContext.numAddedPieces + nodePtr->nnContext.numRemovedPieces);
        if (updateCost > refreshCost)
        {
            // update cost higher than refresh cost, incremetal update not worth it
            break;
        }

        if (!nodePtr->nnContext.accumDirty[perspective])
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
            for (uint32_t i = 0; i < nodePtr->nnContext.numAddedPieces; ++i)
            {
                const int16_t featureIdx = DirtyPieceToFeatureIndex(nodePtr->nnContext.addedPieces[i], node.position.GetSideToMove());
                AppendFeatureIndex(featureIdx, addedFeatures, numAddedFeatures, removedFeatures, numRemovedFeatures);
            }

            for (uint32_t i = 0; i < nodePtr->nnContext.numRemovedPieces; ++i)
            {
                const int16_t featureIdx = DirtyPieceToFeatureIndex(nodePtr->nnContext.removedPieces[i], node.position.GetSideToMove());
                AppendFeatureIndex(featureIdx, removedFeatures, numRemovedFeatures, addedFeatures, numAddedFeatures);
            }
        }

        accumulator.Update(
            prevAccumNode->nnContext.accumulator[perspective],
            network.GetAccumulatorWeights(),
            network.GetNumInputs(), nn::FirstLayerSize,
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);
    }
    else // refresh accumulator
    {
        Position positionCopy = node.position;
        if (node.position.GetSideToMove() == Color::Black) positionCopy = node.position.SwappedColors();

        const uint32_t maxFeatures = 64;
        uint16_t features[maxFeatures];
        const uint32_t numFeatures = positionCopy.ToSparseFeaturesVector(features);
        ASSERT(numFeatures <= maxFeatures);

        accumulator.Refresh(
            network.GetAccumulatorWeights(), network.GetAccumulatorBiases(),
            network.GetNumInputs(), nn::FirstLayerSize,
            numFeatures, features);
    }

    const int32_t nnOutput = network.Run(accumulator);

    //{
    //    const int32_t nnOutputReference = Evaluate(network, node.position);
    //    ASSERT(nnOutput == nnOutputReference);
    //}

    // mark accumulator as computed
    node.nnContext.accumDirty[perspective] = false;

	return nnOutput;
}
