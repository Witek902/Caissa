#pragma once

#include "Common.hpp"
#include "Color.hpp"
#include "Piece.hpp"
#include "Square.hpp"
#include "PackedNeuralNetwork.hpp"

struct DirtyPiece
{
	Piece piece;
	Color color;
	Square square;
};

struct NNEvaluatorContext
{
	// first layer accumulators for both perspectives
	nn::Accumulator accumulator[2];

	// indicates which accumulator is dirty
	bool accumDirty[2] = { true, true };

	// added and removed pieces information
	DirtyPiece addedPieces[2];
	DirtyPiece removedPieces[2];
	uint32_t numAddedPieces = 0;
	uint32_t numRemovedPieces = 0;

	void MarkAsDirty()
	{
		accumDirty[0] = accumDirty[1] = true;
	}
};

class NNEvaluator
{
public:
	// evaluate a position from scratch
	static int32_t Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos);

	// incrementally update and evaluate
	static int32_t Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node);
};