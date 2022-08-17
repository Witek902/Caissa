#pragma once

#include "Common.hpp"
#include "Color.hpp"
#include "Piece.hpp"
#include "Square.hpp"
#include "PackedNeuralNetwork.hpp"

// position to NN input mapping mode
enum class NetworkInputMapping : uint8_t
{
	// full, 1-to-1 mapping with no symmetry
	// always 2*(5*64+48)=736 inputs
	Full,

	// similar to "Full" but with horizontal symetry (white king is on A-D files)
	// always 32+64+2*(4*64+48)=704 inputs
	Full_Symmetrical,

	// material key dependent, vertical and horizontal symmetry exploitation
	// number of inputs depends on material
	// horizontal symmetry in case of pawnful positions
	// horizontal and vertical symmetry in case of pawnless positions
	// TODO: diagonal symmetry
	MaterialPacked_Symmetrical,

	// king-relative
	KingPiece_Symmetrical,

	// king-relative
	MaterialPacked_KingPiece_Symmetrical,
};

struct DirtyPiece
{
	Piece piece;
	Color color;
	Square square;
};

struct alignas(CACHELINE_SIZE) NNEvaluatorContext
{
	// first layer accumulators for both perspectives
	nn::Accumulator accumulator[2];

	// indicates which accumulator is dirty
	bool accumDirty[2];

	// added and removed pieces information
	DirtyPiece addedPieces[2];
	DirtyPiece removedPieces[2];
	uint32_t numAddedPieces;
	uint32_t numRemovedPieces;

	NNEvaluatorContext()
	{
		MarkAsDirty();
	}

	void MarkAsDirty()
	{
		accumDirty[0] = true;
		accumDirty[1] = true;
		numAddedPieces = 0;
		numRemovedPieces = 0;
	}
};

class NNEvaluator
{
public:
	// evaluate a position from scratch
	static int32_t Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos, NetworkInputMapping mapping);

	// incrementally update and evaluate
	static int32_t Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node, NetworkInputMapping mapping);
};