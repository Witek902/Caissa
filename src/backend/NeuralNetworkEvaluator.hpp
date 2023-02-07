#pragma once

#include "Common.hpp"
#include "Memory.hpp"
#include "Color.hpp"
#include "Piece.hpp"
#include "Square.hpp"
#include "PackedNeuralNetwork.hpp"

// #define NN_ACCUMULATOR_STATS

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
	Square fromSquare;
	Square toSquare;
};

// promotion with capture
static constexpr uint32_t MaxNumDirtyPieces = 3;

struct alignas(CACHELINE_SIZE) NNEvaluatorContext
{
	// first layer accumulators for both perspectives
	nn::Accumulator accumulator[2];

	// indicates which accumulator is dirty
	bool accumDirty[2];

	// added and removed pieces information
	DirtyPiece dirtyPieces[MaxNumDirtyPieces];
	uint32_t numDirtyPieces;

	// cache NN output
	int32_t nnScore;

	void* operator new(size_t size)
	{
		return AlignedMalloc(size, CACHELINE_SIZE);
	}

	void operator delete(void* ptr)
	{
		AlignedFree(ptr);
	}

	NNEvaluatorContext()
	{
		MarkAsDirty();
	}

	void MarkAsDirty()
	{
		accumDirty[0] = true;
		accumDirty[1] = true;
		numDirtyPieces = 0;
		nnScore = InvalidValue;
	}
};

class NNEvaluator
{
public:
	// evaluate a position from scratch
	static int32_t Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos);

	// incrementally update and evaluate
	static int32_t Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node);

#ifdef NN_ACCUMULATOR_STATS
	static void GetStats(uint64_t& outNumUpdates, uint64_t& outNumRefreshes);
	static void ResetStats();
#endif // NN_ACCUMULATOR_STATS
};