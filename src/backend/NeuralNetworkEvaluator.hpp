#pragma once

#include "Common.hpp"
#include "Memory.hpp"
#include "Color.hpp"
#include "Piece.hpp"
#include "Square.hpp"
#include "PackedNeuralNetwork.hpp"

//#define NN_ACCUMULATOR_STATS

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

    INLINE void MarkAsDirty()
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

    // update accumulators without evaluating
    static void EnsureAccumulatorUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& node);

#ifdef NN_ACCUMULATOR_STATS
    static void GetStats(uint64_t& outNumUpdates, uint64_t& outNumRefreshes);
    static void ResetStats();
#endif // NN_ACCUMULATOR_STATS
};


INLINE void GetKingSideAndBucket(Square kingSquare, uint32_t& side, uint32_t& bucket)
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

    bucket = nn::KingBucketIndex[kingSquare.Index()];
    ASSERT(bucket < nn::NumKingBuckets);
}

uint32_t GetNetworkVariant(const Position& pos);

template<bool IncludePieceFeatures = false>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective);
