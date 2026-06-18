#pragma once

#include "Common.hpp"
#include "Accumulator.hpp"
#include "Memory.hpp"
#include "Position.hpp"

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

struct NNEvaluatorContext
{
    // indicates which accumulator is dirty
    bool accumDirty[2];

    // added and removed pieces information
    DirtyPiece dirtyPieces[MaxNumDirtyPieces];
    uint32_t numDirtyPieces;

    // cache NN output
    int32_t nnScore;

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

struct AccumulatorCache
{
    struct KingBucket
    {
        nn::Accumulator accum;
        Bitboard pieces[2][6]; // [color][piece type]
    };
    KingBucket kingBuckets[2][2 * nn::NumKingBuckets]; // [side to move][king side * king bucket]
    const nn::PackedNeuralNetwork* currentNet = nullptr;

    void Init(const nn::PackedNeuralNetwork* net);
};

class NNEvaluator
{
public:
    // evaluate a position from scratch
    static int32_t Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos);

    // incrementally update and evaluate
    static int32_t Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache);

    // update accumulators without evaluating
    static void EnsureAccumulatorUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache);

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

INLINE uint8_t GetNetworkVariant(const Position& pos)
{
    // Non-linear bucket mapping (numPieces excluding kings):
    // B0: 2-3,  B1: 4-5,  B2: 6-8,  B3: 9-11,  B4: 12-15,  B5: 16-20,  B6: 21-25,  B7: 26+
    constexpr const uint8_t variantTable[63] =
    {
        0, 0,               // 0 and 1 are never used in neural network evaluation
        0, 0,               // B0: 2-3
        1, 1,               // B1: 4-5
        2, 2, 2,            // B2: 6-8
        3, 3, 3,            // B3: 9-11
        4, 4, 4, 4,         // B4: 12-15
        5, 5, 5, 5, 5,      // B5: 16-20
        6, 6, 6, 6, 6,      // B6: 21-25
        7, 7, 7, 7, 7,      // B7: 26+
        7, 7, 7, 7, 7,
        7, 7, 7, 7, 7,
        7, 7, 7, 7, 7,
        7, 7, 7, 7, 7,
        7, 7, 7, 7, 7,
        7, 7, 7, 7, 7,
        7, 7,
    };
    return variantTable[pos.GetNumPiecesExcludingKing()];
}

template<bool IncludePieceFeatures = false>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective);
