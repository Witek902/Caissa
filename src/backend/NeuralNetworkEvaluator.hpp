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

// Number of associative ways (cached accumulator states) per king-bucket slot.
// On a refresh we pick the closest cached way as the base and write the freshly
// computed state back into the least-recently-used way (LRU eviction).
// Supported values: 1 (legacy single-entry), 2 (128-bit SIMD scan), 4 (256-bit SIMD scan).
// 2/4 map exactly onto a 128/256-bit register's 64-bit lanes; any other value (and non-AVX2
// targets) falls back to the scalar scan.
static constexpr uint32_t NumCacheWays = 2;

struct AccumulatorCache
{
    struct alignas(CACHELINE_SIZE) KingBucket
    {
        nn::Accumulator accum[NumCacheWays];
        Bitboard pieces[2][6][NumCacheWays]; // [color][piece type][way] - structure-of-arrays for SIMD scan
        uint32_t lastUsed[NumCacheWays];     // LRU recency stamps (0 == never used)
        uint32_t clock;                      // monotonic counter driving LRU ordering
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
    static void GetStats(uint64_t& outNumUpdates, uint64_t& outNumRefreshes, uint64_t& outNumRefreshFeatures);
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
    const uint32_t numPieces = pos.GetNumPiecesExcludingKing();
    return static_cast<uint8_t>(std::min(numPieces / 4u, nn::NumVariants - 1u));
}

template<bool IncludePieceFeatures = false>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective);
