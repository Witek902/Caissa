#pragma once

#include "Common.hpp"
#include "Accumulator.hpp"
#include "Memory.hpp"
#include "Position.hpp"

#include <vector>

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

// Per-neuron accumulator statistics gathered over many evaluated positions (search-time collection).
// One instance is owned per search thread; instances are merged on demand for reporting.
struct NNAccumulatorStats
{
    static constexpr uint32_t NumPerspectives = 2;

    uint64_t numPositions = 0;
    uint64_t bucketUsage[nn::NumVariants] = {};                            // how often each output bucket was selected

    // master per-neuron totals (uint64); the hot path accumulates into the uint32 working
    // counters below and folds them in periodically (and before any read) to keep memory traffic low
    uint64_t activationCount[NumPerspectives][nn::AccumulatorSize] = {};   // count of value > 0
    uint64_t saturationCount[NumPerspectives][nn::AccumulatorSize] = {};   // count of value >= ActivationRangeScaling
    uint64_t sumActivation[NumPerspectives][nn::AccumulatorSize] = {};     // sum of clamp(value, 0, ActivationRangeScaling)

    // hot-path working accumulators (uint32, half the width of the masters) and the number of
    // positions accumulated into them since the last Flush()
    uint32_t workActivation[NumPerspectives][nn::AccumulatorSize] = {};
    uint32_t workSaturation[NumPerspectives][nn::AccumulatorSize] = {};
    uint32_t workSum[NumPerspectives][nn::AccumulatorSize] = {};
    uint32_t workCount = 0;

    void Reset();
    void Flush();                                      // fold working accumulators into the masters
    void Accumulate(const NNAccumulatorStats& other);  // merges masters only; both must be Flush()ed first
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

// Detailed per-position NNUE evaluation trace, used by the "eval detailed" UCI command.
// Designed so that aggregation over many positions (search-time stats) can reuse it later.
struct NNEvaluationTrace
{
    static constexpr uint32_t NumPerspectives = 2;

    // raw pre-activation accumulator values (int16)
    // perspective 0 = side to move, perspective 1 = the other side (matches network input order)
    int16_t rawAccumulator[NumPerspectives][nn::AccumulatorSize];

    // raw network output (internal units) for every output bucket variant
    int32_t bucketOutput[nn::NumVariants];

    // output bucket variant actually selected for this position
    uint32_t selectedVariant;
};

class NNEvaluator
{
public:
    // evaluate a position from scratch
    static int32_t Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos);

    // compute a detailed evaluation trace for a single position (from scratch)
    static void Trace(const nn::PackedNeuralNetwork& network, const Position& pos, NNEvaluationTrace& outTrace);

    // search-time accumulator statistics collection

    // enabling/disabling has (almost) zero overhead on the search hot path when disabled
    static void SetStatsCollectionEnabled(bool enabled);
    static bool IsStatsCollectionEnabled();
    // zero all per-thread counters (call only when no search is running)
    static void ResetStatsCollection();
    // merge all per-thread buffers into a single result (call only when no search is running)
    // optionally returns the per-thread position counts
    static void GetMergedStats(NNAccumulatorStats& outMerged, std::vector<uint64_t>* outPerThreadPositions = nullptr);

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
    const uint32_t numPieces = pos.GetNumPiecesExcludingKing();
    return static_cast<uint8_t>(std::min(numPieces / 4u, nn::NumVariants - 1u));
}

template<bool IncludePieceFeatures = false>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective);
