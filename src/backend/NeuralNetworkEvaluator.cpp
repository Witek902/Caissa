#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

#include <atomic>
#include <mutex>
#include <vector>
#include <cstring>

// enable validation of NN output (check if incremental updates work correctly)
//#define VALIDATE_NETWORK_OUTPUT

// search-time accumulator statistics collection

static std::atomic<bool> s_collectAccumulatorStats{ false };
static std::mutex s_statsRegistryMutex;
// per-thread statistics buffers, owned here for the lifetime of the process.
// each buffer is tied to an OS thread (which outlives any individual Search), so collected
// data persists across transient Search instances (e.g. 'bench') without any retire logic.
static std::vector<NNAccumulatorStats*> s_statsRegistry;
static thread_local NNAccumulatorStats* t_statsBuffer = nullptr;

void NNAccumulatorStats::Reset()
{
    numPositions = 0;
    memset(bucketUsage, 0, sizeof(bucketUsage));
    memset(activationCount, 0, sizeof(activationCount));
    memset(saturationCount, 0, sizeof(saturationCount));
    memset(sumActivation, 0, sizeof(sumActivation));
    memset(workActivation, 0, sizeof(workActivation));
    memset(workSaturation, 0, sizeof(workSaturation));
    memset(workSum, 0, sizeof(workSum));
    workCount = 0;
}

void NNAccumulatorStats::Flush()
{
    for (uint32_t p = 0; p < NumPerspectives; ++p)
    {
        for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
        {
            activationCount[p][i] += workActivation[p][i];
            saturationCount[p][i] += workSaturation[p][i];
            sumActivation[p][i] += workSum[p][i];
        }
    }
    memset(workActivation, 0, sizeof(workActivation));
    memset(workSaturation, 0, sizeof(workSaturation));
    memset(workSum, 0, sizeof(workSum));
    workCount = 0;
}

void NNAccumulatorStats::Accumulate(const NNAccumulatorStats& other)
{
    numPositions += other.numPositions;
    for (uint32_t v = 0; v < nn::NumVariants; ++v)
        bucketUsage[v] += other.bucketUsage[v];
    for (uint32_t p = 0; p < NumPerspectives; ++p)
    {
        for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
        {
            activationCount[p][i] += other.activationCount[p][i];
            saturationCount[p][i] += other.saturationCount[p][i];
            sumActivation[p][i] += other.sumActivation[p][i];
        }
    }
}

void NNEvaluator::SetStatsCollectionEnabled(bool enabled)
{
    s_collectAccumulatorStats.store(enabled, std::memory_order_relaxed);
}

bool NNEvaluator::IsStatsCollectionEnabled()
{
    return s_collectAccumulatorStats.load(std::memory_order_relaxed);
}

void NNEvaluator::ResetStatsCollection()
{
    std::lock_guard<std::mutex> lock(s_statsRegistryMutex);
    for (NNAccumulatorStats* buffer : s_statsRegistry)
        buffer->Reset();
}

void NNEvaluator::GetMergedStats(NNAccumulatorStats& outMerged, std::vector<uint64_t>* outPerThreadPositions)
{
    outMerged.Reset();
    if (outPerThreadPositions)
        outPerThreadPositions->clear();

    std::lock_guard<std::mutex> lock(s_statsRegistryMutex);
    for (NNAccumulatorStats* buffer : s_statsRegistry)
    {
        buffer->Flush(); // fold pending working counters into the masters (safe: called when no search is running)
        outMerged.Accumulate(*buffer);
        if (outPerThreadPositions)
            outPerThreadPositions->push_back(buffer->numPositions);
    }
}

// get (lazily allocating) the calling thread's statistics buffer
static NNAccumulatorStats* AcquireThreadStatsBuffer()
{
    if (!t_statsBuffer)
    {
        t_statsBuffer = new NNAccumulatorStats();
        std::lock_guard<std::mutex> lock(s_statsRegistryMutex);
        s_statsRegistry.push_back(t_statsBuffer);
    }
    return t_statsBuffer;
}

// fold working counters into the masters before they could overflow uint32:
// the largest working value is workSum (<= 255 per position), so flush well below 2^32/255
static constexpr uint32_t c_statsFlushInterval = 16000000u;

// collect per-neuron statistics from a single evaluated position (accumulates into the uint32 working
// counters; these are half the width of the masters and folded in periodically by Flush())
static void CollectAccumulatorStats(NNAccumulatorStats& stats, const nn::Accumulator& stmAccum, const nn::Accumulator& nstmAccum, uint32_t variant)
{
    stats.numPositions++;
    stats.bucketUsage[variant]++;

    const nn::Accumulator* accums[2] = { &stmAccum, &nstmAccum };
    for (uint32_t p = 0; p < NNAccumulatorStats::NumPerspectives; ++p)
    {
        const int16_t* values = accums[p]->values;
        uint32_t* wAct = stats.workActivation[p];
        uint32_t* wSat = stats.workSaturation[p];
        uint32_t* wSum = stats.workSum[p];

#if defined(USE_AVX2) || defined(USE_AVX512)
        const __m256i zero = _mm256_setzero_si256();
        const __m256i c254 = _mm256_set1_epi32(nn::ActivationRangeScaling - 1);
        const __m256i c255 = _mm256_set1_epi32(nn::ActivationRangeScaling);
        for (uint32_t i = 0; i < nn::AccumulatorSize; i += 8)
        {
            const __m256i x = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(values + i)));
            const __m256i maskPos = _mm256_cmpgt_epi32(x, zero);  // 0xFFFFFFFF where value > 0
            const __m256i maskSat = _mm256_cmpgt_epi32(x, c254);  // 0xFFFFFFFF where value >= 255
            const __m256i clamped = _mm256_min_epi32(_mm256_max_epi32(x, zero), c255);
            // mask is -1 where true, so subtracting it adds 1
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(wAct + i), _mm256_sub_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(wAct + i)), maskPos));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(wSat + i), _mm256_sub_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSat + i)), maskSat));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(wSum + i), _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(wSum + i)), clamped));
        }
#else
        for (uint32_t i = 0; i < nn::AccumulatorSize; ++i)
        {
            const int32_t x = values[i];
            wAct[i] += (x > 0) ? 1u : 0u;
            wSat[i] += (x >= nn::ActivationRangeScaling) ? 1u : 0u;
            wSum[i] += (uint32_t)(x < 0 ? 0 : (x > nn::ActivationRangeScaling ? nn::ActivationRangeScaling : x));
        }
#endif
    }

    if (++stats.workCount >= c_statsFlushInterval) [[unlikely]]
        stats.Flush();
}

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

void AccumulatorCache::Init(const nn::PackedNeuralNetwork* net)
{
    if (currentNet != net)
    {
        for (uint32_t c = 0; c < 2; ++c)
        {
            for (uint32_t b = 0; b < 2 * nn::NumKingBuckets; ++b)
            {
                memcpy(kingBuckets[c][b].accum.values, net->accumulatorBiases, sizeof(nn::AccumulatorType) * nn::AccumulatorSize);
                memset(kingBuckets[c][b].pieces, 0, sizeof(kingBuckets[c][b].pieces));
            }
        }
        currentNet = net;
    }
}

template<bool IncludePieceFeatures>
uint32_t PositionToFeaturesVector(const Position& pos, uint16_t* outFeatures, const Color perspective)
{
    uint32_t numFeatures = 0;

    const auto& whites = pos.GetSide(perspective);
    const auto& blacks = pos.GetSide(perspective ^ 1);

    Square kingSquare = whites.GetKingSquare();

    uint32_t bitFlipMask = 0;

    if (kingSquare.File() >= 4)
    {
        // flip file
        kingSquare = kingSquare.FlippedFile();
        bitFlipMask = 0b000111;
    }

    if (perspective == Black)
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
    if constexpr (perspective == Black)
    {
        square = square.FlippedRank();
        kingSquare = kingSquare.FlippedRank();
    }

    // flip the according to the king placement
    if (kingSquare.File() >= 4)
    {
        square = square.FlippedFile();
        // Note: no need to flip the file of the king square, because KingBucketIndex is symmetric
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

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, const Position& pos)
{
    constexpr uint32_t maxFeatures = 64;

    uint16_t ourFeatures[maxFeatures];
    const uint32_t numOurFeatures = PositionToFeaturesVector(pos, ourFeatures, pos.GetSideToMove());
    ASSERT(numOurFeatures <= maxFeatures);

    uint16_t theirFeatures[maxFeatures];
    const uint32_t numTheirFeatures = PositionToFeaturesVector(pos, theirFeatures, pos.GetSideToMove() ^ 1);
    ASSERT(numTheirFeatures <= maxFeatures);

    return network.Run(ourFeatures, numOurFeatures, theirFeatures, numTheirFeatures, GetNetworkVariant(pos));
}

void NNEvaluator::Trace(const nn::PackedNeuralNetwork& network, const Position& pos, NNEvaluationTrace& outTrace)
{
    constexpr uint32_t maxFeatures = 64;

    uint16_t stmFeatures[maxFeatures];
    const uint32_t numStmFeatures = PositionToFeaturesVector(pos, stmFeatures, pos.GetSideToMove());
    ASSERT(numStmFeatures <= maxFeatures);

    uint16_t nstmFeatures[maxFeatures];
    const uint32_t numNstmFeatures = PositionToFeaturesVector(pos, nstmFeatures, pos.GetSideToMove() ^ 1);
    ASSERT(numNstmFeatures <= maxFeatures);

    nn::Accumulator stmAccum;
    stmAccum.Refresh(network.accumulatorWeights, network.accumulatorBiases, numStmFeatures, stmFeatures);

    nn::Accumulator nstmAccum;
    nstmAccum.Refresh(network.accumulatorWeights, network.accumulatorBiases, numNstmFeatures, nstmFeatures);

    static_assert(sizeof(outTrace.rawAccumulator[0]) == sizeof(stmAccum.values), "accumulator size mismatch");
    memcpy(outTrace.rawAccumulator[0], stmAccum.values, sizeof(stmAccum.values));
    memcpy(outTrace.rawAccumulator[1], nstmAccum.values, sizeof(nstmAccum.values));

    // run the last layer for every output bucket variant
    for (uint32_t v = 0; v < nn::NumVariants; ++v)
        outTrace.bucketOutput[v] = network.Run(stmAccum, nstmAccum, v);

    outTrace.selectedVariant = GetNetworkVariant(pos);
}

template<Color perspective>
INLINE static void UpdateAccumulator(const nn::PackedNeuralNetwork& network, const NodeInfo* prevAccumNode, NodeInfo& node, AccumulatorCache::KingBucket& cache)
{
    constexpr uint32_t color = (uint32_t)perspective;

    ASSERT(prevAccumNode != &node);
    ASSERT(node.nnContext.accumDirty[color]);

    constexpr uint32_t maxChangedFeatures = 64;
    uint32_t numAddedFeatures = 0;
    uint32_t numRemovedFeatures = 0;
    uint16_t addedFeatures[maxChangedFeatures];
    uint16_t removedFeatures[maxChangedFeatures];

    if (prevAccumNode)
    {
        ASSERT(!prevAccumNode->nnContext.accumDirty[color]);

        // Add feature to one list, or cancel it against the opposite list if already present.
        // This keeps both lists bounded by piece count on the board (max 64 squares), regardless of how many plies are walked.
        const auto addOrCancel = [](uint16_t featureIdx,
            uint16_t* targetList, uint32_t& targetCount,
            uint16_t* oppositeList, uint32_t& oppositeCount) INLINE_LAMBDA
        {
            for (uint32_t j = 0; j < oppositeCount; ++j)
            {
                if (oppositeList[j] == featureIdx)
                {
                    oppositeList[j] = oppositeList[--oppositeCount];
                    return;
                }
            }
            ASSERT(targetCount < maxChangedFeatures);
            targetList[targetCount++] = featureIdx;
        };

        // build a list of features to be updated
        for (const NodeInfo* nodePtr = &node; nodePtr != prevAccumNode; --nodePtr)
        {
            const NNEvaluatorContext& nnContext = nodePtr->nnContext;

            for (uint32_t i = 0; i < nnContext.numDirtyPieces; ++i)
            {
                const DirtyPiece& dirtyPiece = nnContext.dirtyPieces[i];

                if (dirtyPiece.toSquare.IsValid())
                {
                    const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex<perspective>(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.toSquare, node.position);
                    addOrCancel(featureIdx, addedFeatures, numAddedFeatures, removedFeatures, numRemovedFeatures);
                }
                if (dirtyPiece.fromSquare.IsValid())
                {
                    const uint16_t featureIdx = (uint16_t)DirtyPieceToFeatureIndex<perspective>(dirtyPiece.piece, dirtyPiece.color, dirtyPiece.fromSquare, node.position);
                    addOrCancel(featureIdx, removedFeatures, numRemovedFeatures, addedFeatures, numAddedFeatures);
                }
            }

            if (nodePtr->ply == 0)
            {
                // reached end of stack
                break;
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
            // accumulator is unchanged, just point to the previous accumulator
            node.accumulatorPtr[color] = prevAccumNode->accumulatorPtr[color];
        }
        else
        {
            node.accumulatorPtr[color] = &node.accumulatorData[color];
            nn::Accumulator::Update(
                node.accumulatorData[color],
                *(prevAccumNode->accumulatorPtr[color]),
                network.accumulatorWeights,
                numAddedFeatures, addedFeatures,
                numRemovedFeatures, removedFeatures);
        }
    }
    else // refresh accumulator
    {
        for (Color c = 0; c < 2; ++c)
        {
            const Position& pos = node.position;
            const Bitboard* bitboards = &pos.GetSide(c).pawns;
            for (uint32_t p = 0; p < 6; ++p)
            {
                const Bitboard prev = cache.pieces[c][p];
                const Bitboard curr = bitboards[p];
                const Piece piece = (Piece)(p + (uint32_t)Piece::Pawn);

                // additions
                (curr & ~prev).Iterate([&](const Square sq) INLINE_LAMBDA
                {
                    ASSERT(numAddedFeatures < maxChangedFeatures);
                    addedFeatures[numAddedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<perspective>(piece, c, sq, pos);
                });

                // removals
                (prev & ~curr).Iterate([&](const Square sq) INLINE_LAMBDA
                {
                    ASSERT(numRemovedFeatures < maxChangedFeatures);
                    removedFeatures[numRemovedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<perspective>(piece, c, sq, pos);
                });

                cache.pieces[c][p] = curr;
            }
        }

        nn::Accumulator::Update(
            cache.accum,
            node.accumulatorData[color],
            cache.accum,
            network.accumulatorWeights,
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);

        node.accumulatorPtr[color] = &node.accumulatorData[color];
    }

    // mark accumulator as computed
    node.nnContext.accumDirty[color] = false;
}

template<Color perspective>
INLINE static void RefreshAccumulator(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache)
{
    constexpr uint32_t color = (uint32_t)perspective;
    const Position& pos = node.position;

    uint32_t kingSide, kingBucket;
    if constexpr (perspective == White)
        GetKingSideAndBucket(pos.Whites().GetKingSquare(), kingSide, kingBucket);
    else
        GetKingSideAndBucket(pos.Blacks().GetKingSquare().FlippedRank(), kingSide, kingBucket);

    AccumulatorCache::KingBucket& kingBucketCache = cache.kingBuckets[color][kingBucket + kingSide * nn::NumKingBuckets];

    // find closest parent node that has valid accumulator
    NodeInfo* prevAccumNode = nullptr;
    for (NodeInfo* nodePtr = &node; ; --nodePtr)
    {
        uint32_t newKingSide, newKingBucket;
        if constexpr (perspective == White)
            GetKingSideAndBucket(static_cast<const Position&>(nodePtr->position).Whites().GetKingSquare(), newKingSide, newKingBucket);
        else
            GetKingSideAndBucket(static_cast<const Position&>(nodePtr->position).Blacks().GetKingSquare().FlippedRank(), newKingSide, newKingBucket);

        if (newKingSide != kingSide || newKingBucket != kingBucket)
        {
            // king moved, accumulator needs to be refreshed
            break;
        }

        if (!nodePtr->nnContext.accumDirty[color])
        {
            // found parent node with valid accumulator
            prevAccumNode = nodePtr;
            break;
        }

        if (nodePtr->ply == 0)
        {
            // reached end of stack
            break;
        }
    }

    if (prevAccumNode == &node)
    {
        // accumulator is already up to date (was cached)
        return;
    }

    if (prevAccumNode)
    {
        // Update every dirty accumulator on the path, from the closest valid ancestor down to this node.
        // Filling in the intermediate accumulators lets sibling nodes reuse them.
        for (NodeInfo* nodePtr = prevAccumNode + 1; nodePtr <= &node; ++nodePtr)
        {
            UpdateAccumulator<perspective>(network, nodePtr - 1, *nodePtr, kingBucketCache);
        }
    }
    else
    {
        // no valid ancestor found - refresh from the king bucket cache
        UpdateAccumulator<perspective>(network, nullptr, node, kingBucketCache);
    }
}

int32_t NNEvaluator::Evaluate(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache)
{
#ifndef VALIDATE_NETWORK_OUTPUT
    if (node.nnContext.nnScore != InvalidValue)
    {
        return node.nnContext.nnScore;
    }
#endif // VALIDATE_NETWORK_OUTPUT

    RefreshAccumulator<White>(network, node, cache);
    RefreshAccumulator<Black>(network, node, cache);

    const nn::Accumulator& ourAccumulator = *node.accumulatorPtr[(uint32_t)node.position.GetSideToMove()];
    const nn::Accumulator& theirAccumulator = *node.accumulatorPtr[(uint32_t)node.position.GetSideToMove() ^ 1u];
    const int32_t nnOutput = network.Run(ourAccumulator, theirAccumulator, GetNetworkVariant(node.position));

#ifdef VALIDATE_NETWORK_OUTPUT
    {
        const int32_t nnOutputReference = Evaluate(network, node.position);
        ASSERT(nnOutput == nnOutputReference);
    }
    if (node.nnContext.nnScore != InvalidValue)
    {
        ASSERT(node.nnContext.nnScore == nnOutput);
    }
#endif // VALIDATE_NETWORK_OUTPUT

    // cache NN output
    node.nnContext.nnScore = nnOutput;

    // search-time statistics collection (disabled by default: single relaxed load + predicted-not-taken branch)
    if (s_collectAccumulatorStats.load(std::memory_order_relaxed)) [[unlikely]]
    {
        CollectAccumulatorStats(*AcquireThreadStatsBuffer(), ourAccumulator, theirAccumulator, GetNetworkVariant(node.position));
    }

    return nnOutput;
}

void NNEvaluator::EnsureAccumulatorUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache)
{
    RefreshAccumulator<White>(network, node, cache);
    RefreshAccumulator<Black>(network, node, cache);
}
