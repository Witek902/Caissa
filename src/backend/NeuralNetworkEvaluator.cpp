#include "NeuralNetworkEvaluator.hpp"
#include "Search.hpp"

// enable validation of NN output (check if incremental updates work correctly)
//#define VALIDATE_NETWORK_OUTPUT

#ifdef NN_ACCUMULATOR_STATS

static std::atomic<uint64_t> s_NumAccumulatorUpdates = 0;
static std::atomic<uint64_t> s_NumAccumulatorRefreshes = 0;
static std::atomic<uint64_t> s_NumAccumulatorRefreshFeatures = 0;

void NNEvaluator::GetStats(uint64_t& outNumUpdates, uint64_t& outNumRefreshes, uint64_t& outNumRefreshFeatures)
{
    outNumUpdates = s_NumAccumulatorUpdates;
    outNumRefreshes = s_NumAccumulatorRefreshes;
    outNumRefreshFeatures = s_NumAccumulatorRefreshFeatures;
}
void NNEvaluator::ResetStats()
{
    s_NumAccumulatorUpdates = 0;
    s_NumAccumulatorRefreshes = 0;
    s_NumAccumulatorRefreshFeatures = 0;
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
                KingBucket& kingBucket = kingBuckets[c][b];
                for (uint32_t w = 0; w < NumCacheWays; ++w)
                {
                    memcpy(kingBucket.accum[w].values, net->accumulatorBiases, sizeof(nn::AccumulatorType) * nn::AccumulatorSize);
                    kingBucket.lastUsed[w] = 0;
                }
                memset(kingBucket.pieces, 0, sizeof(kingBucket.pieces));
                kingBucket.clock = 0;
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

#if defined(NN_USE_AVX2) || defined(NN_USE_AVX512)
// bytewise population count of a 256-bit vector (Mula's nibble-LUT via pshufb)
INLINE static __m256i Avx2PopcountBytes(const __m256i v)
{
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i lowMask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, lowMask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
}
// bytewise population count of a 128-bit vector (Mula's nibble-LUT via pshufb; SSSE3)
INLINE static __m128i Sse3PopcountBytes(const __m128i v)
{
    const __m128i lookup = _mm_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m128i lowMask = _mm_set1_epi8(0x0f);
    const __m128i lo = _mm_and_si128(v, lowMask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), lowMask);
    return _mm_add_epi8(_mm_shuffle_epi8(lookup, lo), _mm_shuffle_epi8(lookup, hi));
}
#endif // NN_USE_AVX2 || NN_USE_AVX512

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
        const Position& pos = node.position;

        // gather current piece bitboards [color][piece type]
        Bitboard curr[2][6];
        for (Color c = 0; c < 2; ++c)
        {
            const Bitboard* bitboards = &pos.GetSide(c).pawns;
            for (uint32_t p = 0; p < 6; ++p)
                curr[c][p] = bitboards[p];
        }

        // scan all cache ways for the closest one (fewest differing piece bits)
        uint64_t diff[NumCacheWays];
#if defined(NN_USE_AVX2) || defined(NN_USE_AVX512)
        // 4 ways map exactly onto one 256-bit register (4 x uint64)
        if constexpr (NumCacheWays == 4)
        {
            const __m256i zero = _mm256_setzero_si256();
            // accumulate bytewise popcounts, then do a single horizontal sum after the loop
            __m256i byteAcc = zero;
            for (Color c = 0; c < 2; ++c)
            {
                for (uint32_t p = 0; p < 6; ++p)
                {
                    const __m256i ways = _mm256_load_si256(reinterpret_cast<const __m256i*>(&cache.pieces[c][p][0]));
                    const __m256i cur = _mm256_set1_epi64x((long long)curr[c][p].value);
                    byteAcc = _mm256_add_epi8(byteAcc, Avx2PopcountBytes(_mm256_xor_si256(ways, cur)));
                }
            }
            // _mm256_sad_epu8 sums the 8 bytes within each 64-bit lane -> popcount per way
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(diff), _mm256_sad_epu8(byteAcc, zero));
        }
        else if constexpr (NumCacheWays == 2)
        {
            // 2 ways map exactly onto one 128-bit register (2 x uint64)
            const __m128i zero = _mm_setzero_si128();
            __m128i byteAcc = zero;
            for (Color c = 0; c < 2; ++c)
            {
                for (uint32_t p = 0; p < 6; ++p)
                {
                    const __m128i ways = _mm_load_si128(reinterpret_cast<const __m128i*>(&cache.pieces[c][p][0]));
                    const __m128i cur = _mm_set1_epi64x((long long)curr[c][p].value);
                    byteAcc = _mm_add_epi8(byteAcc, Sse3PopcountBytes(_mm_xor_si128(ways, cur)));
                }
            }
            // _mm_sad_epu8 sums the 8 bytes within each 64-bit lane -> popcount per way
            _mm_storeu_si128(reinterpret_cast<__m128i*>(diff), _mm_sad_epu8(byteAcc, zero));
        }
        else
        // TODO SSE and NEON implementations
#endif // NN_USE_AVX2 || NN_USE_AVX512
        {
            // scalar fallback
            for (uint32_t w = 0; w < NumCacheWays; ++w)
                diff[w] = 0;
            for (Color c = 0; c < 2; ++c)
                for (uint32_t p = 0; p < 6; ++p)
                    for (uint32_t w = 0; w < NumCacheWays; ++w)
                        diff[w] += PopCount(cache.pieces[c][p][w].value ^ curr[c][p].value);
        }

        // find the way with the fewest differing bits
        uint32_t baseWay = 0;
        for (uint32_t w = 1; w < NumCacheWays; ++w)
            if (diff[w] < diff[baseWay])
                baseWay = w;

#ifdef NN_ACCUMULATOR_STATS
        s_NumAccumulatorRefreshes++;
#endif // NN_ACCUMULATOR_STATS

        if (diff[baseWay] == 0)
        {
            // exact hit: the cached accumulator already matches the position
            memcpy(node.accumulatorData[color].values, cache.accum[baseWay].values, sizeof(nn::AccumulatorType) * nn::AccumulatorSize);
            cache.lastUsed[baseWay] = ++cache.clock;
            node.accumulatorPtr[color] = &node.accumulatorData[color];
            node.nnContext.accumDirty[color] = false;
            return;
        }

        // build the feature delta against the closest way
        for (Color c = 0; c < 2; ++c)
        {
            for (uint32_t p = 0; p < 6; ++p)
            {
                const Bitboard prev = cache.pieces[c][p][baseWay];
                const Bitboard cur = curr[c][p];
                const Piece piece = (Piece)(p + (uint32_t)Piece::Pawn);

                // additions
                (cur & ~prev).Iterate([&](const Square sq) INLINE_LAMBDA
                {
                    ASSERT(numAddedFeatures < maxChangedFeatures);
                    addedFeatures[numAddedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<perspective>(piece, c, sq, pos);
                });

                // removals
                (prev & ~cur).Iterate([&](const Square sq) INLINE_LAMBDA
                {
                    ASSERT(numRemovedFeatures < maxChangedFeatures);
                    removedFeatures[numRemovedFeatures++] = (uint16_t)DirtyPieceToFeatureIndex<perspective>(piece, c, sq, pos);
                });
            }
        }

        ASSERT(numAddedFeatures > 0 || numRemovedFeatures > 0);

#ifdef NN_ACCUMULATOR_STATS
        s_NumAccumulatorRefreshFeatures += numAddedFeatures + numRemovedFeatures;
#endif // NN_ACCUMULATOR_STATS

        // evict the least-recently-used way and write the freshly computed state into it
        uint32_t lruWay = 0;
        for (uint32_t w = 1; w < NumCacheWays; ++w)
            if (cache.lastUsed[w] < cache.lastUsed[lruWay])
                lruWay = w;

        nn::Accumulator::Update(
            cache.accum[lruWay],
            node.accumulatorData[color],
            cache.accum[baseWay],
            network.accumulatorWeights,
            numAddedFeatures, addedFeatures,
            numRemovedFeatures, removedFeatures);

        for (Color c = 0; c < 2; ++c)
            for (uint32_t p = 0; p < 6; ++p)
                cache.pieces[c][p][lruWay] = curr[c][p];
        cache.lastUsed[lruWay] = ++cache.clock;

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
    const NodeInfo* prevAccumNode = nullptr;
    for (const NodeInfo* nodePtr = &node; ; --nodePtr)
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

    NodeInfo* parentInfo = &node - 1;

    if (prevAccumNode == &node)
    {
        // do nothing - accumulator is already up to date (was cached)
    }
    else if (node.ply > 0 && prevAccumNode &&
        parentInfo != prevAccumNode &&
        parentInfo->nnContext.accumDirty[color])
    {
        // two-stage update:
        // if parent node has invalid accumulator, update it first
        // this way, sibling nodes can reuse parent's accumulator
        UpdateAccumulator<perspective>(network, prevAccumNode, *parentInfo, kingBucketCache);
        UpdateAccumulator<perspective>(network, parentInfo, node, kingBucketCache);
    }
    else
    {
        UpdateAccumulator<perspective>(network, prevAccumNode, node, kingBucketCache);
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

    return nnOutput;
}

void NNEvaluator::EnsureAccumulatorUpdated(const nn::PackedNeuralNetwork& network, NodeInfo& node, AccumulatorCache& cache)
{
    RefreshAccumulator<White>(network, node, cache);
    RefreshAccumulator<Black>(network, node, cache);
}
