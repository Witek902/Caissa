#include "TranspositionTable.hpp"
#include "Position.hpp"
#include "Memory.hpp"

#include <algorithm>
#include <cstring>

#ifdef USE_SSE
    #include <xmmintrin.h>
#endif // USE_SSE

static_assert(sizeof(TTEntry) == 3 * sizeof(uint32_t), "Invalid TT entry size");
static_assert(sizeof(TranspositionTable::TTCluster) == CACHELINE_SIZE, "Invalid TT cluster size");

ScoreType ScoreToTT(ScoreType v, int32_t height)
{
    ASSERT(v > -CheckmateValue && v < CheckmateValue);
    ASSERT(height < MaxSearchDepth);

    return ScoreType(
        v >= ( TablebaseWinValue - MaxSearchDepth) ? v + height :
        v <= (-TablebaseWinValue + MaxSearchDepth) ? v - height :
        v);
}

ScoreType ScoreFromTT(ScoreType v, int32_t height, int32_t fiftyMoveRuleCount)
{
    ASSERT(height < MaxSearchDepth);

    // based on Stockfish

    if (v >= TablebaseWinValue - MaxSearchDepth)  // TB win or better
    {
        if ((v >= CheckmateValue - MaxSearchDepth) && (CheckmateValue - v > 99 - fiftyMoveRuleCount))
        {
            // do not return a potentially false mate score
            return CheckmateValue - MaxSearchDepth - 1;
        }
        return std::min<ScoreType>(ScoreType(v - height), CheckmateValue - 1);
    }

    if (v <= -TablebaseWinValue + MaxSearchDepth) // TB loss or worse
    {
        if ((v <= -CheckmateValue + MaxSearchDepth) && (CheckmateValue + v > 99 - fiftyMoveRuleCount))
        {
            // do not return a potentially false mate score
            return -CheckmateValue + MaxSearchDepth + 1;
        }
        return std::max<ScoreType>(ScoreType(v + height), -CheckmateValue + 1);
    }

    return v;
}


TranspositionTable::TranspositionTable(size_t initialSize)
    : clusters(nullptr)
    , numClusters(0)
    , hashMask(0)
    , generation(0)
{
    Resize(initialSize);
}

TranspositionTable::~TranspositionTable()
{
    Free(clusters);
}

TranspositionTable::TranspositionTable(TranspositionTable&& rhs)
	: clusters(rhs.clusters)
	, numClusters(rhs.numClusters)
	, hashMask(rhs.hashMask)
	, generation(rhs.generation)
{
	rhs.clusters = nullptr;
	rhs.numClusters = 0;
	rhs.hashMask = 0;
	rhs.generation = 0;
}

TranspositionTable& TranspositionTable::operator = (TranspositionTable&& rhs)
{
    if (&rhs != this)
    {
        Free(clusters);

        clusters = rhs.clusters;
        numClusters = rhs.numClusters;
        hashMask = rhs.hashMask;
        generation = rhs.generation;

        rhs.clusters = nullptr;
        rhs.numClusters = 0;
        rhs.hashMask = 0;
        rhs.generation = 0;
    }

    return *this;
}

void TranspositionTable::Clear()
{
    memset(clusters, 0, numClusters * sizeof(TTCluster));
    generation = 0;
}

static uint64_t NextPowerOfTwo(uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;

}

void TranspositionTable::Resize(size_t newSizeInBytes)
{
    const size_t newNumClusters = NextPowerOfTwo(newSizeInBytes) / sizeof(TTCluster);
    const size_t newSize = newNumClusters / NumEntriesPerCluster;

    if (numClusters == newNumClusters)
    {
        return;
    }

    if (newSize == 0)
    {
        Free(clusters);
        clusters = nullptr;
        numClusters = 0;
        return;
    }

    Free(clusters);
    clusters = nullptr;

    clusters = (TTCluster*)Malloc(newNumClusters * sizeof(TTCluster));
    numClusters = newNumClusters;
    hashMask = numClusters - 1;
    ASSERT(clusters);
    ASSERT((size_t)clusters % CACHELINE_SIZE == 0);

    if (!clusters)
    {
        std::cerr << "Failed to allocate transposition table" << std::endl;
        ::exit(1);
    }
}

void TranspositionTable::NextGeneration()
{
    generation++;
}

void TranspositionTable::Prefetch(const Position& position) const
{
#ifdef USE_SSE
    if (clusters)
    {
        const TTCluster* cluster = clusters + (position.GetHash() & hashMask);
        _mm_prefetch(reinterpret_cast<const char*>(cluster), _MM_HINT_T0);
    }
#else
    (void)position;
#endif // USE_SSE
}

bool TranspositionTable::Read(const Position& position, TTEntry& outEntry) const
{
    if (clusters)
    {
        TTCluster& cluster = clusters[position.GetHash() & hashMask];

        const uint32_t posKey = (position.GetHash() >> 32);

        for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
        {
            uint32_t hash;
            TTEntry data;
            cluster[i].Load(hash, data);

            if (hash == posKey && data.bounds != TTEntry::Bounds::Invalid)
            {
                if (data.generation != generation)
                {
                    // update entry generation
                    data.generation = generation;
                    cluster[i].Store(hash, data);
                }

                outEntry = data;
                return true;
            }
        }
    }

    return false;
}

void TranspositionTable::Write(const Position& position, ScoreType score, ScoreType staticEval, int32_t depth, TTEntry::Bounds bounds, uint32_t numMoves, const PackedMove* moves)
{
    ASSERT(position.GetHash() == position.ComputeHash());

    TTEntry entry;
    entry.score = score;
    entry.staticEval = staticEval;
    entry.depth = (int8_t)std::clamp<int32_t>(depth, INT8_MIN, INT8_MAX);
    entry.bounds = bounds;

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        entry.moves[i] = moves[i];
    }

    ASSERT(entry.IsValid());

    if (!clusters)
    {
        return;
    }

    const uint64_t positionHash = position.GetHash();
    const uint32_t positionKey = (uint32_t)(positionHash >> 32);

    TTCluster& cluster = clusters[positionHash & hashMask];

    uint32_t replaceIndex = 0;
    int32_t minRelevanceInCluster = INT32_MAX;
    uint32_t prevKey = 0;
    TTEntry prevEntry;

    // find target entry in the cluster (the one with lowest depth)
    for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
    {
        uint32_t hash;
        TTEntry data;
        cluster[i].Load(hash, data);

        // found entry with same hash or empty entry
        if (hash == positionKey || !data.IsValid())
        {
            replaceIndex = i;
            prevKey = hash;
            prevEntry = data;
            break;
        }

        // old entriess are less relevant
        const int32_t entryAge = (TTEntry::GenerationCycle + this->generation - data.generation) & (TTEntry::GenerationCycle - 1);
        const int32_t entryRelevance = (int32_t)data.depth - 8 * entryAge + 4 * (data.bounds == TTEntry::Bounds::Exact);

        if (entryRelevance < minRelevanceInCluster)
        {
            minRelevanceInCluster = entryRelevance;
            replaceIndex = i;
            prevKey = hash;
            prevEntry = data;
        }
    }

    // don't overwrite entries with worse depth if the bounds are not exact
    if (entry.bounds != TTEntry::Bounds::Exact &&
        positionKey == prevKey &&
        entry.depth < prevEntry.depth - 3)
    {
        return;
    }

    // preserve existing move
    if (positionKey == prevKey && !entry.moves[0].IsValid())
    {
        entry.moves = prevEntry.moves;
    }

    entry.generation = generation;

    cluster[replaceIndex].Store(positionKey, entry);
}

size_t TranspositionTable::GetNumUsedEntries() const
{
    size_t num = 0;

    for (size_t i = 0; i < numClusters; ++i)
    {
        const TTCluster& cluster = clusters[i];
        for (size_t j = 0; j < NumEntriesPerCluster; ++j)
        {
            const InternalEntry& entry = cluster[j];
            if (entry.entry.IsValid())
            {
                num++;
            }
        }
    }

    return num;
}