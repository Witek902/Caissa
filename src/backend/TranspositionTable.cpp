#include "TranspositionTable.hpp"
#include "Position.hpp"
#include "Memory.hpp"

#include <algorithm>
#include <cstring>
#include <thread>

#ifdef USE_SSE
    #include <xmmintrin.h>
#endif // USE_SSE

static_assert(sizeof(TTEntry) == 10, "Invalid TT entry size");
static_assert(sizeof(TranspositionTable::TTCluster) == 32, "Invalid TT cluster size");

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
    , generation(rhs.generation)
{
    rhs.clusters = nullptr;
    rhs.numClusters = 0;
    rhs.generation = 0;
}

TranspositionTable& TranspositionTable::operator = (TranspositionTable&& rhs)
{
    if (&rhs != this)
    {
        Free(clusters);

        clusters = rhs.clusters;
        numClusters = rhs.numClusters;
        generation = rhs.generation;

        rhs.clusters = nullptr;
        rhs.numClusters = 0;
        rhs.generation = 0;
    }

    return *this;
}

void TranspositionTable::Clear()
{
    const size_t numThreads = std::min(std::thread::hardware_concurrency(), 4u);

    if (numClusters * sizeof(TTCluster) <= 256 * 1024 * 1024 || numThreads == 1)
    {
        std::fill(clusters, clusters + numClusters, TTCluster{});
    }
    else // clear using multiple threads
    {
        const size_t numClustersPerThread = numClusters / numThreads;

        std::vector<std::thread> threads;
        threads.reserve(numThreads);

        for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex)
        {
            threads.emplace_back([this, threadIndex, numClustersPerThread]()
            {
                const size_t start = threadIndex * numClustersPerThread;
                const size_t end = start + numClustersPerThread;
                std::fill(clusters + start, clusters + end, TTCluster{});
            });
        }

        for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex)
        {
            threads[threadIndex].join();
        }
    }

    generation = 0;
}

void TranspositionTable::Resize(size_t newSizeInBytes)
{
    const size_t newNumClusters = newSizeInBytes / sizeof(TTCluster);
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
    ASSERT(clusters);
    ASSERT((size_t)clusters % CACHELINE_SIZE == 0);

    if (!clusters)
    {
        numClusters = 0;
        std::cerr << "Failed to allocate transposition table" << std::endl;
    }
}

void TranspositionTable::NextGeneration()
{
    generation++;
}

void TranspositionTable::Prefetch(const uint64_t hash) const
{
#ifdef USE_SSE
    _mm_prefetch(reinterpret_cast<const char*>(&GetCluster(hash)), _MM_HINT_T0);
#elif defined(USE_ARM_NEON)
    __builtin_prefetch(reinterpret_cast<const char*>(&GetCluster(hash)), 0, 0);
#else
    (void)hash;
#endif // USE_SSE
}

bool TranspositionTable::Read(const Position& position, TTEntry*& outEntry) const
{
    ASSERT(clusters);

    TTCluster& cluster = GetCluster(position.GetHash());
    const uint16_t posKey = (uint16_t)position.GetHash();

    for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
    {
        TTEntry& entry = cluster.entries[i];
        if (entry.key == posKey && entry.bounds != TTEntry::Bounds::Invalid)
        {
            outEntry = &entry;
            return true;
        }
    }

    // find the entry with the lowest relevance
    uint32_t replaceIndex = 0;
    int32_t minRelevanceInCluster = INT32_MAX;
    for (uint32_t i = 1; i < NumEntriesPerCluster; ++i)
    {
        TTEntry& entry = cluster.entries[i];

        const int32_t entryAge = (TTEntry::GenerationCycle + this->generation - entry.generation) & (TTEntry::GenerationCycle - 1);
        const int32_t entryRelevance = (int32_t)entry.depth - entryAge;

        if (entryRelevance < minRelevanceInCluster)
        {
            minRelevanceInCluster = entryRelevance;
            replaceIndex = i;
        }
    }

    outEntry = cluster.entries + replaceIndex;
    return false;
}

void TranspositionTable::Write(TTEntry* entry, const Position& position, ScoreType score, ScoreType staticEval, int32_t depth, TTEntry::Bounds bounds, PackedMove move)
{
    ASSERT(clusters);
    ASSERT(entry);
    ASSERT(position.GetHash() == position.ComputeHash());
    ASSERT(depth >= 0);

    const uint16_t positionKey = (uint16_t)position.GetHash();

    // don't overwrite entries with worse depth if the bounds are not exact
    if (bounds != TTEntry::Bounds::Exact &&
        positionKey == entry->key &&
        depth < entry->depth - 4)
    {
        return;
    }

    entry->score = score;
    entry->staticEval = staticEval;
    if (move.IsValid()) entry->move = move; // preserve existing move
    entry->depth = static_cast<int8_t>(depth);
    entry->bounds = bounds;
    entry->generation = generation;
    entry->key = positionKey;
}

void TranspositionTable::PrintInfo() const
{
    size_t totalCount = 0;
    size_t exactCount = 0;
    size_t lowerBoundCount = 0;
    size_t upperBoundCount = 0;

    for (size_t i = 0; i < numClusters; ++i)
    {
        const TTCluster& cluster = clusters[i];
        for (size_t j = 0; j < NumEntriesPerCluster; ++j)
        {
            const TTEntry& entry = cluster.entries[j];
            if (entry.IsValid())
            {
                totalCount++;
                if (entry.bounds == TTEntry::Bounds::Exact) exactCount++;
                if (entry.bounds == TTEntry::Bounds::Lower) lowerBoundCount++;
                if (entry.bounds == TTEntry::Bounds::Upper) upperBoundCount++;
            }
        }
    }

    std::cout << "=== TT statistics ===" << std::endl;
    std::cout << "Entries in use:      " << totalCount << " (" << (100.0f * (float)totalCount / (float)GetSize()) << "%)" << std::endl;
    std::cout << "Exact entries:       " << exactCount << " (" << (100.0f * (float)exactCount / (float)totalCount) << "%)" << std::endl;
    std::cout << "Lower-bound entries: " << lowerBoundCount << " (" << (100.0f * (float)lowerBoundCount / (float)totalCount) << "%)" << std::endl;
    std::cout << "Upper-bound entries: " << upperBoundCount << " (" << (100.0f * (float)upperBoundCount / (float)totalCount) << "%)" << std::endl;
}

uint32_t TranspositionTable::GetHashFull() const
{
    const uint32_t clusterCount = 1000 / NumEntriesPerCluster;

    uint32_t count = 0;
    if (clusterCount <= numClusters)
    {
        for (uint32_t i = 0; i < clusterCount; ++i)
        {
            for (const TTEntry& entry : clusters[i].entries)
            {
                count += (entry.IsValid() && entry.generation == generation);
            }
        }
    }
    return count;
}