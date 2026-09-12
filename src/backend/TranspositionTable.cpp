#include "TranspositionTable.hpp"
#include "Position.hpp"
#include "Memory.hpp"

#include <algorithm>
#include <thread>

uint32_t TranspositionTable::NumInitThreads = 1;

static_assert(sizeof(TTEntry) == 2 * sizeof(uint32_t), "Invalid TT entry size");
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
    const uint32_t maxThreads = std::max(1u, std::thread::hardware_concurrency() / 2u);
    const size_t numThreads = std::clamp(NumInitThreads, 1u, maxThreads);

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
        return;
    }

    Clear();
}

void TranspositionTable::NextGeneration()
{
    generation = (uint8_t)((generation + 1) & (TTEntry::GenerationCycle - 1));
}

void TranspositionTable::Prefetch(const uint64_t hash) const
{
    ::Prefetch(&GetCluster(hash));
}

bool TranspositionTable::Read(const Position& position, TTEntry& outEntry) const
{
    if (clusters)
    {
        TTCluster& cluster = GetCluster(position.GetHash());

        const uint16_t posKey = (uint16_t)position.GetHash();

        for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
        {
            const uint16_t key = cluster.entries[i].key;
            const TTEntry data = cluster.entries[i].entry;

            if (key == posKey && data.bounds != TTEntry::Bounds::Invalid)
            {
                outEntry = data;
                return true;
            }
        }
    }

    return false;
}

void TranspositionTable::Write(const Position& position, ScoreType score, ScoreType staticEval, uint32_t depth, TTEntry::Bounds bounds, PackedMove move)
{
    ASSERT(position.GetHash() == position.ComputeHash());

    if (!clusters)
    {
        return;
    }

    const uint64_t positionHash = position.GetHash();
    const uint16_t positionKey = (uint16_t)positionHash;

    TTCluster& cluster = GetCluster(position.GetHash());

    uint32_t replaceIndex = 0;
    int32_t minRelevanceInCluster = INT32_MAX;

    // find target entry in the cluster (the one with lowest depth)
    for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
    {
        const uint16_t key = cluster.entries[i].key;
        const TTEntry data = cluster.entries[i].entry;

        // found entry with same hash or empty entry
        if (key == positionKey || !data.IsValid())
        {
            replaceIndex = i;
            break;
        }

        // old entries are less relevant
        const int32_t entryAge = (TTEntry::GenerationCycle + this->generation - data.generation) & (TTEntry::GenerationCycle - 1);
        const int32_t entryRelevance = (int32_t)data.depth - entryAge;

        if (entryRelevance < minRelevanceInCluster)
        {
            minRelevanceInCluster = entryRelevance;
            replaceIndex = i;
        }
    }

    InternalEntry& entry = cluster.entries[replaceIndex];
    const bool sameKey = positionKey == entry.key;
    const uint8_t newDepth = (uint8_t)std::min<uint32_t>(depth, UINT8_MAX);

    // don't overwrite entries with worse depth if the bounds are not exact, unless the entry is from an older search
    if (bounds != TTEntry::Bounds::Exact &&
        sameKey &&
        (int32_t)newDepth < (int32_t)entry.entry.depth - 4 &&
        entry.entry.generation == generation)
    {
        if (!entry.entry.move.IsValid() && move.IsValid())
        {
            // but update move if needed
            entry.entry.move = move;
        }

        return;
    }

    // preserve existing move if the new one is invalid
    if (move.IsValid() || !sameKey)
    {
        entry.entry.move = move;
    }

    entry.entry.score = score;
    entry.entry.staticEval = staticEval;
    entry.entry.depth = newDepth;
    entry.entry.bounds = bounds;
    entry.entry.generation = generation;
    ASSERT(entry.entry.IsValid());

    entry.key = positionKey;
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
            const InternalEntry& entry = cluster.entries[j];
            if (entry.entry.IsValid())
            {
                totalCount++;

                if (entry.entry.bounds == TTEntry::Bounds::Exact) exactCount++;
                if (entry.entry.bounds == TTEntry::Bounds::Lower) lowerBoundCount++;
                if (entry.entry.bounds == TTEntry::Bounds::Upper) upperBoundCount++;
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
            for (const InternalEntry& entry : clusters[i].entries)
            {
                count += (entry.entry.IsValid() && entry.entry.generation == generation);
            }
        }
    }
    return count;
}