#include "TranspositionTable.hpp"
#include "Position.hpp"

#include <cstring>
#include <xmmintrin.h>

static_assert(sizeof(TTEntry) == sizeof(uint64_t), "Invalid TT entry size");
static_assert(sizeof(TranspositionTable::TTCluster) == CACHELINE_SIZE, "TT cluster is too big");

#if defined(_MSC_VER)

    static void* Malloc(size_t size, size_t alignment)
    {
        return _aligned_malloc(size, alignment);
    }

    static void Free(void* ptr)
    {
        _aligned_free(ptr);
    }

#elif defined(__GNUC__) || defined(__clang__)

    static void* Malloc(size_t size, size_t alignment)
    {
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, alignment, size);
        return ret != 0 ? nullptr : ptr;
    }

    static void Free(void* ptr)
    {
        free(ptr);
    }

#endif

TranspositionTable::TranspositionTable(size_t initialSize)
    : clusters(nullptr)
    , numClusters(0)
    , numCollisions(0)
{
    Resize(initialSize);
}

TranspositionTable::~TranspositionTable()
{
    Free(clusters);
}

void TranspositionTable::Clear()
{
    memset(clusters, 0, numClusters * sizeof(TTCluster));
}

void TranspositionTable::Resize(size_t newSize)
{
    ASSERT(IsPowerOfTwo(newSize));

    const size_t newNumClusters = (newSize + NumEntriesPerCluster - 1) / NumEntriesPerCluster;

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

    clusters = (TTCluster*)Malloc(newNumClusters * sizeof(TTCluster), CACHELINE_SIZE);
    numClusters = newNumClusters;
    ASSERT(clusters);
    ASSERT((size_t)clusters % CACHELINE_SIZE == 0);

    if (!clusters)
    {
        std::cerr << "Failed to allocate transposition table" << std::endl;
        ::exit(1);
    }

    Clear();
}

void TranspositionTable::Prefetch(const Position& position) const
{
    if (clusters)
    {
        const size_t hashmapMask = numClusters - 1;

        const TTCluster* cluster = clusters + (position.GetHash() & hashmapMask);
        _mm_prefetch(reinterpret_cast<const char*>(cluster), _MM_HINT_T0);
    }
}

bool TranspositionTable::Read(const Position& position, TTEntry& outEntry) const
{
    if (clusters)
    {
        const size_t hashmapMask = numClusters - 1;

        const TTCluster& cluster = clusters[position.GetHash() & hashmapMask];

        for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
        {
            const InternalEntry& entry = cluster[i];

            const uint64_t key = entry.key;
            const TTEntry data = entry.data;

            // Xor trick by Robert Hyatt and Tim Mann
            if (((key ^ data.packed) == position.GetHash()) && data.flag != TTEntry::Flag_Invalid)
            {
                outEntry = data;
                return true;
            }
        }
    }

    return false;
}

void TranspositionTable::Write(const Position& position, const TTEntry& entry)
{
    ASSERT(entry.IsValid());

    if (!clusters)
    {
        return;
    }

    const uint64_t hash = position.GetHash();
    const size_t hashmapMask = numClusters - 1;

    TTCluster& cluster = clusters[hash & hashmapMask];

    // find target entry in the cluster
    uint32_t targetIndex = 0;
    uint32_t minDepthInCluster = MaxSearchDepth;
    for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
    {
        InternalEntry& existingEntry = cluster[i];

        const uint64_t key = existingEntry.key;
        const TTEntry data = existingEntry.data;

        // found entry with same hash and bouds type
        if (((key ^ data.packed) == hash) && data.flag == entry.flag)
        {
            // if there's already an entry with higher depth, don't overwrite it
            if (data.depth > entry.depth)
            {
                return;
            }

            targetIndex = i;
            break;
        }

        // if no entry with same hash is found, use one with lowest value
        if (data.depth < minDepthInCluster)
        {
            minDepthInCluster = data.depth;
            targetIndex = i;
        }
    }

    InternalEntry& targetEntry = cluster[targetIndex];

    // Xor trick by Robert Hyatt and Tim Mann
    targetEntry.key = hash ^ entry.packed;
    targetEntry.data = entry;

    //// preserve existing move
    //if (entry.move.IsValid() || targetEntry.hash != entry.hash)
    //{
    //    targetEntry.move = entry.move;
    //}
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
            if (entry.data.load().IsValid())
            {
                num++;
            }
        }
    }

    return num;
}