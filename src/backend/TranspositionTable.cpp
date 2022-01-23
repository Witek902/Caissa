#include "TranspositionTable.hpp"
#include "Position.hpp"

#include <algorithm>
#include <cstring>
#include <xmmintrin.h>

static_assert(sizeof(TranspositionTable::TTCluster) == CACHELINE_SIZE, "TT cluster is too big");

#if defined(_MSC_VER)

#define NOMINMAX
#include <Windows.h>

    static bool EnableLargePagesSupport()
    {
        HANDLE hToken;
        TOKEN_PRIVILEGES tp;

        // open process token
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        {
            std::cerr << "OpenProcessToken failed, error code: " << GetLastError() << std::endl;
            return false;
        }

        // get the luid
        if (!LookupPrivilegeValue(NULL, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid))
        {
            std::cerr << "LookupPrivilegeValue failed, error code: " << GetLastError() << std::endl;
            return false;
        }

        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

        // enable or disable privilege
        BOOL status = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);

        // It is possible for AdjustTokenPrivileges to return TRUE and still not succeed.
        // So always check for the last error value.
        DWORD error = GetLastError();
        if (!status || (error != ERROR_SUCCESS))
        {
            std::cerr << "AdjustTokenPrivileges failed, error code: " << error << std::endl;
            return false;
        }

        CloseHandle(hToken);

#ifndef CONFIGURATION_FINAL
        std::cout << "Large page support enabled. Minimum page size: " << (GetLargePageMinimum() / 1024u) << " KB" << std::endl;
#endif // CONFIGURATION_FINAL

        return true;
    }

    NO_INLINE static void* Malloc(size_t size)
    {
        void* ptr = nullptr;

        // try large pages first
        const size_t largePageMinNumpages = 4;
        const size_t minLargePageSize = largePageMinNumpages * ::GetLargePageMinimum();
        if (size >= minLargePageSize)
        {
            const size_t roundedSize = ((size + minLargePageSize - 1) / minLargePageSize) * minLargePageSize;
            ptr = ::VirtualAlloc(NULL, roundedSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        }

        // fallback to regular pages
        if (!ptr)
        {
            ptr = ::VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        }

        return ptr;
    }

    static void Free(void* ptr)
    {
        ::VirtualFree(ptr, 0, MEM_RELEASE);
    }

#elif defined(__GNUC__) || defined(__clang__)

    static bool EnableLargePagesSupport()
    {
        return false;
    }

    static void* Malloc(size_t size)
    {
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, CACHELINE_SIZE, size);
        return ret != 0 ? nullptr : ptr;
    }

    static void Free(void* ptr)
    {
        free(ptr);
    }

#endif


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

void TranspositionTable::Init()
{
    EnableLargePagesSupport();
}

TranspositionTable::TranspositionTable(size_t initialSize)
    : clusters(nullptr)
    , numClusters(0)
    , generation(0)
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

        TTCluster& cluster = clusters[position.GetHash() & hashmapMask];

        const uint32_t posKey = (position.GetHash() >> 32);

        for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
        {
            uint32_t hash;
            TTEntry data;
            cluster[i].Load(hash, data);

            if (hash == posKey && data.bounds != TTEntry::Bounds::Invalid)
            {
                // update entry generation
                data.generation = generation;
                cluster[i].Store(hash, data);

                outEntry = data;
                return true;
            }
        }
    }

    return false;
}

void TranspositionTable::Write(const Position& position, ScoreType score, ScoreType staticEval, int32_t depth, TTEntry::Bounds bounds, uint32_t numMoves, const PackedMove* moves)
{
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
    const size_t hashmapMask = numClusters - 1;
    const uint32_t positionKey = (uint32_t)(positionHash >> 32);

    TTCluster& cluster = clusters[positionHash & hashmapMask];

    uint32_t replaceIndex = 0;
    int32_t minDepthInCluster = INT32_MAX;
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

        const int32_t entryAge = (TTEntry::GenerationCycle + this->generation - data.generation) & (TTEntry::GenerationCycle - 1);

        if ((int32_t)data.depth - entryAge < minDepthInCluster)
        {
            minDepthInCluster = (int32_t)data.depth - entryAge;
            replaceIndex = i;
            prevKey = hash;
            prevEntry = data;
        }
    }

    if (positionKey == prevKey)
    {
        // don't overwrite entries with worse depth when preserving bounds
        if (entry.depth < prevEntry.depth &&
            entry.bounds == prevEntry.bounds)
        {
            return;
        }

        // don't demote Exact to Lower/UpperBound if overwriting entry with same depth
        if (entry.depth == prevEntry.depth &&
            prevEntry.bounds == TTEntry::Bounds::Exact &&
            entry.bounds != TTEntry::Bounds::Exact)
        {
            return;
        }

        // preserve existing move
        if (!entry.moves[0].IsValid())
        {
            entry.moves[0] = prevEntry.moves[0];
        }
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