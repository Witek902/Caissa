#include "TranspositionTable.hpp"
#include "Position.hpp"

#include <cstring>
#include <xmmintrin.h>

static_assert(sizeof(TTEntry) == sizeof(uint64_t), "Invalid TT entry size");
static_assert(sizeof(TranspositionTable::TTCluster) == CACHELINE_SIZE, "TT cluster is too big");

#if defined(_MSC_VER)

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

        std::cout << "Large page support enabled. Minimum page size: " << (GetLargePageMinimum() / 1024u) << " KB" << std::endl;
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
            //ptr = ::VirtualAlloc(NULL, roundedSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
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
    ASSERT(v >= -CheckmateValue && v <= CheckmateValue);
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
        if ((v >= TablebaseWinValue - MaxSearchDepth) && (CheckmateValue - v > 99 - fiftyMoveRuleCount))
        {
            // do not return a potentially false mate score
            return CheckmateValue - MaxSearchDepth - 1;
        }
        return ScoreType(v - height);
    }

    if (v <= -TablebaseWinValue + MaxSearchDepth) // TB loss or worse
    {
        if ((v <= TablebaseWinValue - MaxSearchDepth) && (CheckmateValue + v > 99 - fiftyMoveRuleCount))
        {
            // do not return a potentially false mate score
            return CheckmateValue - MaxSearchDepth + 1;
        }
        return ScoreType(v + height);
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

    Clear();
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

        for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
        {
            uint64_t hash;
            TTEntry data;
            cluster[i].Load(hash, data);

            if (hash == position.GetHash() && data.bounds != TTEntry::Bounds::Invalid)
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

void TranspositionTable::Write(const Position& position, ScoreType score, ScoreType staticEval, uint8_t depth, TTEntry::Bounds bounds, PackedMove move)
{
    TTEntry entry;
    entry.score = score;
    entry.staticEval = staticEval;
    entry.move = move;
    entry.depth = depth;
    entry.bounds = bounds;

    Write(position, entry);

    /*
    if (position.Blacks().pawns == 0 && position.Whites().pawns == 0 && position.GetNumPieces() <= 6)
    {
        Position positionMirrorV = position; positionMirrorV.MirrorVertically();
        Position positionMirrorH = position; positionMirrorH.MirrorHorizontally();
        Position positionMirrorVH = positionMirrorV; positionMirrorVH.MirrorHorizontally();

        Write(positionMirrorV, entry);
        Write(positionMirrorH, entry);
        Write(positionMirrorVH, entry);
    }
    */
}

void TranspositionTable::Write(const Position& position, TTEntry entry)
{
    ASSERT(entry.IsValid());

    if (!clusters)
    {
        return;
    }

    const uint64_t positionHash = position.GetHash();
    const size_t hashmapMask = numClusters - 1;

    TTCluster& cluster = clusters[positionHash & hashmapMask];

    uint32_t replaceIndex = 0;
    int32_t minDepthInCluster = INT32_MAX;
    uint64_t prevHash = 0;
    TTEntry prevEntry;

    // find target entry in the cluster (the one with lowest depth)
    for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
    {
        uint64_t hash;
        TTEntry data;
        cluster[i].Load(hash, data);

        // found entry with same hash or empty entry
        if (hash == positionHash || !data.IsValid())
        {
            replaceIndex = i;
            prevHash = hash;
            prevEntry = data;
            break;
        }

        const int32_t entryAge = (TTEntry::GenerationCycle + this->generation - data.generation) & (TTEntry::GenerationCycle - 1);

        if ((int32_t)data.depth - entryAge < minDepthInCluster)
        {
            minDepthInCluster = (int32_t)data.depth - entryAge;
            replaceIndex = i;
            prevHash = hash;
            prevEntry = data;
        }
    }

    if (positionHash == prevHash)
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
        if (!entry.move.IsValid())
        {
            entry.move = prevEntry.move;
        }
    }

    entry.generation = generation;

    cluster[replaceIndex].Store(positionHash, entry);
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
            if (entry.data.load(std::memory_order_relaxed).IsValid())
            {
                num++;
            }
        }
    }

    return num;
}