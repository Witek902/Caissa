#include "Numa.hpp"
#include "Memory.hpp"

#if defined(PLATFORM_WINDOWS)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include <Windows.h>

namespace numa {

void Init()
{
    const uint32_t numNodes = GetNumNodes();
    std::cout << "info string " << numNodes << " NUMA nodes detected (WinAPI)" << std::endl;
}

uint32_t GetNumNodes()
{
    ULONG highestNodeNumber = 0;
    if (FALSE == ::GetNumaHighestNodeNumber(&highestNodeNumber))
    {
        // fallback to 1 node if NUMA is not supported or an error occurs
        return 1;
    }
    return static_cast<uint32_t>(highestNodeNumber + 1);
}

bool PinCurrentThreadToNumaNode(uint32_t node)
{
    GROUP_AFFINITY nodeGroupAffinity{};
    if (!GetNumaNodeProcessorMaskEx((USHORT)node, &nodeGroupAffinity) || nodeGroupAffinity.Mask == 0)
    {
        return false;
    }

    GROUP_AFFINITY prev{};
    if (!SetThreadGroupAffinity(GetCurrentThread(), &nodeGroupAffinity, &prev))
    {
        return false;
    }

    return true;
}

void* AllocateOnNode(size_t size, uint32_t node)
{
    void* ptr = nullptr;

    // try NUMA-local large pages first
    const size_t largePageMin = ::GetLargePageMinimum(); // 0 if large pages are unsupported
    if (largePageMin != 0)
    {
        const size_t roundedSize = ((size + largePageMin - 1) / largePageMin) * largePageMin;
        ptr = ::VirtualAllocExNuma(GetCurrentProcess(), nullptr, roundedSize,
                                   MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE, node);
    }

    // fallback: regular pages on the node
    if (!ptr)
        ptr = ::VirtualAllocExNuma(GetCurrentProcess(), nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, node);

    // final fallback: any node
    if (!ptr)
        ptr = ::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    return ptr;
}

void FreeOnNode(void* ptr, size_t size)
{
    UNUSED(size);
    ::VirtualFreeEx(GetCurrentProcess(), ptr, 0, MEM_RELEASE);
}

} // namespace numa


#elif defined(USE_LIBNUMA)

#include <numa.h> // libnuma

#if defined(__linux__)
#include <sys/mman.h> // madvise, MADV_HUGEPAGE
#endif

namespace numa {

constexpr size_t HugePageSize = 2 * 1024 * 1024;

static INLINE size_t RoundUpToHugePage(size_t size)
{
    return (size + HugePageSize - 1) & ~(HugePageSize - 1);
}

static bool g_numaAvailable = false;
static std::vector<cpu_set_t> s_sets;

void Init()
{
    if (numa_available() < 0)
    {
        g_numaAvailable = false;
        return;
    }

    g_numaAvailable = true;

    const int maxNode = numa_max_node();
    std::cout << "info string " << (maxNode + 1) << " NUMA nodes detected (libnuma)" << std::endl;

    s_sets.clear();
    s_sets.resize(maxNode + 1);
    for (int node = 0; node <= maxNode; ++node)
    {
        bitmask* bm = numa_allocate_cpumask();
        if (numa_node_to_cpus(node, bm) != 0)
            std::terminate();

        cpu_set_t set;
        CPU_ZERO(&set);

        for (unsigned cpu = 0; cpu < bm->size; ++cpu)
            if (numa_bitmask_isbitset(bm, cpu))
                CPU_SET(cpu, &set);

        numa_free_cpumask(bm);
        s_sets[node] = set;
    }
}

uint32_t GetNumNodes()
{
    if (!g_numaAvailable)
        return 1;
    return (uint32_t)(numa_max_node() + 1);
}

bool PinCurrentThreadToNumaNode(uint32_t node)
{
    if (!g_numaAvailable)
        return false;

    if (node >= s_sets.size())
        return false;

    const cpu_set_t& set = s_sets[node];
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    return rc == 0;
}

void* AllocateOnNode(size_t size, uint32_t node)
{
    if (!g_numaAvailable)
        return Malloc(size); // already madvises MADV_HUGEPAGE on Linux

    const size_t rounded = RoundUpToHugePage(size);
    void* ptr = numa_alloc_onnode(rounded, (int)node);
#if defined(MADV_HUGEPAGE)
    if (ptr)
        madvise(ptr, rounded, MADV_HUGEPAGE);
#endif
    return ptr;
}

void FreeOnNode(void* ptr, size_t size)
{
    if (!g_numaAvailable)
    {
        Free(ptr);
        return;
    }
    numa_free(ptr, RoundUpToHugePage(size));
}

} // namespace numa


#else

namespace numa {

void Init() { }

uint32_t GetNumNodes()
{
    return 1;
}

bool PinCurrentThreadToNumaNode(uint32_t node)
{
    UNUSED(node);
    return true;
}

void* AllocateOnNode(size_t size, uint32_t node)
{
    UNUSED(node);
    return Malloc(size); // already madvises MADV_HUGEPAGE on Linux
}

void FreeOnNode(void* ptr, size_t size)
{
    UNUSED(size);
    Free(ptr);
}

} // namespace numa

#endif

