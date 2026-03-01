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
    void* ptr = ::VirtualAllocExNuma(GetCurrentProcess(), nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, node);
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

namespace numa {

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
        return AlignedMalloc(size, CACHELINE_SIZE);
    return numa_alloc_onnode(size, (int)node);
}

void FreeOnNode(void* ptr, size_t size)
{
    if (!g_numaAvailable)
    {
        AlignedFree(ptr);
        return;
    }
    numa_free(ptr, size);
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
    return AlignedMalloc(size, CACHELINE_SIZE);
}

void FreeOnNode(void* ptr, size_t size)
{
    UNUSED(size);
    AlignedFree(ptr);
}

} // namespace numa

#endif

