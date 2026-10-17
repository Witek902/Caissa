#pragma once

#include "Memory.hpp"

namespace numa {

// initialize NUMA subsystem, must be called before any other function in this namespace
void Init();

// returns the number of NUMA nodes available on the system
uint32_t GetNumNodes();

// pins the current thread to a specific NUMA node, returns true on success
bool PinCurrentThreadToNumaNode(uint32_t node);

// allocates memory on a specific NUMA node
void* AllocateOnNode(size_t size, uint32_t node);

// frees memory allocated on a specific NUMA node
void FreeOnNode(void* ptr, size_t size);

// helper class that allocates given type T on every NUMA node,
// so that each node can access its own copy of T without contention
template <typename T>
class PerNodeAllocation
{
public:
    PerNodeAllocation()
        : m_numNodes(GetNumNodes())
        , m_nodeData(new T*[m_numNodes])
    {
        for (uint32_t i = 0; i < m_numNodes; ++i)
        {
            void* ptr = AllocateOnNode(sizeof(T), i);
            m_nodeData[i] = PLACEMENT_NEW(ptr) T();
        }
    }

    PerNodeAllocation(const PerNodeAllocation&) = delete;
    PerNodeAllocation& operator = (const PerNodeAllocation&) = delete;

    ~PerNodeAllocation()
    {
        for (uint32_t i = 0; i < m_numNodes; ++i)
        {
            if (T* data = m_nodeData[i])
            {
                data->~T();
                FreeOnNode(data, sizeof(T));
            }
        }
        delete[] m_nodeData;
    }

    // access the copy of T allocated on the specified NUMA node
    [[nodiscard]] T* Get(uint32_t node) const
    {
        ASSERT(node < m_numNodes);
        return m_nodeData[node];
    }

private:
    uint32_t m_numNodes;
    T** m_nodeData;
};

} // namespace numa