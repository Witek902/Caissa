#pragma once

#include "Common.hpp"

#include <vector>


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
    {
        const uint32_t numNodes = GetNumNodes();
        m_nodeData.resize(numNodes);
        for (uint32_t i = 0; i < numNodes; ++i)
        {
            void* ptr = AllocateOnNode(sizeof(T), i);
            m_nodeData[i] = new (ptr) T();
        }
    }

    ~PerNodeAllocation()
    {
        for (T* data : m_nodeData)
        {
            if (data)
            {
                data->~T();
                FreeOnNode(data, sizeof(T));
            }
        }
    }

    // access the copy of T allocated on the specified NUMA node
    [[nodiscard]] T* Get(uint32_t node) const
    {
        ASSERT(node < m_nodeData.size());
        return m_nodeData[node];
    }

private:
    std::vector<T*> m_nodeData;
};


} // namespace numa