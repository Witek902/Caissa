#pragma once

#include "TranspositionTable.hpp"
#include "Position.hpp"

#include <intrin.h>

static_assert(sizeof(TranspositionTable::TTCluster) == CACHELINE_SIZE, "TT cluster is too big");

TranspositionTable::TranspositionTable()
    : clusters(nullptr)
    , numClusters(0)
    , generation(0)
    , numCollisions(0)
{
}

TranspositionTable::~TranspositionTable()
{
    _aligned_free(clusters);
}

void TranspositionTable::Clear()
{
    memset(clusters, 0, numClusters * sizeof(TTCluster));
}

void TranspositionTable::Resize(size_t newSize, bool preserveEntries)
{
    ASSERT(IsPowerOfTwo(newSize));

    const size_t newNumClusters = (newSize + NumEntriesPerCluster - 1) / NumEntriesPerCluster;

    if (numClusters == newNumClusters)
    {
        return;
    }

    if (newSize == 0)
    {
        _aligned_free(clusters);
        clusters = nullptr;
        numClusters = 0;
        return;
    }

    TTCluster* oldClusters = clusters;
    size_t oldNumClusters = numClusters;

    if (!preserveEntries)
    {
        _aligned_free(oldClusters);
        oldClusters = nullptr;
        oldNumClusters = 0;
    }

    clusters = (TTCluster*)_aligned_malloc(newNumClusters * sizeof(TTCluster), CACHELINE_SIZE);
    numClusters = newNumClusters;
    ASSERT(clusters);
    ASSERT((size_t)clusters % CACHELINE_SIZE == 0);

    if (!clusters)
    {
        std::cerr << "Failed to allocate transposition table" << std::endl;
        ::exit(1);
    }

    Clear();

    if (preserveEntries)
    {
        // copy old entries
        if (oldClusters)
        {
            const size_t hashmapMask = oldNumClusters - 1;

            for (size_t i = 0; i < oldNumClusters; ++i)
            {
                const TTCluster& cluster = oldClusters[i];
                for (size_t j = 0; j < NumEntriesPerCluster; ++j)
                {
                    const TTEntry& oldEntry = cluster[j];
                    if (oldEntry.IsValid())
                    {
                        Write(oldEntry);
                    }
                }
            }
        }

        _aligned_free(oldClusters);
    }
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

const TTEntry* TranspositionTable::Read(const Position& position) const
{
    if (clusters)
    {
        const size_t hashmapMask = numClusters - 1;

        const TTCluster& cluster = clusters[position.GetHash() & hashmapMask];

        for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
        {
            const TTEntry& entry = cluster[i];
            if (entry.hash == position.GetHash() && entry.flag != TTEntry::Flag_Invalid)
            {
                return &entry;
            }
        }
    }

    return nullptr;
}

void TranspositionTable::Write(const TTEntry& entry)
{
    ASSERT(entry.IsValid());

    if (!clusters)
    {
        return;
    }

    const size_t hashmapMask = numClusters - 1;

    TTCluster& cluster = clusters[entry.hash & hashmapMask];

    // find target entry in the cluster
    uint32_t targetIndex = 0;
    uint32_t minDepthInCluster = MaxSearchDepth;
    for (uint32_t i = 0; i < NumEntriesPerCluster; ++i)
    {
        TTEntry& existingEntry = cluster[i];

        // found entry with same hash and bouds type
        if (existingEntry.hash == entry.hash && existingEntry.flag == entry.flag)
        {
            // if there's already an entry with higher depth, don't overwrite it
            if (existingEntry.depth > entry.depth)
            {
                return;
            }

            targetIndex = i;
            break;
        }

        // if no entry with same hash is found, use one with lowest value
        if (existingEntry.depth < minDepthInCluster)
        {
            minDepthInCluster = existingEntry.depth;
            targetIndex = i;
        }
    }

    TTEntry& targetEntry = cluster[targetIndex];

    targetEntry.hash = entry.hash;
    targetEntry.score = entry.score;
    targetEntry.staticEval = entry.staticEval;
    targetEntry.depth = entry.depth;
    targetEntry.flag = entry.flag;

    // preserve existing move
    if (entry.move.IsValid() || targetEntry.hash != entry.hash)
    {
        targetEntry.move = entry.move;
    }
}

size_t TranspositionTable::GetNumUsedEntries() const
{
    size_t num = 0;

    for (size_t i = 0; i < numClusters; ++i)
    {
        const TTCluster& cluster = clusters[i];
        for (size_t j = 0; j < NumEntriesPerCluster; ++j)
        {
            const TTEntry& entry = cluster[j];
            if (entry.IsValid())
            {
                num++;
            }
        }
    }

    return num;
}