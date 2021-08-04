#pragma once

#include "Move.hpp"

#include <vector>

class Position;

struct TTEntry
{
    enum Flags : uint8_t
    {
        Flag_Invalid        = 0,
        Flag_Exact          = 1,
        Flag_LowerBound     = 2,
        Flag_UpperBound     = 3,
    };

    uint64_t hash;
    ScoreType score = InvalidValue;
    ScoreType staticEval = InvalidValue;
    PackedMove move;
    uint8_t depth = 0;
    Flags flag = Flag_Invalid;

    bool IsValid() const
    {
        return flag != TTEntry::Flag_Invalid;
    }
};

class TranspositionTable
{
public:
    // one cluster occupies one cache line
    static constexpr uint32_t NumEntriesPerCluster = 4;
    using TTCluster = TTEntry[NumEntriesPerCluster];

    TranspositionTable();
    ~TranspositionTable();

    const TTEntry* Read(const Position& position) const;
    void Write(const TTEntry& entry);
    void Prefetch(const Position& position) const;

    // invalidate all entries
    void Clear();

    // resize the table
    // old entries will be preserved if possible
    void Resize(size_t newSize, bool preserveEntries = false);

    size_t GetSize() const { return numClusters * NumEntriesPerCluster; }

    // compute number of used entries
    size_t GetNumUsedEntries() const;

    uint64_t GetNumCollisions() const { return numCollisions; }

private:

    TTCluster* clusters;
    size_t numClusters;
    uint8_t generation;

    uint64_t numCollisions;
};
