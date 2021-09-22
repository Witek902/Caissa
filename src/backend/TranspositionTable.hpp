#pragma once

#include "Move.hpp"

#include <vector>
#include <atomic>

class Position;

union TTEntry
{
    enum Flags : uint8_t
    {
        Flag_Invalid        = 0,
        Flag_Exact          = 1,
        Flag_LowerBound     = 2,
        Flag_UpperBound     = 3,
    };

    struct
    {
        ScoreType score;
        ScoreType staticEval;
        PackedMove move;
        uint8_t depth;
        Flags flag;
    };

    uint64_t packed;

    INLINE TTEntry() : packed(0) { }

    bool IsValid() const
    {
        return flag != TTEntry::Flag_Invalid;
    }
};

class TranspositionTable
{
public:
    struct InternalEntry
    {
        std::atomic<uint64_t> key;
        std::atomic<TTEntry> data;
    };

    // one cluster occupies one cache line
    static constexpr uint32_t NumEntriesPerCluster = 4;
    using TTCluster = InternalEntry[NumEntriesPerCluster];

    TranspositionTable(size_t initialSize = 0);
    ~TranspositionTable();

    bool Read(const Position& position, TTEntry& outEntry) const;
    void Write(const Position& position, const TTEntry& entry);
    void Prefetch(const Position& position) const;

    // invalidate all entries
    void Clear();

    // resize the table
    // old entries will be preserved if possible
    void Resize(size_t newSize);

    size_t GetSize() const { return numClusters * NumEntriesPerCluster; }

    // compute number of used entries
    size_t GetNumUsedEntries() const;

    uint64_t GetNumCollisions() const { return numCollisions; }

private:

    TTCluster* clusters;
    size_t numClusters;

    uint64_t numCollisions;
};
