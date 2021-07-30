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

    uint64_t positionHash;
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

using AtomicTTEntry = std::atomic<TTEntry>;

static_assert(sizeof(TTEntry) == 16, "TT entry is too big");

class TranspositionTable
{
public:
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

    size_t GetSize() const { return size; }

    // compute number of used entries
    size_t GetNumUsedEntries() const;

    uint64_t GetNumCollisions() const { return numCollisions; }

private:

    TTEntry* entries;
    size_t size;

    uint64_t numCollisions;
};
