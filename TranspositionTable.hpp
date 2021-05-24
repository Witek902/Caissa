#pragma once

#include "Move.hpp"

#include <vector>

class Position;

struct TranspositionTableEntry
{
    enum Flags : uint8_t
    {
        Flag_Invalid,
        Flag_Exact,
        Flag_LowerBound,
        Flag_UpperBound,
    };

    uint64_t positionHash;
    int32_t score = INT32_MIN;  // TODO 16 bits should be enough
    PackedMove move;
    uint8_t depth = 0;
    Flags flag = Flag_Invalid;

    bool IsValid() const
    {
        return flag != TranspositionTableEntry::Flag_Invalid;
    }
};

static_assert(sizeof(TranspositionTableEntry) == 16, "TT entry is too big");

class TranspositionTable
{
public:
    TranspositionTable();

    const TranspositionTableEntry* Read(const Position& position) const;
    void Write(const TranspositionTableEntry& entry);
    void Prefetch(const Position& position) const;

    // invalidate all entries
    void Clear();

    // resize the table
    // old entries will be preserved if possible
    void Resize(size_t newSize);

    // compute number of used entries
    size_t GetNumUsedEntries() const;

private:

    std::vector<TranspositionTableEntry> entries;
};
