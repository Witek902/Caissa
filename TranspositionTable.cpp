#pragma once

#include "TranspositionTable.hpp"
#include "Position.hpp"

#include <intrin.h>

TranspositionTable::TranspositionTable()
{

}

void TranspositionTable::Resize(size_t newSize)
{
    ASSERT(IsPowerOfTwo(newSize));

    if (entries.size() != newSize)
    {
        std::vector<TranspositionTableEntry> oldEntries = std::move(entries);
        entries.resize(newSize);

        // copy old entries
        if (!oldEntries.empty())
        {
            const size_t hashmapMask = oldEntries.size() - 1;

            for (const TranspositionTableEntry& oldEntry : oldEntries)
            {
                if (oldEntry.IsValid())
                {
                    Write(oldEntry);
                }
            }
        }
    }
}

void TranspositionTable::Prefetch(const Position& position) const
{
    if (!entries.empty())
    {
        const size_t hashmapMask = entries.size() - 1;

        const TranspositionTableEntry& ttEntry = entries[position.GetHash() & hashmapMask];
        _mm_prefetch(reinterpret_cast<const char*>(&ttEntry), _MM_HINT_T0);
    }
}

const TranspositionTableEntry* TranspositionTable::Read(const Position& position) const
{
    if (!entries.empty())
    {
        const size_t hashmapMask = entries.size() - 1;

        const TranspositionTableEntry& ttEntry = entries[position.GetHash() & hashmapMask];
        if (ttEntry.positionHash == position.GetHash() && ttEntry.flag != TranspositionTableEntry::Flag_Invalid)
        {
            return &ttEntry;
        }
    }

    return nullptr;
}

void TranspositionTable::Write(const TranspositionTableEntry& entry)
{
    ASSERT(entry.IsValid());

    if (!entries.empty())
    {
        const size_t hashmapMask = entries.size() - 1;

        entries[entry.positionHash & hashmapMask] = entry;
    }
}

size_t TranspositionTable::GetNumUsedEntries() const
{
    size_t num = 0;

    for (const TranspositionTableEntry& entry : entries)
    {
        if (entry.IsValid())
        {
            num++;
        }
    }

    return num;
}