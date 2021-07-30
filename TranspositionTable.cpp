#pragma once

#include "TranspositionTable.hpp"
#include "Position.hpp"

#include <intrin.h>

TranspositionTable::TranspositionTable()
    : entries(nullptr)
    , size(0)
    , numCollisions(0)
{
}

TranspositionTable::~TranspositionTable()
{
    _aligned_free(entries);
}

void TranspositionTable::Clear()
{
    memset(entries, 0, size * sizeof(TTEntry));
}

void TranspositionTable::Resize(size_t newSize, bool preserveEntries)
{
    ASSERT(IsPowerOfTwo(newSize));

    if (size == newSize)
    {
        return;
    }

    if (newSize == 0)
    {
        _aligned_free(entries);
        entries = nullptr;
        size = 0;
        return;
    }

    TTEntry* oldEntries = entries;
    size_t oldSize = size;

    if (!preserveEntries)
    {
        _aligned_free(oldEntries);
        oldEntries = nullptr;
        oldSize = 0;
    }

    entries = (TTEntry*)_aligned_malloc(newSize * sizeof(TTEntry), CACHELINE_SIZE);
    size = newSize;
    ASSERT(entries);
    ASSERT((size_t)entries % CACHELINE_SIZE == 0);

    if (!entries)
    {
        std::cerr << "Failed to allocate transposition table" << std::endl;
        ::exit(1);
    }

    Clear();

    if (preserveEntries)
    {
        // copy old entries
        if (oldEntries)
        {
            const size_t hashmapMask = oldSize - 1;

            for (size_t i = 0; i < oldSize; ++i)
            {
                const TTEntry& oldEntry = oldEntries[i];
                if (oldEntry.IsValid())
                {
                    Write(oldEntry);
                }
            }
        }

        _aligned_free(oldEntries);
    }
}

void TranspositionTable::Prefetch(const Position& position) const
{
    if (entries)
    {
        const size_t hashmapMask = size - 1;

        const TTEntry* ttEntry = entries + (position.GetHash() & hashmapMask);
        _mm_prefetch(reinterpret_cast<const char*>(ttEntry), _MM_HINT_T0);
    }
}

const TTEntry* TranspositionTable::Read(const Position& position) const
{
    if (entries)
    {
        const size_t hashmapMask = size - 1;

        const TTEntry* ttEntry = entries + (position.GetHash() & hashmapMask);
        if (ttEntry->positionHash == position.GetHash() && ttEntry->flag != TTEntry::Flag_Invalid)
        {
            return ttEntry;
        }
    }

    return nullptr;
}

void TranspositionTable::Write(const TTEntry& entry)
{
    ASSERT(entry.IsValid());

    if (!entries)
    {
        return;
    }

    const size_t hashmapMask = size - 1;
    TTEntry& existingEntry = entries[entry.positionHash & hashmapMask];

    if (existingEntry.positionHash == entry.positionHash)
    {
        // only keep higher values computed at higher depth
        if (existingEntry.depth > entry.depth && existingEntry.flag == entry.flag)
        {
            return;
        }
    }
#ifndef CONFIGURATION_FINAL
    else if (existingEntry.positionHash != 0)
    {
        numCollisions++;
    }
#endif // CONFIGURATION_FINAL

    entries[entry.positionHash & hashmapMask] = entry;
}

size_t TranspositionTable::GetNumUsedEntries() const
{
    size_t num = 0;

    for (size_t i = 0; i < size; ++i)
    {
        const TTEntry& entry = entries[i];
        if (entry.IsValid())
        {
            num++;
        }
    }

    return num;
}