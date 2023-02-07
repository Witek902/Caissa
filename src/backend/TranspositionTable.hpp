#pragma once

#include "Move.hpp"

#include <vector>
#include <atomic>

class Position;

struct TTEntry
{
    static constexpr uint32_t NumMoves = 3;
    static constexpr uint32_t GenerationBits = 6;
    static constexpr uint32_t GenerationCycle = 1 << GenerationBits;

    enum class Bounds : uint8_t
    {
        Invalid = 0,
        Lower   = 1,
        Upper   = 2,
        Exact   = Lower|Upper,
    };

    ScoreType score;
    ScoreType staticEval;
    MovesArray<PackedMove, NumMoves> moves;
    int8_t depth;
    Bounds bounds : 2;
    uint8_t generation : GenerationBits;

    INLINE TTEntry()
        : score(0)
        , staticEval(0)
        , depth(0)
        , bounds(Bounds::Invalid)
        , generation(0)
    {}

    INLINE bool IsValid() const
    {
        return bounds != Bounds::Invalid;
    }

    INLINE uint32_t GetHash() const
    {
        const uint32_t* t = reinterpret_cast<const uint32_t*>(this);
        return t[0] ^ t[1] ^ t[2];
    }
};

class TranspositionTable
{
public:
    struct InternalEntry
    {
        uint32_t key;
        TTEntry entry;

        INLINE void Load(uint32_t& outHash, TTEntry& outEntry) const
        {
            // Xor trick by Robert Hyatt and Tim Mann
            
            // equivalent of:
            // outHash = key ^ entry.GetHash();
            const uint64_t* t = reinterpret_cast<const uint64_t*>(this);
            const uint64_t hash = t[0] ^ t[1];
            outHash = (uint32_t)hash ^ (uint32_t)(hash >> 32);
            
            outEntry = entry;
        }

        INLINE void Store(uint32_t positionKey, const TTEntry newEntry)
        {
            // Xor trick by Robert Hyatt and Tim Mann
            key = positionKey ^ newEntry.GetHash();
            entry = newEntry;
        }
    };

    // one cluster occupies one cache line
    static constexpr uint32_t NumEntriesPerCluster = 4;
    using TTCluster = InternalEntry[NumEntriesPerCluster];

    TranspositionTable(size_t initialSize = 0);
    TranspositionTable(TranspositionTable&& rhs);
    TranspositionTable& operator = (TranspositionTable&& rhs);
    ~TranspositionTable();

    // should be called before running a new search
    void NextGeneration();

    bool Read(const Position& position, TTEntry& outEntry) const;
    void Write(const Position& position, ScoreType score, ScoreType staticEval, int32_t depth, TTEntry::Bounds bounds, uint32_t numMoves = 0, const PackedMove* moves = nullptr);
    void Prefetch(const Position& position) const;

    // invalidate all entries
    void Clear();

    // resize the table
    // old entries will be preserved if possible
    void Resize(size_t newSizeInBytes);

    size_t GetSize() const { return numClusters * NumEntriesPerCluster; }

    // compute number of used entries
    size_t GetNumUsedEntries() const;

private:

    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator = (const TranspositionTable&) = delete;

    INLINE TTCluster& GetCluster(uint64_t hash) const
    {
        const uint64_t index = MulHi64(hash, numClusters);
        ASSERT(index < numClusters);
        return clusters[index];
    }

    mutable TTCluster* clusters;
    size_t numClusters;
    uint8_t generation;
};

INLINE TTEntry::Bounds operator & (const TTEntry::Bounds a, const TTEntry::Bounds b)
{
    return static_cast<TTEntry::Bounds>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

// convert from score that is relative to root to an TT score (absolute, position dependent)
ScoreType ScoreToTT(ScoreType v, int32_t height);

// convert TT score (absolute, position dependent) to search node score (relative to root)
ScoreType ScoreFromTT(ScoreType v, int32_t height, int32_t fiftyMoveRuleCount);
