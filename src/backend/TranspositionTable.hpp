#pragma once

#include "Move.hpp"

#include <vector>
#include <atomic>

class Position;

union TTEntry
{
    static constexpr uint32_t GenerationBits = 6;
    static constexpr uint32_t GenerationCycle = 1 << GenerationBits;

    enum class Bounds : uint8_t
    {
        Invalid        = 0,
        Exact          = 1,
        LowerBound     = 2,
        UpperBound     = 3,
    };

    UNNAMED_STRUCT struct
    {
        ScoreType score;
        ScoreType staticEval;
        PackedMove move;
        int8_t depth;
        Bounds bounds : 2;
        uint8_t generation : GenerationBits;
    };

    uint64_t packed;

    INLINE TTEntry() : packed(0) { }

    INLINE bool IsValid() const
    {
        return bounds != Bounds::Invalid;
    }
};

class TranspositionTable
{
public:
    struct InternalEntry
    {
        std::atomic<uint64_t> key;
        std::atomic<TTEntry> data;

        INLINE void Load(uint64_t& outHash, TTEntry& outData) const
        {
            const uint64_t k = key.load();
            const TTEntry d = data.load();
            // Xor trick by Robert Hyatt and Tim Mann
            outHash = key ^ d.packed;
            outData = d;
        }

        INLINE void Store(uint64_t hash, TTEntry newData)
        {
            // Xor trick by Robert Hyatt and Tim Mann
            key = hash ^ newData.packed;
            data = newData;
        }
    };

    // one cluster occupies one cache line
    static constexpr uint32_t NumEntriesPerCluster = 4;
    using TTCluster = InternalEntry[NumEntriesPerCluster];

    TranspositionTable(size_t initialSize = 0);
    ~TranspositionTable();

    // should be called before running a new search
    void NextGeneration();

    bool Read(const Position& position, TTEntry& outEntry) const;
    void Write(const Position& position, ScoreType score, ScoreType staticEval, int32_t depth, TTEntry::Bounds bounds, PackedMove move = PackedMove());
    void Write(const Position& position, TTEntry entry);
    void Prefetch(const Position& position) const;

    // invalidate all entries
    void Clear();

    // resize the table
    // old entries will be preserved if possible
    void Resize(size_t newSizeInBytes);

    size_t GetSize() const { return numClusters * NumEntriesPerCluster; }

    // compute number of used entries
    size_t GetNumUsedEntries() const;

    uint64_t GetNumCollisions() const { return numCollisions; }

    static void Init();

private:

    mutable TTCluster* clusters;
    size_t numClusters;

    uint8_t generation;

    uint64_t numCollisions;
};

// convert from score that is relative to root to an TT score (absolute, position dependent)
ScoreType ScoreToTT(ScoreType v, int32_t height);

// convert TT score (absolute, position dependent) to search node score (relative to root)
ScoreType ScoreFromTT(ScoreType v, int32_t height, int32_t fiftyMoveRuleCount);
