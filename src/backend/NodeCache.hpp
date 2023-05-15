#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include <vector>
#include <atomic>

struct NodeCacheEntry
{
    struct MoveInfo
    {
        Move move = Move::Invalid();
        uint64_t nodesSearched = 0;
    };

    static constexpr uint32_t MaxMoves = 32;

    uint32_t generation = 0;
    uint32_t distanceFromRoot = 0;
    uint64_t nodesSum = 0;

    Position position;
    MoveInfo moves[MaxMoves];

    const MoveInfo* GetMove(const Move move) const;

    void ClearMoves();
    void ScaleDown();
    void AddMoveStats(const Move& move, uint64_t numNodes);
    void PrintMoves() const;
};

class NodeCache
{
public:

    void Reset();
    void OnNewSearch();

    const NodeCacheEntry* TryGetEntry(const Position& pos) const;
    NodeCacheEntry* GetEntry(const Position& pos, uint32_t distanceFromRoot);

private:

    static constexpr uint32_t Size = 256;

    uint32_t generation = 0;

    NodeCacheEntry entries[Size];
};
