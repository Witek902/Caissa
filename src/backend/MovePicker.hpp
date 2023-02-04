#pragma once

#include "MoveList.hpp"

class MoveOrderer;
struct NodeInfo;
struct NodeCacheEntry;

class MovePicker
{
public:
    MovePicker(const Position& pos,
               const MoveOrderer& moveOrderer,
               const NodeCacheEntry* nodeCacheEntry,
               const TTEntry& ttEntry,
               const Move pvMove,
               uint32_t moveGenFlags)
        : position(pos)
        , ttEntry(ttEntry)
        , nodeCacheEntry(nodeCacheEntry)
        , pvMove(pvMove)
        , moveGenFlags(moveGenFlags)
        , moveOrderer(moveOrderer)
    {
    }

    void Shuffle();

    bool PickMove(const NodeInfo& node, const Game& game, Move& outMove, int32_t& outScore);

    INLINE uint32_t GetNumMoves() const { return moves.Size(); }

private:

    enum class Stage : uint8_t
    {
        PVMove,
        TTMove,
        Captures,
        Quiet,
        End,
    };

    const Position& position;
    const TTEntry& ttEntry;
    const NodeCacheEntry* nodeCacheEntry;
    const Move pvMove;
    const uint32_t moveGenFlags;

    const MoveOrderer& moveOrderer;
    uint32_t moveIndex = 0;
    Stage stage = Stage::PVMove;
    bool shuffleEnabled = false;
    MoveList moves;
};