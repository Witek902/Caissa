#pragma once

#include "MoveList.hpp"

class MoveOrderer;
struct NodeInfo;
struct NodeCacheEntry;

class MovePicker
{
public:

    enum class Stage : uint8_t
    {
        PVMove,
        TTMove,
        Captures,
        Killer1,
        Killer2,
        GenerateQuiets,
        PickQuiets,
        End,
    };

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

    bool PickMove(const NodeInfo& node, const Game& game, Move& outMove, int32_t& outScore);

    INLINE Stage GetStage() const { return stage; }
    INLINE uint32_t GetNumMoves() const { return moves.Size(); }

private:

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