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
        GenerateCaptures,
        Captures,
        Killer1,
        Killer2,
        Counter,
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
        : m_position(pos)
        , m_ttEntry(ttEntry)
        , m_nodeCacheEntry(nodeCacheEntry)
        , m_pvMove(pvMove)
        , m_moveGenFlags(moveGenFlags)
        , m_moveOrderer(moveOrderer)
    {
    }

    bool PickMove(const NodeInfo& node, const Game& game, Move& outMove, int32_t& outScore);

    INLINE Stage GetStage() const { return m_stage; }
    INLINE uint32_t GetNumMoves() const { return m_moves.Size(); }

private:

    const Position& m_position;
    const TTEntry& m_ttEntry;
    const NodeCacheEntry* m_nodeCacheEntry;
    const Move m_pvMove;
    const uint32_t m_moveGenFlags;

    const MoveOrderer& m_moveOrderer;
    uint32_t m_moveIndex = 0;
    Stage m_stage = Stage::PVMove;

    Move m_counterMove = Move::Invalid();
    Move m_killerMoves[2] = { Move::Invalid(), Move::Invalid() };

    MoveList m_moves;
};