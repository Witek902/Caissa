#pragma once

#include "MoveList.hpp"
#include "Position.hpp"

class MoveOrderer;
struct NodeInfo;
struct NodeCacheEntry;

class MovePicker
{
public:

    enum class Stage : uint8_t
    {
        TTMove = 0,
        GenerateCaptures,
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
               const PackedMove ttMove,
               bool generateQuiets)
        : m_position(pos)
        , m_nodeCacheEntry(nodeCacheEntry)
        , m_ttMove(ttMove)
        , m_generateQuiets(generateQuiets)
        , m_moveOrderer(moveOrderer)
    {
    }

    bool PickMove(const NodeInfo& node, Move& outMove, int32_t& outScore);

    INLINE Stage GetStage() const { return m_stage; }
    INLINE uint32_t GetNumMoves() const { return m_moves.Size(); }
    INLINE void SkipQuiets() { m_generateQuiets = false; }

private:

    const Position& m_position;
    const NodeCacheEntry* m_nodeCacheEntry;
    const PackedMove m_ttMove;
    bool m_generateQuiets;

    const MoveOrderer& m_moveOrderer;
    uint32_t m_moveIndex = 0;
    Stage m_stage = Stage::TTMove;

    Move m_killerMoves[2] = { Move::Invalid(), Move::Invalid() };

    MoveList m_moves;
};