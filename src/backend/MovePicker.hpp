#pragma once

#include "MoveList.hpp"

class MoveOrderer;
struct NodeInfo;

class MovePicker
{
public:
    MovePicker(const Position& pos, const MoveOrderer& moveOrderer, const TTEntry& ttEntry, const Move pvMove, uint32_t moveGenFlags);

    void Shuffle();

    bool PickMove(const NodeInfo& node, const Game& game, Move& outMove, int32_t& outScore);

    INLINE uint32_t GetNumMoves() const { return moves.Size(); }

private:

    enum class Stage
    {
        PVMove,
        TTMove,
        Captures,
        Quiet,
        End,
    };

    const Position& position;
    const TTEntry& ttEntry;
    const Move pvMove;
    const uint32_t moveGenFlags;

    const MoveOrderer& moveOrderer;
    Stage stage = Stage::PVMove;
    uint32_t moveIndex = 0;
    bool shuffleEnabled = false;
    MoveList moves;
};