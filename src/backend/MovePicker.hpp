#pragma once

#include "MoveList.hpp"

class MoveOrderer;
struct NodeInfo;

class MovePicker
{
public:
    MovePicker(const Position& pos, const MoveOrderer& moveOrderer, const TTEntry& ttEntry, uint32_t moveGenFlags = 0);

    void Shuffle();

    bool IsOnlyOneLegalMove();

    bool PickMove(const NodeInfo& node, Move& outMove, int32_t& outScore);

private:
    const Position& position;
    const MoveOrderer& moveOrderer;
    uint32_t moveGenFlags;
    uint32_t numScoredMoves = 0;
    uint32_t moveIndex = 0;
    bool shuffleEnabled = false;
    MoveList moves;
};