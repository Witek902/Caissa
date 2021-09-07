#pragma once

#include "Move.hpp"

struct NodeInfo;

class MoveOrderer
{
public:

    static constexpr int32_t PVMoveValue = INT32_MAX;
    static constexpr int32_t TTMoveValue = PVMoveValue - 1;

    void Clear();

    void OnBetaCutoff(const NodeInfo& node, const Move move);

    // assign scores to move list
    void ScoreMoves(const NodeInfo& node, MoveList& moves) const;

    void DebugPrint() const;

private:
    uint32_t searchHistory[2][64][64];

    static constexpr uint32_t NumKillerMoves = 4;
    PackedMove killerMoves[MaxSearchDepth][NumKillerMoves];

    PackedMove counterMoveHistory[2][64][64]; // TODO piece type?
};
