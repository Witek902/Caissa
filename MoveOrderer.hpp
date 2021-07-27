#pragma once

#include "Move.hpp"

struct NodeInfo;

class MoveOrderer
{
public:

    void Clear();

    void OnBetaCutoff(const NodeInfo& node, const Move move);

    void OrderMoves(const NodeInfo& node, MoveList& moves) const;

    void DebugPrint() const;

private:
    uint32_t searchHistory[2][64][64];

    static constexpr uint32_t NumKillerMoves = 4;
    PackedMove killerMoves[MaxSearchDepth][NumKillerMoves];

    PackedMove counterMoveHistory[2][64][64]; // TODO piece type?
};
