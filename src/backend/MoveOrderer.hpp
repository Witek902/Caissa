#pragma once

#include "Move.hpp"

struct NodeInfo;

class MoveOrderer
{
public:

    static constexpr int32_t PVMoveValue            = INT32_MAX;
    static constexpr int32_t TTMoveValue            = PVMoveValue - 1;

    static constexpr int32_t WinningCaptureValue    = 20000000;
    static constexpr int32_t GoodCaptureValue       = 10000000;
    static constexpr int32_t KillerMoveBonus        = 1000000;
    static constexpr int32_t LosingCaptureValue     = 100000;

    static constexpr uint32_t NumKillerMoves = 2;

    using CounterType = int16_t;

    void NewSearch();
    void Clear();

    void UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth);

    void UpdateKillerMove(const NodeInfo& node, const Move move);

    // assign scores to move list
    void ScoreMoves(const NodeInfo& node, const Game& game, MoveList& moves) const;

    void DebugPrint() const;

private:

    static void UpdateHistoryCounter(CounterType& counter, int32_t delta);

    alignas(CACHELINE_SIZE)

    CounterType quietMoveHistory[2][64][64];
    CounterType quietMoveContinuationHistory[6][64][6][64];
    CounterType quietMoveFollowupHistory[6][64][6][64];

    Move killerMoves[MaxSearchDepth][NumKillerMoves];
};
