#pragma once

#include "Move.hpp"

struct NodeInfo;
struct NodeCacheEntry;


class MoveOrderer
{
public:

    static constexpr int32_t PVMoveValue            = INT32_MAX;
    static constexpr int32_t TTMoveValue            = PVMoveValue - 1;

    static constexpr int32_t WinningCaptureValue    = 20000000;
    static constexpr int32_t GoodCaptureValue       = 10000000;
    static constexpr int32_t PromotionValue         = 5000000;
    static constexpr int32_t PromotionValues[]      = { 0, 0, -30000000, -40000000, -40000000, PromotionValue, 0 };
    static constexpr int32_t KillerMoveBonus        = 1000000;
    static constexpr int32_t LosingCaptureValue     = -4000;

    using CounterType = int16_t;
    using PieceSquareHistory = CounterType[6][64];
    using PieceSquareHistoryPtr = PieceSquareHistory*;

    MoveOrderer();

    void NewSearch();
    void Clear();

    void InitContinuationHistoryPointers(NodeInfo& node);

    CounterType GetHistoryScore(const NodeInfo& node, const Move move) const;

    INLINE const Move GetKillerMove(uint32_t treeHeight) const
    {
        return killerMoves[treeHeight];
    }

    void UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove);
    void UpdateCapturesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove);

    INLINE void ClearKillerMoves(uint32_t depth)
    {
        ASSERT(depth <= MaxSearchDepth);
        killerMoves[depth] = Move::Invalid();
    }

    INLINE void UpdateKillerMove(uint32_t depth, const Move move)
    {
        ASSERT(depth < MaxSearchDepth);
        killerMoves[depth] = move;
    }

    // assign scores to move list
    void ScoreMoves(
        const NodeInfo& node,
        MoveList& moves,
        bool withQuiets = true,
        const NodeCacheEntry* nodeCacheEntry = nullptr) const;

    void DebugPrint() const;

private:

    alignas(CACHELINE_SIZE)

    CounterType quietMoveHistory[2][2][2][64][64];          // stm, from-threated, to-threated, from-square, to-square
    PieceSquareHistory continuationHistory[2][2][2][6][64]; // prev is capture, prev stm, current stm, piece, to-square
    CounterType capturesHistory[2][6][5][64];               // stm, capturing piece, captured piece, to-square

    Move killerMoves[MaxSearchDepth + 1];
};
