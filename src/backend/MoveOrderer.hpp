#pragma once

#include "Move.hpp"

struct NodeInfo;
struct NodeCacheEntry;

template<uint32_t Size>
struct KillerMoves
{
    Move moves[Size];

    INLINE void Clear()
    {
        for (uint32_t i = 0; i < Size; ++i)
        {
            moves[i] = Move::Invalid();
        }
    }

    void Push(const Move move)
    {
        for (uint32_t i = 0; i < Size; ++i)
        {
            if (move == moves[i])
            {
                // move to the front
                for (uint32_t j = i; j > 0; j--)
                {
                    moves[j] = moves[j - 1];
                }

                moves[0] = move;
                return;
            }
        }

        for (uint32_t j = Size; j-- > 1u; )
        {
            moves[j] = moves[j - 1];
        }
        moves[0] = move;
    }

    int32_t Find(const Move move) const
    {
        for (uint32_t i = 0; i < Size; ++i)
        {
            if (move == moves[i])
            {
                return (int32_t)i;
            }
        }
        return -1;
    }
};

class MoveOrderer
{
public:

    static constexpr int32_t PVMoveValue            = INT32_MAX;
    static constexpr int32_t TTMoveValue            = PVMoveValue - 1;

    static constexpr int32_t WinningCaptureValue    = 20000000;
    static constexpr int32_t GoodCaptureValue       = 10000000;
    static constexpr int32_t PromotionValue         = 5000000;
    static constexpr int32_t KillerMoveBonus        = 1000000;
    static constexpr int32_t CounterMoveBonus       = 900000;
    static constexpr int32_t LosingCaptureValue     = -4000;

    static constexpr uint32_t NumKillerMoves        = 2;

    using CounterType = int16_t;
    using PieceSquareHistory = CounterType[6][64];
    using PieceSquareHistoryPtr = PieceSquareHistory*;
    using ContinuationHistory = PieceSquareHistory[2][6][64];

    MoveOrderer();

    void NewSearch();
    void Clear();

    void InitContinuationHistoryPointers(NodeInfo& node);

    INLINE CounterType GetHistoryScore(const Color color, const Move move) const
    {
        ASSERT(move.IsValid());
        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();
        ASSERT(from < 64);
        ASSERT(to < 64);
        return quietMoveHistory[(uint32_t)color][from][to];
    }

    INLINE const KillerMoves<NumKillerMoves>& GetKillerMoves(uint32_t treeHeight) const
    {
        return killerMoves[treeHeight];
    }

    INLINE PackedMove GetCounterMove(const Color color, const Move prevMove) const
    {
        if (prevMove.IsValid())
        {
            const uint32_t from = (uint32_t)prevMove.FromSquare().Index();
            const uint32_t to = prevMove.ToSquare().Index();
            ASSERT(from < 64 && to < 64);
            return counterMoves[(uint32_t)color][from][to];
        }
        else
        {
            return Move::Invalid();
        }
    }

    void UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove);
    void UpdateCapturesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove);

    INLINE void ClearKillerMoves(uint32_t depth)
    {
        ASSERT(depth < MaxSearchDepth);
        killerMoves[depth].Clear();
    }

    INLINE void UpdateKillerMove(uint32_t depth, const Move move)
    {
        ASSERT(depth < MaxSearchDepth);
        killerMoves[depth].Push(move);
    }

    // assign scores to move list
    void ScoreMoves(
        const NodeInfo& node,
        const Game& game,
        MoveList& moves,
        bool withQuiets = true,
        const NodeCacheEntry* nodeCacheEntry = nullptr) const;

    void DebugPrint() const;

private:

    alignas(CACHELINE_SIZE)

    CounterType quietMoveHistory[2][64][64];                // side, from-square, to-square
    ContinuationHistory counterMoveHistory;
    ContinuationHistory continuationHistory;
    CounterType capturesHistory[2][6][5][64];               // side, capturing piece, captured piece, to-square
    PackedMove counterMoves[2][64][64];                     // side, from-square, to-square

    KillerMoves<NumKillerMoves> killerMoves[MaxSearchDepth];
};
