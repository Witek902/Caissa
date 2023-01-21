#pragma once

#include "Move.hpp"

struct NodeInfo;

template<uint32_t Size>
struct KillerMoves
{
    Move moves[Size];

    void Clear()
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
    static constexpr int32_t LosingCaptureValue     = -4000;

    static constexpr uint32_t NumKillerMoves        = 2;

    using CounterType = int16_t;

    MoveOrderer();

    void NewSearch();
    void Clear();

    void UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth);
    void UpdateCapturesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth);

    void ClearKillerMoves(uint32_t depth);
    void UpdateKillerMove(const NodeInfo& node, const Move move);

    // assign scores to move list
    void ScoreMoves(const NodeInfo& node, const Game& game, MoveList& moves, bool withQuiets = true) const;

    void DebugPrint() const;

private:

    alignas(CACHELINE_SIZE)

    CounterType quietMoveHistory[2][64][64];                // side, from-square, to-square
    CounterType quietMoveContinuationHistory[6][64][6][64];
    CounterType quietMoveFollowupHistory[6][64][6][64];

    CounterType capturesHistory[2][6][5][64];               // side, capturing piece, captured piece, to-square

    KillerMoves<NumKillerMoves> killerMoves[MaxNumPieces+1][MaxSearchDepth];
};
