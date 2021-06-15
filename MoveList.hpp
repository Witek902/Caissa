#pragma once

#include "Move.hpp"

class MoveList
{
    friend class Position;
    friend class Search;

public:

    struct MoveEntry
    {
        Move move;
        int32_t score;
    };

    static constexpr uint32_t MaxMoves = 255;

    uint32_t Size() const { return numMoves; }
    const Move& GetMove(uint32_t index) const { ASSERT(index < numMoves); return moves[index].move; }

    const MoveEntry& operator [] (uint32_t index) const { ASSERT(index < numMoves); return moves[index]; }
    MoveEntry& operator [] (uint32_t index) { ASSERT(index < numMoves); return moves[index]; }

    void RemoveMove(const Move& move);

    Move PickBestMove(uint32_t index, int32_t& outMoveScore)
    {
        ASSERT(index < numMoves);

        int32_t bestScore = -1;
        uint32_t bestMoveIndex = index;
        for (uint32_t i = index; i < numMoves; ++i)
        {
            if (moves[i].score > bestScore)
            {
                bestScore = moves[i].score;
                bestMoveIndex = i;
            }
        }

        if (bestMoveIndex != index)
        {
            std::swap(moves[index], moves[bestMoveIndex]);
        }

        outMoveScore = moves[index].score;
        return moves[index].move;
    }

    bool HasMove(const Move move) const
    {
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i].move == move)
            {
                return true;
            }
        }

        return false;
    }

    bool HasMove(const PackedMove move) const
    {
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i].move == move)
            {
                return true;
            }
        }

        return false;
    }

    void Print(bool sorted = true) const;

private:

    void PushMove(const Move move, int32_t score)
    {
        ASSERT(numMoves < MaxMoves);
        uint32_t index = numMoves++;
        moves[index] = { move, score };
    }

    uint32_t numMoves = 0;
    MoveEntry moves[MaxMoves];
};
