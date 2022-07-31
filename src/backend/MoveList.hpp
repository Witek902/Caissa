#pragma once

#include "Move.hpp"

#include <algorithm>

class MoveList
{
    friend class Position;
    friend class Search;
    friend class MovePicker;

public:

    struct MoveEntry
    {
        Move move;
        int32_t score;
    };

    static constexpr uint32_t MaxMoves = 255;

    INLINE uint32_t Size() const { return numMoves; }
    INLINE const Move& GetMove(uint32_t index) const { ASSERT(index < numMoves); return moves[index].move; }

    INLINE const MoveEntry& operator [] (uint32_t index) const { ASSERT(index < numMoves); return moves[index]; }
    INLINE MoveEntry& operator [] (uint32_t index) { ASSERT(index < numMoves); return moves[index]; }

    template<typename MoveType>
    void RemoveMove(const MoveType move)
    {
        static_assert(std::is_same_v<MoveType, Move> || std::is_same_v<MoveType, PackedMove>, "Invalid move type");

        if (!move.IsValid()) return;

        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i].move == move)
            {
                std::swap(moves[i], moves[numMoves - 1]);
                numMoves--;
                i--;
            }
        }
    }

    void Clear()
    {
        numMoves = 0;
    }

    void Push(const Move move)
    {
        ASSERT(numMoves < MaxMoves);
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            ASSERT(move != moves[i].move);
        }

        uint32_t index = numMoves++;
        moves[index] = { move, INT32_MIN };
    }

    uint32_t AssignTTScores(const TTEntry& ttEntry);

    void RemoveByIndex(uint32_t index)
    {
        ASSERT(index < numMoves);
        std::swap(moves[numMoves - 1], moves[index]);
        numMoves--;
    }

    uint32_t BestMoveIndex() const
    {
        int32_t bestScore = INT32_MIN;
        uint32_t bestMoveIndex = UINT32_MAX;
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i].score > bestScore)
            {
                bestScore = moves[i].score;
                bestMoveIndex = i;
            }
        }
        return bestMoveIndex;
    }

    const Move PickBestMove(uint32_t index, int32_t& outMoveScore)
    {
        ASSERT(index < numMoves);

        int32_t bestScore = INT32_MIN;
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

    void Sort()
    {
        std::stable_sort(moves, moves + numMoves, [](const MoveEntry& a, const MoveEntry& b)
        {
            return a.score > b.score;
        });
    }

    void Shuffle();

    void Print(const Position& pos) const;

private:

    uint32_t numMoves = 0;
    MoveEntry moves[MaxMoves];
};
