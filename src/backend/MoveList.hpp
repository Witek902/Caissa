#pragma once

#include "Move.hpp"

#include <algorithm>

class MoveList
{
    friend class Search;
    friend class MovePicker;
    friend class MoveOrderer;

public:

    static constexpr uint32_t MaxMoves = 240;

    INLINE uint32_t Size() const { return numMoves; }
    INLINE const Move GetMove(uint32_t index) const { ASSERT(index < numMoves); return moves[index]; }
    INLINE int32_t GetScore(uint32_t index) const { ASSERT(index < numMoves); return scores[index]; }

    template<typename MoveType>
    void RemoveMove(const MoveType move)
    {
        static_assert(std::is_same_v<MoveType, Move> || std::is_same_v<MoveType, PackedMove>, "Invalid move type");

        if (!move.IsValid()) return;

        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i] == move)
            {
                std::swap(moves[i], moves[numMoves - 1]);
                std::swap(scores[i], scores[numMoves - 1]);
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
            ASSERT(move != moves[i]);
        }

        uint32_t index = numMoves++;
        moves[index] = move;
        scores[index] = INT32_MIN;
    }

    uint32_t AssignTTScores(const TTEntry& ttEntry);

    void RemoveByIndex(uint32_t index)
    {
        ASSERT(index < numMoves);
        std::swap(moves[numMoves - 1], moves[index]);
        std::swap(scores[numMoves - 1], scores[index]);
        numMoves--;
    }

    uint32_t BestMoveIndex() const
    {
        int32_t bestScore = INT32_MIN;
        uint32_t bestMoveIndex = UINT32_MAX;

        for (uint32_t j = 0; j < numMoves; ++j)
        {
            const int32_t score = scores[j];
            if (score > bestScore)
            {
                bestScore = score;
                bestMoveIndex = j;
            }
        }

        return bestMoveIndex;
    }

    bool HasMove(const Move move) const
    {
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i] == move)
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
            if (moves[i] == move)
            {
                return true;
            }
        }

        return false;
    }

    void Sort();

    void Print(const Position& pos) const;

private:

    uint32_t numMoves = 0;
    Move moves[MaxMoves];
    alignas(32) int32_t scores[MaxMoves];
};
