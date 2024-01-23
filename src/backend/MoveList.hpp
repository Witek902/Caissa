#pragma once

#include "Move.hpp"

#include <algorithm>

template<uint32_t MaxSize>
class TMoveList
{
    friend class Search;
    friend class MovePicker;
    friend class MoveOrderer;
    friend void PrintMoveList(const Position& pos, const MoveList& moves);

public:

    static constexpr uint32_t MaxMoves = MaxSize;

    INLINE uint32_t Size() const { return numMoves; }
    INLINE const Move GetMove(uint32_t index) const { ASSERT(index < numMoves); return entries[index].move; }
    INLINE int32_t GetScore(uint32_t index) const { ASSERT(index < numMoves); return entries[index].score; }

    template<typename MoveType>
    INLINE void RemoveMove(const MoveType move)
    {
        static_assert(std::is_same_v<MoveType, Move> || std::is_same_v<MoveType, PackedMove>, "Invalid move type");

        if (!move.IsValid()) return;

        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (entries[i].move == move)
            {
                RemoveByIndex(i);
                return;
            }
        }
    }

    INLINE void Clear()
    {
        numMoves = 0;
    }

    INLINE void Push(const Move move)
    {
        ASSERT(numMoves < MaxMoves);

        // check for duplicate moves
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            ASSERT(move != entries[i].move);
        }

        uint32_t index = numMoves++;
        entries[index].move = move;
        entries[index].score = INT32_MIN;
    }

    template<uint32_t MaxSize2>
    INLINE void Push(const TMoveList<MaxSize2>& other)
    {
        ASSERT(numMoves + other.numMoves <= MaxMoves);

        memcpy(entries + numMoves, other.entries, other.numMoves * sizeof(Entry));
        numMoves += other.numMoves;
    }

    INLINE void RemoveByIndex(uint32_t index)
    {
        ASSERT(index < numMoves);
        entries[index] = entries[--numMoves];
    }

    INLINE uint32_t BestMoveIndex() const
    {
        int32_t bestScore = INT32_MIN;
        uint32_t bestMoveIndex = UINT32_MAX;

        for (uint32_t j = 0; j < numMoves; ++j)
        {
            const int32_t score = entries[j].score;
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
            if (entries[i].move == move)
                return true;
        return false;
    }

    bool HasMove(const PackedMove move) const
    {
        for (uint32_t i = 0; i < numMoves; ++i)
            if (entries[i].move == move)
                return true;
        return false;
    }

    void Sort()
    {
        std::sort(entries, entries + numMoves, [](const Entry& a, const Entry& b) { return a.score > b.score; });
    }

private:

    struct Entry
    {
        Move move;
        int32_t score;
    };

    uint32_t numMoves = 0;
    Entry entries[MaxMoves];
};

void PrintMoveList(const Position& pos, const MoveList& moves);
