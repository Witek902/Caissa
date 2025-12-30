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
    INLINE const Move GetMove(uint32_t index) const { ASSERT(index < numMoves); return moves[index]; }
    INLINE int32_t GetScore(uint32_t index) const { ASSERT(index < numMoves); return scores[index]; }

    template<typename MoveType>
    INLINE void RemoveMove(const MoveType move)
    {
        static_assert(std::is_same_v<MoveType, Move> || std::is_same_v<MoveType, PackedMove>, "Invalid move type");

        if (!move.IsValid()) return;

        for (uint32_t i = 0; i < numMoves; ++i)
        {
            if (moves[i] == move)
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
            ASSERT(move != moves[i]);
        }

        uint32_t index = numMoves++;
        moves[index] = move;
        scores[index] = INT32_MIN;
    }

    INLINE void RemoveByIndex(uint32_t index)
    {
        ASSERT(index < numMoves);
        --numMoves;
        moves[index] = moves[numMoves];
        scores[index] = scores[numMoves];
    }

    INLINE uint32_t BestMoveIndex() const
    {
        ASSERT(numMoves > 0);

        uint32_t i = 0;

#ifdef USE_SSE2
        __m128i bestScores = _mm_set1_epi32(INT32_MIN);
        __m128i bestIndices = _mm_setzero_si128();
        __m128i indices = _mm_setr_epi32(0, 1, 2, 3);
        const __m128i increment = _mm_set1_epi32(4);

        // process 4 moves at a time
        for (; i + 4 <= numMoves; i += 4)
        {
            __m128i s = _mm_loadu_si128((const __m128i*)(scores + i));
            __m128i m = _mm_cmpgt_epi32(s, bestScores);
            bestScores = _mm_max_epi32(bestScores, s);
            bestIndices = _mm_blendv_epi8(bestIndices, indices, m);
            indices = _mm_add_epi32(indices, increment);
        }

        int32_t sLane[4];
        uint32_t iLane[4];
        _mm_storeu_si128((__m128i*)sLane, bestScores);
        _mm_storeu_si128((__m128i*)iLane, bestIndices);

        // extract best score from SIMD lanes
        int32_t bestScore = sLane[0];
        uint32_t bestMoveIndex = iLane[0];
        for (uint32_t k = 1; k < 4; ++k)
        {
            if (sLane[k] > bestScore)
            {
                bestScore = sLane[k];
                bestMoveIndex = iLane[k];
            }
        }
#endif // USE_SSE2

        // pick best from remaining
        for (; i < numMoves; ++i)
        {
            if (scores[i] > bestScore)
            {
                bestScore = scores[i];
                bestMoveIndex = i;
            }
        }

        return bestMoveIndex;
    }

    template <typename MoveType>
    bool HasMove(const MoveType move) const
    {
        for (uint32_t i = 0; i < numMoves; ++i)
            if (moves[i] == move)
                return true;
        return false;
    }

    void Sort()
    {
        std::sort(&moves[0], &moves[numMoves], [this](const Move& a, const Move& b)
        {
            const int32_t scoreA = this->scores[&a - &this->moves[0]];
            const int32_t scoreB = this->scores[&b - &this->moves[0]];
            return scoreA > scoreB;
        });
    }

private:
    uint32_t numMoves = 0;
    Move moves[MaxMoves];
    alignas(32) int32_t scores[MaxMoves];
};

void PrintMoveList(const Position& pos, const MoveList& moves);
