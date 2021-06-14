#pragma once

#include "Square.hpp"
#include "Piece.hpp"

#include <iostream>

// "C++ nonstandard extension: nameless struct"
#pragma warning(disable : 4201)

struct Move;

struct PackedMove
{
    union
    {
        struct
        {
            Square fromSquare;
            Square toSquare;
        };

        uint16_t value;
    };

    INLINE PackedMove() : value(0u) { }
    INLINE PackedMove(const PackedMove&) = default;
    INLINE PackedMove& operator = (const PackedMove&) = default;

    PackedMove(const Move& rhs);

    // valid move does not mean it's a legal move for a given position
    // use Position::IsMoveLegal() to fully validate a move
    bool IsValid() const
    {
        return value != 0u;
    }
};

static_assert(sizeof(PackedMove) <= 4, "Invalid Move size");

struct Move
{
    union
    {
        struct
        {
            Square fromSquare;
            Square toSquare;
            Piece piece;                // piece that is moved
            Piece promoteTo : 4;        // select target piece after promotion (only valid is piece is pawn)
            bool isCapture : 1;
            bool isEnPassant : 1;       // is en passant capture
            bool isCastling : 1;        // only valid if piece is king
        };

        uint32_t value;
    };

    INLINE static const Move Invalid() { return { 0 }; }

    INLINE bool operator == (const Move rhs) const
    {
        return value == rhs.value;
    }

    INLINE bool operator == (const PackedMove rhs) const
    {
        return (value & 0xFFFF) == rhs.value;
    }

    // valid move does not mean it's a legal move for a given position
    // use Position::IsMoveLegal() to fully validate a move
    bool IsValid() const
    {
        return value != 0u;
    }

    bool IsQuiet() const
    {
        return !isCapture && promoteTo == Piece::None;
    }

    std::string ToString() const;
};

INLINE PackedMove::PackedMove(const Move& rhs)
{
    value = static_cast<uint16_t>(rhs.value);
}

static_assert(sizeof(Move) <= 4, "Invalid Move size");

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
    const MoveEntry& GetMoveEntry(uint32_t index) const { ASSERT(index < numMoves); return moves[index]; }

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

    void Print()
    {
        for (uint32_t i = 0; i < numMoves; ++i)
        {
            std::cout << moves[i].move.ToString() << " " << moves[i].score << std::endl;
        }
    }

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
