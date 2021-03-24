#pragma once

#include "Square.hpp"
#include <assert.h>

struct Move
{
    Square fromSquare;
    Square toSquare;
    Piece piece : 4;            // piece that is moved
    Piece promoteTo : 4;        // select target piece after promotion (only valid is piece is pawn)
    bool isCapture : 1;
    bool isEnPassant : 1;
    bool isCastling : 1;        // only valid if piece is king

    Move() = default;
    Move(const Move&) = default;
    Move& operator = (const Move&) = default;

    // valid move does not mean it's a legal move for a given position
    // use Position::IsMoveLegal() to fully validate a move
    bool IsValid() const
    {
        return fromSquare.IsValid() && toSquare.IsValid();
    }
};

static_assert(sizeof(Move) <= 4, "Invalid Move size");

class MoveList
{
    friend class Position;

public:

    // https://chess.stackexchange.com/questions/4490/maximum-possible-movement-in-a-turn
    static constexpr uint32_t MaxMoves = 216;

    uint32_t Size() const { return mNumMoves; }
    const Move& GetMove(uint32_t index) const { assert(index < mNumMoves); return mMoves[index]; }

private:

    Move& PushMove()
    {
        assert(mNumMoves < MaxMoves);
        return mMoves[mNumMoves++];
    }

    uint32_t mNumMoves = 0;
    Move mMoves[MaxMoves];
};