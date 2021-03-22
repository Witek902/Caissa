#pragma once

#include "Square.hpp"
#include <assert.h>

struct Move
{
    Square fromSquare;
    Square toSquare;
    Piece piece;        // piece that is moved
    Piece promoteTo;    // select target piece after promotion
    bool isCapture = false;
    bool isCastling = false;
};

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