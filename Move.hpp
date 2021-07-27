#pragma once

#include "Square.hpp"
#include "Piece.hpp"

#include <iostream>

// "C++ nonstandard extension: nameless struct"
#pragma warning(disable : 4201)

class MoveList;

struct Move;

struct PackedMove
{
    union
    {
        struct
        {
            uint16_t fromSquare : 6;
            uint16_t toSquare : 6;
            uint16_t promoteTo : 4;
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

    std::string ToString() const;
};

static_assert(sizeof(PackedMove) == 2, "Invalid PackedMove size");

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
        return
            rhs.fromSquare == fromSquare.Index() &&
            rhs.toSquare == toSquare.Index() &&
            rhs.promoteTo == static_cast<uint8_t>(promoteTo);
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
    fromSquare = rhs.fromSquare.Index();
    toSquare = rhs.toSquare.Index();
    promoteTo = static_cast<uint8_t>(rhs.promoteTo);
}

static_assert(sizeof(Move) <= 4, "Invalid Move size");
