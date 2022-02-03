#pragma once

#include "Square.hpp"
#include "Piece.hpp"

#include <iostream>

struct PackedMove
{
    // data layout is following:
    //
    // [type]   [property]  [bits]
    // 
    // Square   fromSquare  : 6
    // Square   toSquare    : 6
    // Piece    promoteTo   : 4     target piece after promotion (only valid is piece is pawn)
    //
    uint16_t value;

    INLINE constexpr PackedMove() = default;

    INLINE constexpr PackedMove(const Square fromSquare, const Square toSquare, Piece promoteTo = Piece::None)
    {
        value = ((uint32_t)fromSquare.mIndex) | ((uint32_t)toSquare.mIndex << 6) | ((uint32_t)promoteTo << 12);
    }

    INLINE static constexpr const PackedMove Invalid()
    {
        PackedMove move;
        move.value = 0;
        return move;
    }

    INLINE constexpr PackedMove(const PackedMove&) = default;
    INLINE constexpr PackedMove& operator = (const PackedMove&) = default;

    // make from regular move
    PackedMove(const Move rhs);

    INLINE const Square FromSquare() const { return value & 0b111111; }
    INLINE const Square ToSquare() const { return (value >> 6) & 0b111111; }
    INLINE const Piece GetPromoteTo() const { return (Piece)((value >> 12) & 0b1111); }

    // valid move does not mean it's a legal move for a given position
    // use Position::IsMoveLegal() to fully validate a move
    bool constexpr IsValid() const
    {
        return value != 0u;
    }

    INLINE constexpr bool operator == (const PackedMove& rhs) const
    {
        return value == rhs.value;
    }

    std::string ToString() const;
};

static_assert(sizeof(PackedMove) == 2, "Invalid PackedMove size");

struct Move
{
    INLINE const Square FromSquare() const      { return value & 0b111111; }
    INLINE const Square ToSquare() const        { return (value >> 6) & 0b111111; }
    INLINE const Piece GetPromoteTo() const     { return (Piece)((value >> 12) & 0b1111); }
    INLINE const Piece GetPiece() const         { return (Piece)((value >> 16) & 0b1111); }
    INLINE constexpr bool IsCapture() const     { return (value >> 20) & 1; }
    INLINE constexpr bool IsEnPassant() const   { return (value >> 21) & 1; }
    INLINE constexpr bool IsCastling() const    { return (value >> 22) & 1; }

    // data layout is following:
    //
    // [type]   [property]  [bits]
    // 
    // Square   fromSquare  : 6
    // Square   toSquare    : 6
    // Piece    promoteTo   : 4     target piece after promotion (only valid is piece is pawn)
    // Piece    piece       : 4
    // bool     isCapture   : 1
    // bool     isEnPassant : 1     (is en passant capture)
    // bool     isCastling  : 1     (only valid if piece is king)
    //
    uint32_t value;

    static constexpr uint32_t mask = (1 << 23) - 1;

    INLINE static constexpr Move Make(
        Square fromSquare, Square toSquare, Piece piece, Piece promoteTo = Piece::None,
        bool isCapture = false, bool isEnPassant = false, bool isCastling = false)
    {
        return
        {
            ((uint32_t)fromSquare.mIndex) |
            ((uint32_t)toSquare.mIndex << 6) |
            ((uint32_t)promoteTo << 12) |
            ((uint32_t)piece << 16) |
            ((uint32_t)isCapture << 20) |
            ((uint32_t)isEnPassant << 21) |
            ((uint32_t)isCastling << 22)
        };
    }

    INLINE static constexpr const Move Invalid()
    {
        return { 0 };
    }

    INLINE constexpr bool operator == (const Move rhs) const
    {
        return (value & mask) == (rhs.value & mask);
    }

    INLINE constexpr bool operator != (const Move rhs) const
    {
        return (value & mask) != (rhs.value & mask);
    }

    INLINE bool operator == (const PackedMove rhs) const
    {
        return (value & 0xFFFFu) == rhs.value;
    }

    INLINE bool operator != (const PackedMove rhs) const
    {
        return (value & 0xFFFFu) != rhs.value;
    }

    // valid move does not mean it's a legal move for a given position
    // use Position::IsMoveLegal() to fully validate a move
    INLINE bool constexpr IsValid() const
    {
        return value != 0u;
    }

    INLINE bool IsQuiet() const
    {
        return !IsCapture() && GetPromoteTo() == Piece::None;
    }

    INLINE bool IsPromotion() const
    {
        return GetPromoteTo() != Piece::None;
    }

    std::string ToString() const;
};

INLINE PackedMove::PackedMove(const Move rhs)
{
    value = static_cast<uint16_t>(rhs.value);
}

static_assert(sizeof(Move) <= 4, "Invalid Move size");


template<typename MoveType, uint32_t MaxSize>
class MovesArray
{
public:

    INLINE MovesArray()
    {
        for (uint32_t i = 0; i < MaxSize; ++i)
        {
            moves[i] = MoveType::Invalid();
        }
    }

    INLINE MoveType& operator[](uint32_t i) { return moves[i]; }
    INLINE const MoveType operator[](uint32_t i) const { return moves[i]; }

    INLINE MoveType* Data() { return moves; }
    INLINE const MoveType* Data() const { return moves; }

    template<typename MoveType2>
    void Remove(const MoveType2 move)
    {
        for (uint32_t j = 0; j < MaxSize; ++j)
        {
            if (move == moves[j])
            {
                // push remaining moves to the front
                for (uint32_t i = j; i < MaxSize - 1; ++j)
                {
                    moves[i] = moves[i + 1];
                }
                moves[MaxSize - 1] = MoveType();

                break;
            }
        }
    }

    template<typename MoveType2, uint32_t MaxSize2>
    uint32_t MergeWith(const MovesArray<MoveType2, MaxSize2>& other)
    {
        uint32_t outSize = 0;
        for (uint32_t i = 0; i < MaxSize; ++i)
        {
            if (!moves[i].IsValid()) break;
            outSize++;
        }

        for (uint32_t i = 0; i < MaxSize2 && outSize < MaxSize; ++i)
        {
            if (!other.moves[i].IsValid()) break;
            bool moveExists = false;
            for (uint32_t j = 0; j < outSize; ++j)
            {
                if (moves[j] == other.moves[i])
                {
                    moveExists = true;
                    break;
                }
            }
            if (!moveExists)
            {
                moves[outSize++] = other.moves[i];
            }
        }

        ASSERT(outSize <= MaxSize);

        return outSize;
    }

    MoveType moves[MaxSize];
};