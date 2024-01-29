#pragma once

#include "Square.hpp"
#include "Piece.hpp"

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

    INLINE PackedMove() = default;

    INLINE PackedMove(const Square fromSquare, const Square toSquare, Piece promoteTo = Piece::None)
    {
        value = ((uint32_t)fromSquare.mIndex) | ((uint32_t)toSquare.mIndex << 6) | ((uint32_t)promoteTo << 12);
    }

    INLINE static const PackedMove Invalid()
    {
        PackedMove move;
        move.value = 0;
        return move;
    }

    INLINE constexpr PackedMove(const PackedMove&) = default;
    INLINE constexpr PackedMove& operator = (const PackedMove&) = default;

    // make from regular move
    PackedMove(const Move rhs);

    INLINE const Square FromSquare() const { return value & 0x3F; }
    INLINE const Square ToSquare() const { return (value >> 6) & 0x3F; }
    INLINE Piece GetPromoteTo() const { return (Piece)((value >> 12) & 0xF); }

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

    INLINE constexpr bool operator != (const PackedMove& rhs) const
    {
        return value != rhs.value;
    }

    std::string ToString() const;
};

static_assert(sizeof(PackedMove) == 2, "Invalid PackedMove size");

struct Move
{
    INLINE const Square FromSquare() const      { return value & 0x3F; }
    INLINE const Square ToSquare() const        { return (value >> 6) & 0x3F; }
    INLINE uint32_t FromTo() const              { return value & 0xFFF; }
    INLINE Piece GetPromoteTo() const           { return (Piece)((value >> 12) & 0xF); }
    INLINE Piece GetPiece() const               { return (Piece)((value >> 16) & 0xF); }
    INLINE constexpr bool IsCapture() const     { return value & (1u << 20); }
    INLINE constexpr bool IsEnPassant() const   { return value & (1u << 21); }
    INLINE constexpr bool IsLongCastle() const  { return value & (1u << 22); }
    INLINE constexpr bool IsShortCastle() const { return value & (1u << 23); }
    INLINE constexpr bool IsCastling() const    { return (value >> 22) & 3; }

    // data layout is following:
    //
    // [type]   [property]      [bits]
    // 
    // Square   fromSquare      : 6
    // Square   toSquare        : 6
    // Piece    promoteTo       : 4     target piece after promotion (only valid is piece is pawn)
    // Piece    piece           : 4
    // bool     isCapture       : 1
    // bool     isEnPassant     : 1     (is en passant capture)
    // bool     isLongCastle    : 1     (only valid if piece is king)
    // bool     isShortCastle   : 1     (only valid if piece is king)
    //
    uint32_t value;

    static constexpr uint32_t mask = (1 << 23) - 1;

    INLINE static constexpr Move Make(
        Square fromSquare, Square toSquare, Piece piece, Piece promoteTo = Piece::None,
        bool isCapture = false, bool isEnPassant = false, bool isLongCastle = false, bool isShortCastle = false)
    {
        return
        {
            ((uint32_t)fromSquare.mIndex) |
            ((uint32_t)toSquare.mIndex << 6) |
            ((uint32_t)promoteTo << 12) |
            ((uint32_t)piece << 16) |
            ((uint32_t)isCapture << 20) |
            ((uint32_t)isEnPassant << 21) |
            ((uint32_t)isLongCastle << 22) |
            ((uint32_t)isShortCastle << 23)
        };
    }

    template<Piece piece, bool isCapture>
    INLINE static constexpr Move MakeSimple(Square fromSquare, Square toSquare)
    {
        return
        {
            ((uint32_t)fromSquare.mIndex) |
            ((uint32_t)toSquare.mIndex << 6) |
            ((uint32_t)piece << 16) |
            ((uint32_t)isCapture << 20)
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

    INLINE bool IsUnderpromotion() const
    {
        static_assert((int32_t)Piece::Rook - (int32_t)Piece::Knight == 2, "Unexpected piece order");
        return GetPromoteTo() >= Piece::Knight && GetPromoteTo() <= Piece::Rook;
    }

    INLINE bool IsIrreversible() const
    {
        return IsCapture() || GetPiece() == Piece::Pawn;
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
    INLINE bool HasMove(const MoveType2 move) const
    {
        if (move.IsValid())
        {
            for (uint32_t j = 0; j < MaxSize; ++j)
            {
                if (move == moves[j]) return true;
            }
        }
        return false;
    }

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

    MoveType moves[MaxSize];
};