#include "Move.hpp"
#include "Position.hpp"

MoveString PackedMove::ToString() const
{
    if (FromSquare() == ToSquare())
    {
        return "0000";
    }

    MoveString str;

    str.Append(FromSquare().ToString());
    str.Append(ToSquare().ToString());

    if (GetPromoteTo() != Piece::None)
    {
        str.Append(PieceToChar(GetPromoteTo(), false));
    }

    return str;
}

MoveString Move::ToString() const
{
    if (FromSquare() == ToSquare())
    {
        return "0000";
    }

    MoveString str;

    Square toSquare = ToSquare();

    if (!Position::s_enableChess960)
    {
        if (IsShortCastle() && toSquare == Square_h1)   toSquare = Square_g1;
        if (IsShortCastle() && toSquare == Square_h8)   toSquare = Square_g8;
        if (IsLongCastle() && toSquare == Square_a1)    toSquare = Square_c1;
        if (IsLongCastle() && toSquare == Square_a8)    toSquare = Square_c8;
    }

    str.Append(FromSquare().ToString());
    str.Append(toSquare.ToString());

    if (GetPromoteTo() != Piece::None)
    {
        str.Append(PieceToChar(GetPromoteTo(), false));
    }

    return str;
}
