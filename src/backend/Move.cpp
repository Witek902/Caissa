#include "Move.hpp"
#include "Position.hpp"

std::string PackedMove::ToString() const
{
    if (FromSquare() == ToSquare())
    {
        return "0000";
    }

    std::string str;

    str += FromSquare().ToString();
    str += ToSquare().ToString();

    if (GetPromoteTo() != Piece::None)
    {
        str += PieceToChar(GetPromoteTo(), false);
    }

    return str;
}


std::string Move::ToString() const
{
    if (FromSquare() == ToSquare())
    {
        return "0000";
    }

    std::string str;

    Square toSquare = ToSquare();

    if (!Position::s_enableChess960)
    {
        if (IsShortCastle() && toSquare == Square_h1)   toSquare = Square_g1;
        if (IsShortCastle() && toSquare == Square_h8)   toSquare = Square_g8;
        if (IsLongCastle() && toSquare == Square_a1)    toSquare = Square_c1;
        if (IsLongCastle() && toSquare == Square_a8)    toSquare = Square_c8;
    }

    str += FromSquare().ToString();
    str += toSquare.ToString();

    if (GetPromoteTo() != Piece::None)
    {
        str += PieceToChar(GetPromoteTo(), false);
    }

    return str;
}
