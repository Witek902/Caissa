#include "Move.hpp"

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
    ASSERT(GetPiece() != Piece::None);

    std::string str;

    str += FromSquare().ToString();
    str += ToSquare().ToString();

    if (GetPromoteTo() != Piece::None)
    {
        str += PieceToChar(GetPromoteTo(), false);
    }

    return str;
}
