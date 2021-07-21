#include "Move.hpp"

std::string Move::ToString() const
{
    ASSERT(piece != Piece::None);

    std::string str;

    str += fromSquare.ToString();
    str += toSquare.ToString();

    if (promoteTo != Piece::None)
    {
        str += PieceToChar(promoteTo, false);
    }

    return str;
}
