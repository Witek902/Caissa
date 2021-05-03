#include "Move.hpp"

std::string Move::ToString() const
{
    ASSERT(piece != Piece::None);

    std::string str;

    str += fromSquare.ToString();
    str += toSquare.ToString();

    if (promoteTo != Piece::None)
    {
        str += PieceToChar(promoteTo);
    }

    return str;
}

void MoveList::RemoveMove(const Move& move)
{
    for (uint32_t i = 0; i < numMoves; ++i)
    {
        if (moves[i].move == move)
        {
            std::swap(moves[i], moves[numMoves - 1]);
            numMoves--;
            i--;
        }
    }
}