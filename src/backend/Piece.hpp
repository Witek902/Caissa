#pragma once

#include "Common.hpp"

#include <string>

enum class Piece : uint8_t
{
    None,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
};

INLINE Piece NextPiece(Piece piece)
{
    ASSERT((uint32_t)piece < (uint32_t)Piece::King);
    return (Piece)(uint32_t(piece) + 1);
}

inline char PieceToChar(Piece p, bool upperCase = true)
{
    char c = ' ';
    switch (p)
    {
    case Piece::Pawn:   c = 'p'; break;
    case Piece::Knight: c = 'n'; break;
    case Piece::Bishop: c = 'b'; break;
    case Piece::Rook:   c = 'r'; break;
    case Piece::Queen:  c = 'q'; break;
    case Piece::King:   c = 'k'; break;
    }

    if (upperCase)
    {
        c -= 32;
    }

    return c;
}

inline const char* PieceToString(Piece p)
{
    switch (p)
    {
    case Piece::Pawn:   return "Pawn";
    case Piece::Knight: return "Knight";
    case Piece::Bishop: return "Bishop";
    case Piece::Rook:   return "Rook";
    case Piece::Queen:  return "Queen";
    case Piece::King:   return "King";
    default: return "";
    }
}

inline bool CharToPiece(const char ch, Piece& outPiece)
{
    switch (ch)
    {
    case 'p':
    case 'P':
        outPiece = Piece::Pawn; return true;
    case 'n':
    case 'N':
        outPiece = Piece::Knight; return true;
    case 'b':
    case 'B':
        outPiece = Piece::Bishop; return true;
    case 'r':
    case 'R':
        outPiece = Piece::Rook; return true;
    case 'q':
    case 'Q':
        outPiece = Piece::Queen; return true;
    case 'k':
    case 'K':
        outPiece = Piece::King; return true;
    }

    return false;
}