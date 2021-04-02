#pragma once

#include <inttypes.h>
#include <string>

#include "Color.hpp"

enum class Piece : uint8_t
{
    None,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,

    Invalid = 0xFF
};

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

inline std::string PieceToUnicodeFancy(Piece p, Color color)
{
    std::string str;

    switch (p)
    {
    case Piece::Pawn:   str = color == Color::White ? "♙" : "♟︎"; break;
    case Piece::Knight: str = color == Color::White ? "♘" : "♞"; break;
    case Piece::Bishop: str = color == Color::White ? "♗" : "♝"; break;
    case Piece::Rook:   str = color == Color::White ? "♖" : "♜"; break;
    case Piece::Queen:  str = color == Color::White ? "♕" : "♛"; break;
    case Piece::King:   str = color == Color::White ? "♔" : "♚"; break;
    }

    return str;
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