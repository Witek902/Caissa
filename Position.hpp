#pragma once

#include <inttypes.h>
#include <string>

#include "Bitboard.hpp"
#include "Piece.hpp"
#include "Square.hpp"

enum class Color : uint8_t
{
    White,
    Black,
};

inline Color GetOppositeColor(Color color)
{
    return Color((uint32_t)color ^ 1);
}

enum CastlingRights : uint8_t
{
    CastlingRights_ShortCastleAllowed = 1 << 0,
    CastlingRights_LongCastleAllowed = 1 << 1,

    CastlingRights_All = CastlingRights_ShortCastleAllowed | CastlingRights_LongCastleAllowed
};

// class representing one side's pieces state
struct SidePosition
{
    SidePosition();

    Piece GetPieceAtSquare(const Square sqare) const;
    void SetPieceAtSquare(const Square square, Piece piece);

    Bitboard& GetPieceBitBoard(Piece piece);
    const Bitboard& GetPieceBitBoard(Piece piece) const;

    __forceinline Bitboard Occupied() const
    {
        return (pawns | knights) | (bishops | rooks) | (queens | king);
    }

    __forceinline Bitboard OccupiedExcludingKing() const
    {
        return (pawns | knights) | (bishops | rooks) | queens;
    }

    Bitboard pawns = 0;
    Bitboard knights = 0;
    Bitboard bishops = 0;
    Bitboard rooks = 0;
    Bitboard queens = 0;
    Bitboard king = 0;

    CastlingRights castlingRights;
};

inline Bitboard& SidePosition::GetPieceBitBoard(Piece piece)
{
    uint32_t index = (uint32_t)piece;
    ASSERT(index >= (uint32_t)Piece::Pawn);
    ASSERT(index <= (uint32_t)Piece::King);
    return (&pawns)[index - (uint32_t)Piece::Pawn];
}

inline const Bitboard& SidePosition::GetPieceBitBoard(Piece piece) const
{
    uint32_t index = (uint32_t)piece;
    ASSERT(index >= (uint32_t)Piece::Pawn);
    ASSERT(index <= (uint32_t)Piece::King);
    return (&pawns)[index - (uint32_t)Piece::Pawn];
}

struct Move;
class MoveList;

// class representing whole board state
class Position
{
public:
    Position();
    Position(const std::string& fenString);

    // load position from Forsyth–Edwards Notation
    bool FromFEN(const std::string& fenString);

    // save position to Forsyth–Edwards Notation
    std::string ToFEN() const;

    // print board as ASCI art
    std::string Print() const;

    // convert move to string
    std::string MoveToString(const Move& move) const;

    // parse move from string
    Move MoveFromString(const std::string& str) const;

    // set piece on given square
    void SetPieceAtSquare(const Square square, Piece piece, Color color);

    // check if board state is valid (proper number of pieces, no double checks etc.)
    bool IsValid() const;

    // check if given side is in check
    bool IsInCheck(Color sideColor) const;

    // evaluate board
    float Evaluate() const;
    
    void GenerateMoveList(MoveList& outMoveList) const;

    // Check if a move is valid pseudomove. This is a partial test, it does not include checks/checkmates.
    bool IsMoveValid(const Move& move) const;

    // Check if a valid pseudomove is legal. This is a vull validity test, it includes checks/checkmates.
    // NOTE: It's assumed that provided move is a valid move, otherwise the function will assert
    bool IsMoveLegal(const Move& move) const;

    // apply a move
    bool DoMove(const Move& move);

    // get bitboard of attacked squares
    Bitboard GetAttackedSquares(Color side) const;

private:

    void GeneratePawnMoveList(MoveList& outMoveList) const;
    void GenerateKnightMoveList(MoveList& outMoveList) const;
    void GenerateRookMoveList(MoveList& outMoveList) const;
    void GenerateBishopMoveList(MoveList& outMoveList) const;
    void GenerateQueenMoveList(MoveList& outMoveList) const;
    void GenerateKingMoveList(MoveList& outMoveList) const;

    void ClearRookCastlingRights(const Square affectedSquare);

    SidePosition mWhites;
    SidePosition mBlacks;
    Square mEnPassantSquare;
    Color mSideToMove : 1;
    uint32_t mHalfMoveCount;
    uint32_t mMoveCount;
};
