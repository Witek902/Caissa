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
    Square enPassantSquare;
};

inline Bitboard& SidePosition::GetPieceBitBoard(Piece piece)
{
    uint32_t index = (uint32_t)piece;
    assert(index >= (uint32_t)Piece::Pawn);
    assert(index <= (uint32_t)Piece::King);
    return (&pawns)[index - (uint32_t)Piece::Pawn];
}

inline const Bitboard& SidePosition::GetPieceBitBoard(Piece piece) const
{
    uint32_t index = (uint32_t)piece;
    assert(index >= (uint32_t)Piece::Pawn);
    assert(index <= (uint32_t)Piece::King);
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

    // set piece on given square
    void SetPieceAtSquare(const Square square, Piece piece, Color color);

    // check if board state is valid (proper number of pieces, no double checks etc.)
    bool IsValid() const;

    // <1 - whites are checked
    // >0 - blacks are checked
    // =0 - not check
    int IsCheck() const;

    // evaluate board
    float Evaluate() const;
    
    void GenerateMoveList(MoveList& outMoveList) const;

    // check if move is valid
    bool IsMoveValid(const Move& move) const;

    // apply a move
    bool DoMove(const Move& move);

private:

    void GeneratePawnMoveList(MoveList& outMoveList) const;
    void GenerateKnightMoveList(MoveList& outMoveList) const;
    void GenerateRookMoveList(MoveList& outMoveList) const;
    void GenerateBishopMoveList(MoveList& outMoveList) const;
    void GenerateQueenMoveList(MoveList& outMoveList) const;
    void GenerateKingMoveList(MoveList& outMoveList) const;

    SidePosition mWhites;
    SidePosition mBlacks;

    Color mSideToMove : 1;
    uint32_t mMoveCount;
};
