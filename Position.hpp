#pragma once

#include <inttypes.h>
#include <string>

#include "Bitboard.hpp"
#include "Piece.hpp"
#include "Square.hpp"
#include "Color.hpp"

enum CastlingRights : uint8_t
{
    CastlingRights_ShortCastleAllowed = 1 << 0,
    CastlingRights_LongCastleAllowed = 1 << 1,

    CastlingRights_All = CastlingRights_ShortCastleAllowed | CastlingRights_LongCastleAllowed
};

// class representing one side's pieces state
struct SidePosition
{
    Piece GetPieceAtSquare(const Square sqare) const;

    Bitboard& GetPieceBitBoard(Piece piece);
    const Bitboard& GetPieceBitBoard(Piece piece) const;

    INLINE Bitboard Occupied() const
    {
        return (pawns | knights) | (bishops | rooks) | (queens | king);
    }

    INLINE Bitboard OccupiedExcludingKing() const
    {
        return (pawns | knights) | (bishops | rooks) | queens;
    }

    bool operator == (const SidePosition& rhs) const
    {
        return
            pawns == rhs.pawns &&
            knights == rhs.knights &&
            bishops == rhs.bishops &&
            rooks == rhs.rooks &&
            queens == rhs.queens &&
            king == rhs.king;
    }

    Bitboard pawns = 0;
    Bitboard knights = 0;
    Bitboard bishops = 0;
    Bitboard rooks = 0;
    Bitboard queens = 0;
    Bitboard king = 0;
};

INLINE Bitboard& SidePosition::GetPieceBitBoard(Piece piece)
{
    uint32_t index = (uint32_t)piece;
    ASSERT(index >= (uint32_t)Piece::Pawn);
    ASSERT(index <= (uint32_t)Piece::King);
    return (&pawns)[index - (uint32_t)Piece::Pawn];
}

INLINE const Bitboard& SidePosition::GetPieceBitBoard(Piece piece) const
{
    uint32_t index = (uint32_t)piece;
    ASSERT(index >= (uint32_t)Piece::Pawn);
    ASSERT(index <= (uint32_t)Piece::King);
    return (&pawns)[index - (uint32_t)Piece::Pawn];
}

struct Move;
struct PackedMove;
class MoveList;

#define MOVE_GEN_ONLY_CAPTURES 1

// class representing whole board state
class Position
{
public:
    Position();
    Position(const Position&) = default;
    Position(const std::string& fenString);

    // compare position (not hash)
    bool operator == (const Position& rhs) const;

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

    // parse move from packed move
    Move MoveFromPacked(const PackedMove& packedMove) const;

    // set piece on given square (square is expected to be empty)
    void SetPiece(const Square square, const Piece piece, const Color color);

    // remove piece on given square
    void RemovePiece(const Square square, const Piece piece, const Color color);

    // update en passant square
    void SetEnPassantSquare(const Square square);
    void ClearEnPassantSquare();

    // check if board state is valid (proper number of pieces, no double checks etc.)
    bool IsValid() const;

    // check if given side is in check
    bool IsInCheck(Color sideColor) const;
    
    void GenerateMoveList(MoveList& outMoveList, uint32_t flags = 0) const;

    // Check if a move is valid pseudomove. This is a partial test, it does not include checks/checkmates.
    bool IsMoveValid(const Move& move) const;

    // Check if a valid pseudomove is legal. This is a vull validity test, it includes checks/checkmates.
    // NOTE: It's assumed that provided move is a valid move, otherwise the function will assert
    bool IsMoveLegal(const Move& move) const;

    // apply a move
    bool DoMove(const Move& move);

    // compute (SLOW) Zobrist hash
    uint64_t ComputeHash() const;

    // get bitboard of attacked squares
    Bitboard GetAttackedSquares(Color side) const;

    const SidePosition& Whites() const { return mWhites; }
    const SidePosition& Blacks() const { return mBlacks; }

    INLINE CastlingRights GetWhitesCastlingRights() const { return mWhitesCastlingRights; }
    INLINE CastlingRights GetBlacksCastlingRights() const { return mBlacksCastlingRights; }

    // get board hash
    INLINE uint64_t GetHash() const { return mHash; }

    // get color to move
    INLINE Color GetSideToMove() const { return mSideToMove; }

    INLINE Bitboard GetAttackedByWhites() const { return mAttackedByWhites; }
    INLINE Bitboard GetAttackedByBlacks() const { return mAttackedByBlacks; }

private:

    friend class Search;

    void PushMove(const Move move, MoveList& outMoveList) const;

    void GeneratePawnMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateKnightMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateRookMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateBishopMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateQueenMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateKingMoveList(MoveList& outMoveList, uint32_t flags = 0) const;

    void ClearRookCastlingRights(const Square affectedSquare);

    // BOARD STATE & FLAGS

    // bitboards
    SidePosition mWhites;
    SidePosition mBlacks;

    // who's next move?
    Color mSideToMove;

    // en passant target square
    Square mEnPassantSquare;

    CastlingRights mWhitesCastlingRights;
    CastlingRights mBlacksCastlingRights;

    uint16_t mHalfMoveCount;
    uint16_t mMoveCount;

    // METADATA

    uint64_t mHash; // whole position hash

    Bitboard mAttackedByWhites;
    Bitboard mAttackedByBlacks;
};

static_assert(sizeof(Position) == 128, "Invalid position size");

void InitZobristHash();
