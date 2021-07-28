#pragma once

#include "Common.hpp"
#include "Bitboard.hpp"
#include "Piece.hpp"
#include "Square.hpp"
#include "Color.hpp"

#include <string>
#include <vector>

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
        return occupied;
    }

    INLINE Bitboard OccupiedExcludingKing() const
    {
        return occupied & ~king;
    }

    INLINE Square GetKingSquare() const
    {
        unsigned long kingSquareIndex;
        const bool kingSquareFound = _BitScanForward64(&kingSquareIndex, king);
        ASSERT(kingSquareFound);
        return Square(kingSquareIndex);
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
    Bitboard occupied = 0;
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

#define MOVE_GEN_ONLY_TACTICAL 1

// class representing whole board state
class Position
{
public:
    static const char* InitPositionFEN;

    Position();
    Position(const Position&) = default;
    explicit Position(const std::string& fenString);

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

    // get bitboard of attacking squares
    const Bitboard GetAttackers(const Square square, const Color sideColor) const;

    // check if given square is visible by any other piece
    bool IsSquareVisible(const Square square, const Color sideColor) const;

    // check if given side is in check
    bool IsInCheck(Color sideColor) const;

    // get number of legal moves
    uint32_t GetNumLegalMoves(std::vector<Move>* outMoves = nullptr) const;

    // check if side to move is checkmated
    bool IsMate() const;

    // check if side to move has no legal moves
    bool IsStalemate() const;
    
    void GenerateMoveList(MoveList& outMoveList, uint32_t flags = 0) const;

    // Check if a move is valid pseudomove. This is a partial test, it does not include checks/checkmates.
    bool IsMoveValid(const Move& move) const;
    bool IsMoveValid_Fast(const PackedMove& move) const;

    // Check if a valid pseudomove is legal. This is a vull validity test, it includes checks/checkmates.
    // NOTE: It's assumed that provided move is a valid move, otherwise the function will assert
    bool IsMoveLegal(const Move& move) const;

    // apply a move
    bool DoMove(const Move& move);

    // apply null move
    bool DoNullMove();

    // evaluate material exchange on a single square
    int32_t StaticExchangeEvaluation(const Move& move) const;

    // compute (SLOW) Zobrist hash
    uint64_t ComputeHash() const;

    // get bitboard of attacked squares
    Bitboard GetAttackedSquares(Color side) const;

    // run performance test
    uint64_t Perft(uint32_t depth, bool print = true) const;

    INLINE const SidePosition& Whites() const { return mColors[0]; }
    INLINE const SidePosition& Blacks() const { return mColors[1]; }

    INLINE const SidePosition& GetCurrentSide() const { return mColors[(uint8_t)mSideToMove]; }
    INLINE const SidePosition& GetOpponentSide() const { return mColors[(uint8_t)mSideToMove ^ 1]; }

    INLINE CastlingRights GetWhitesCastlingRights() const { return mWhitesCastlingRights; }
    INLINE CastlingRights GetBlacksCastlingRights() const { return mBlacksCastlingRights; }

    // get board hash
    INLINE uint64_t GetHash() const { return mHash; }

    // get color to move
    INLINE Color GetSideToMove() const { return mSideToMove; }

    INLINE Square GetEnPassantSquare() const { return mEnPassantSquare; }
    INLINE uint16_t GetHalfMoveCount() const { return mHalfMoveCount; }

private:

    friend class Search;

    INLINE SidePosition& Whites() { return mColors[0]; }
    INLINE SidePosition& Blacks() { return mColors[1]; }

    INLINE SidePosition& GetCurrentSide() { return mColors[(uint8_t)mSideToMove]; }
    INLINE SidePosition& GetOpponentSide() { return mColors[(uint8_t)mSideToMove ^ 1]; }

    void GeneratePawnMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateKnightMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateRookMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateBishopMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateQueenMoveList(MoveList& outMoveList, uint32_t flags = 0) const;
    void GenerateKingMoveList(MoveList& outMoveList, uint32_t flags = 0) const;

    void ClearRookCastlingRights(const Square affectedSquare);

    // BOARD STATE & FLAGS

    // bitboards for whites and blacks
    SidePosition mColors[2];

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
};

static_assert(sizeof(Position) == 128, "Invalid position size");

void InitZobristHash();
