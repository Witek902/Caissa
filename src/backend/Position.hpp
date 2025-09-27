#pragma once

#include "PositionHash.hpp"
#include "Bitboard.hpp"

#include <string>
#include <vector>

// class representing one side's pieces state
struct alignas(16) SidePosition
{
    INLINE Piece GetPieceAtSquare(const Square square) const
    {
        ASSERT(square.IsValid());
        const Piece piece = pieces[square.mIndex];

        if (piece != Piece::None)
            ASSERT(GetPieceBitBoard(piece).IsBitSet(square.mIndex));

        return piece;
    }

    Bitboard& GetPieceBitBoard(Piece piece);
    const Bitboard& GetPieceBitBoard(Piece piece) const;

    INLINE Bitboard Occupied() const
    {
        return pawns | knights | bishops | rooks | queens | king;
    }

    INLINE Bitboard OccupiedExcludingKing() const
    {
        return pawns | knights | bishops | rooks | queens;
    }

    INLINE Square GetKingSquare() const
    {
        ASSERT(king);
        return Square(FirstBitSet(king));
    }

    INLINE bool operator == (const SidePosition& rhs) const
    {
        return
            pawns == rhs.pawns &&
            knights == rhs.knights &&
            bishops == rhs.bishops &&
            rooks == rhs.rooks &&
            queens == rhs.queens &&
            king == rhs.king;
    }

    INLINE bool operator != (const SidePosition& rhs) const
    {
        return
            pawns != rhs.pawns ||
            knights != rhs.knights ||
            bishops != rhs.bishops ||
            rooks != rhs.rooks ||
            queens != rhs.queens ||
            king != rhs.king;
    }

    Bitboard pawns = 0;
    Bitboard knights = 0;
    Bitboard bishops = 0;
    Bitboard rooks = 0;
    Bitboard queens = 0;
    Bitboard king = 0;
    Piece pieces[64] = {};
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

enum class MoveNotation : uint8_t
{
    SAN,    // Standard Algebraic Notation
    LAN,    // Long Algebraic Notation
};

struct Threats
{
    Bitboard attackedByPawns;
    Bitboard attackedByMinors;
    Bitboard attackedByRooks;
    Bitboard allThreats;
};

// class representing whole board state
class alignas(64) Position
{
public:
    static const char* InitPositionFEN;

    static bool s_enableChess960;

    Position();

    INLINE Position(const Position& rhs)
    {
        AlignedMemcpy64(this, &rhs);
    }

    INLINE Position(Position&& rhs)
    {
        AlignedMemcpy64(this, &rhs);
    }

    INLINE Position& operator=(const Position& rhs)
    {
        AlignedMemcpy64(this, &rhs);
        return *this;
    }

    INLINE Position& operator=(Position&& rhs)
    {
        AlignedMemcpy64(this, &rhs);
        return *this;
    }

    explicit Position(const std::string& fenString);

    // compare position (not hash)
    bool operator == (const Position& rhs) const;
    bool operator != (const Position& rhs) const;

    // load position from Forsyth-Edwards Notation
    bool FromFEN(const std::string& fenString);

    // save position to Forsyth-Edwards Notation
    std::string ToFEN(bool skipMoveCounts = false) const;

    // print board as ASCI art
    std::string Print() const;

    // convert move to string
    std::string MoveToString(const Move& move, MoveNotation notation = MoveNotation::SAN) const;

    // parse move from string
    Move MoveFromString(const std::string& str, MoveNotation notation = MoveNotation::LAN) const;

    // parse move from packed move
    Move MoveFromPacked(const PackedMove& packedMove) const;

    // set piece on given square (square is expected to be empty)
    void SetPiece(const Square square, const Piece piece, const Color color);

    // remove piece on given square
    void RemovePiece(const Square square, const Piece piece, const Color color);

    // update en passant square
    void SetEnPassantSquare(const Square square);
    void ClearEnPassantSquare();

    // check if board state is valid (no double checks etc.)
    // strict mode checks if the position is "normal" chess position (proper piece count, proper pawn placement, etc.)
    bool IsValid(bool strict = false) const;

    // get pieces attacking given square
    const Bitboard GetAttackers(const Square square, const Bitboard occupied) const;

    // get pieces of one side attacking given square
    const Bitboard GetAttackers(const Square square, const Color sideColor) const;

    // check if given square is visible by any other piece
    bool IsSquareVisible(const Square square, const Color sideColor) const;

    // check if side to move is in check
    bool IsInCheck() const;

    // check if given side is in check
    bool IsInCheck(Color sideColor) const;

    // check if a move gives check
    // approximate, does not handle discovered checks, check by castling, etc.
    bool GivesCheck_Approx(const Move move) const;

    // get number of legal moves
    uint32_t GetNumLegalMoves(std::vector<Move>* outMoves = nullptr) const;

    // check if side to move is checkmated
    bool IsMate() const;

    // check if side to move has no legal moves
    bool IsStalemate() const;

    // check if position is drawn due to fifty move rule
    bool IsFiftyMoveRuleDraw() const;

    // check if position is relatively quiet (no immediate winning captures and promotions)
    bool IsQuiet() const;
    
    // Check if a move is valid pseudomove. This is a partial test, it does not include checks/checkmates.
    bool IsMoveValid(const Move& move) const;
    bool IsMoveValid_Fast(const PackedMove& move) const;

    // Approximate check if packed move is a capture
    bool IsCapture(const PackedMove& move) const;

    // get captured piece type
    Piece GetCapturedPiece(const Move move) const;

    // Check if a valid pseudomove is legal. This is full validity test, it includes checks/checkmates.
    // NOTE: It's assumed that provided move is a valid move, otherwise the function will assert
    bool IsMoveLegal(const Move& move) const;

    // apply a move
    bool DoMove(const Move& move);
    bool DoMove(const Move& move, NNEvaluatorContext& nnContext);

    // apply null move
    bool DoNullMove();

    // check what is theoretically possible best move value (without generating and analyzing actual moves)
    int32_t BestPossibleMoveValue() const;

    // evaluate material exchange on a single square
    bool StaticExchangeEvaluation(const Move& move, int32_t treshold = 0) const;

    // compute (SLOW) Zobrist hash
    uint64_t ComputeHash() const;

    // get bitboard of attacked squares
    Bitboard GetAttackedSquares(Color side) const;

    // return position where colors are swapped (but the board is not flipped)
    Position SwappedColors() const;

    void MirrorVertically();
    void MirrorHorizontally();
    void FlipDiagonally();

    Position MirroredVertically() const;
    Position MirroredHorizontally() const;

    // run performance test
    uint64_t Perft(uint32_t depth, bool print = false) const;

    INLINE const SidePosition& Whites() const { return mColors[0]; }
    INLINE const SidePosition& Blacks() const { return mColors[1]; }
    INLINE const SidePosition& GetSide(const Color color) const { return color == White ? mColors[0] : mColors[1]; }
    INLINE const SidePosition& GetCurrentSide() const { return mSideToMove == White ? mColors[0] : mColors[1]; }
    INLINE const SidePosition& GetOpponentSide() const { return mSideToMove == White ? mColors[1] : mColors[0]; }
    INLINE const Square GetCurrentSideKingSquare() const { return GetCurrentSide().GetKingSquare(); }

    INLINE uint8_t GetWhitesCastlingRights() const { return mCastlingRights[0]; }
    INLINE uint8_t GetBlacksCastlingRights() const { return mCastlingRights[1]; }
    INLINE uint8_t GetOurCastlingRights() const { return mCastlingRights[(uint32_t)mSideToMove]; }

    INLINE uint32_t GetNumPiecesExcludingKing() const
    {
        ASSERT(Whites().king.Count() == 1);
        ASSERT(Blacks().king.Count() == 1);
        return (Whites().OccupiedExcludingKing() | Blacks().OccupiedExcludingKing()).Count();
    }

    INLINE uint32_t GetNumPieces() const
    {
        return 2 + GetNumPiecesExcludingKing();
    }

    // get occupied squares bitboard
    INLINE Bitboard Occupied() const { return Whites().Occupied() | Blacks().Occupied(); }
    INLINE Bitboard OccupiedExcludingKing() const { return Whites().OccupiedExcludingKing() | Blacks().OccupiedExcludingKing(); }

    // get board hash
    INLINE uint64_t GetHash() const { return mHash; }
    INLINE uint64_t GetPawnsHash() const { return mPawnsHash; }
    INLINE uint32_t GetNonPawnsHash(Color color) const { return mNonPawnsHash[color]; }
    uint64_t HashAfterMove(const Move move) const;
    uint64_t HashAfterMove(const PackedMove move) const;

    INLINE Color GetSideToMove() const { return mSideToMove; }
    INLINE Square GetEnPassantSquare() const { return mEnPassantSquare; }
    INLINE uint16_t GetHalfMoveCount() const { return mHalfMoveCount; }
    INLINE uint16_t GetMoveCount() const { return mMoveCount; }

    void SetSideToMove(Color color);
    void SetCastlingRights(Color color, uint8_t rightsMask);
    INLINE void SetHalfMoveCount(uint16_t halfMoveCount) { mHalfMoveCount = halfMoveCount; }
    INLINE void SetMoveCount(uint16_t moveCount) { mMoveCount = moveCount; }

    // check if a side features non-pawn pieces
    bool HasNonPawnMaterial(Color color) const;

    // compute material key (number of pieces of each kind)
    const MaterialKey GetMaterialKey() const;

    void ComputeThreats(Threats& outThreats) const;

    static Square GetLongCastleRookSquare(const Square kingSquare, uint8_t castlingRights);
    static Square GetShortCastleRookSquare(const Square kingSquare, uint8_t castlingRights);

private:

    INLINE SidePosition& GetSide(const Color color) { return color == White ? mColors[0] : mColors[1]; }
    INLINE SidePosition& GetCurrentSide() { return mSideToMove == White ? mColors[0] : mColors[1]; }
    INLINE SidePosition& GetOpponentSide() { return mSideToMove == White ? mColors[1] : mColors[0]; }

    Square ExtractEnPassantSquareFromMove(const Move& move) const;

    void ClearRookCastlingRights(const Square affectedSquare);

    // BOARD STATE & FLAGS

    // bitboards for whites and blacks
    SidePosition mColors[2];

    // who's next move?
    Color mSideToMove;

    // en passant target square
    Square mEnPassantSquare;

    uint8_t mCastlingRights[2];

    uint16_t mHalfMoveCount;
    uint16_t mMoveCount;

    // METADATA

    uint64_t mHash;
    uint64_t mPawnsHash;
    uint32_t mNonPawnsHash[2];
};

static_assert(sizeof(Position) <= 256, "Invalid position size");

static constexpr uint8_t c_shortCastleMask = (1 << 7);
static constexpr uint8_t c_longCastleMask = (1 << 0);
