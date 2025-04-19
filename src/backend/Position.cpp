#include "Position.hpp"
#include "Material.hpp"
#include "Evaluate.hpp"
#include "MoveGen.hpp"
#include "NeuralNetworkEvaluator.hpp"

const char* Position::InitPositionFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

bool Position::s_enableChess960 = false;

uint64_t Position::ComputeHash() const
{
    uint64_t hash = mSideToMove == Black ? c_SideToMoveZobristHash : 0llu;

    for (Color color = 0; color < 2; ++color)
    {
        const SidePosition& pos = mColors[color];

        pos.pawns.Iterate([&](uint32_t square)   INLINE_LAMBDA { hash ^= GetPieceZobristHash(color, Piece::Pawn, square); });
        pos.knights.Iterate([&](uint32_t square) INLINE_LAMBDA { hash ^= GetPieceZobristHash(color, Piece::Knight, square); });
        pos.bishops.Iterate([&](uint32_t square) INLINE_LAMBDA { hash ^= GetPieceZobristHash(color, Piece::Bishop, square); });
        pos.rooks.Iterate([&](uint32_t square)   INLINE_LAMBDA { hash ^= GetPieceZobristHash(color, Piece::Rook, square); });
        pos.queens.Iterate([&](uint32_t square)  INLINE_LAMBDA { hash ^= GetPieceZobristHash(color, Piece::Queen, square); });
        pos.king.Iterate([&](uint32_t square)    INLINE_LAMBDA { hash ^= GetPieceZobristHash(color, Piece::King, square); });
    }

    if (mEnPassantSquare.IsValid())
    {
        hash ^= GetEnPassantFileZobristHash(mEnPassantSquare.File());
    }

    for (uint32_t i = 0; i < 8; ++i)
    {
        if (GetWhitesCastlingRights() & (1 << i))
        {
            hash ^= GetCastlingRightsZobristHash(White, i);
        }
        if (GetBlacksCastlingRights() & (1 << i))
        {
            hash ^= GetCastlingRightsZobristHash(Black, i);
        }
    }

    hash ^= GetHalfMoveZobristHash(mHalfMoveCount);

    return hash;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

Position::Position()
    : mSideToMove(White)
    , mEnPassantSquare(Square::Invalid())
    , mCastlingRights{0,0}
    , mHalfMoveCount(0u)
    , mMoveCount(1u)
    , mHash(0u)
    , mPawnsHash(0u)
    , mNonPawnsHash{0u,0u}
{}

void Position::SetPiece(const Square square, const Piece piece, const Color color)
{
    ASSERT(square.IsValid());
    ASSERT((uint8_t)piece <= (uint8_t)Piece::King);
    ASSERT(color == White || color == Black);

    const Bitboard mask = square.GetBitboard();
    SidePosition& pos = GetSide(color);

    ASSERT((pos.pawns & mask) == 0);
    ASSERT((pos.knights & mask) == 0);
    ASSERT((pos.bishops & mask) == 0);
    ASSERT((pos.rooks & mask) == 0);
    ASSERT((pos.queens & mask) == 0);
    ASSERT((pos.king & mask) == 0);
    ASSERT(pos.pieces[square.Index()] == Piece::None);

    const uint64_t pieceHash = GetPieceZobristHash(color, piece, square.Index());
    mHash ^= pieceHash;
    if (piece == Piece::Pawn)
        mPawnsHash ^= pieceHash;
    else if (color == White)
        mNonPawnsHash[White] ^= (uint32_t)pieceHash;
    else
        mNonPawnsHash[Black] ^= (uint32_t)pieceHash;

    pos.GetPieceBitBoard(piece) |= mask;
    pos.pieces[square.Index()] = piece;
}

void Position::RemovePiece(const Square square, const Piece piece, const Color color)
{
    const Bitboard mask = square.GetBitboard();
    SidePosition& pos = GetSide(color);
    Bitboard& targetBitboard = pos.GetPieceBitBoard(piece);

    ASSERT((targetBitboard & mask) == mask);
    targetBitboard &= ~mask;

    ASSERT(pos.pieces[square.Index()] == piece);
    pos.pieces[square.Index()] = Piece::None;

    const uint64_t pieceHash = GetPieceZobristHash(color, piece, square.Index());
    mHash ^= pieceHash;
    if (piece == Piece::Pawn)
        mPawnsHash ^= pieceHash;
    else if (color == White)
        mNonPawnsHash[White] ^= (uint32_t)pieceHash;
    else
        mNonPawnsHash[Black] ^= (uint32_t)pieceHash;
}

uint64_t Position::HashAfterMove(const Move move) const
{
    ASSERT(move.IsValid());

    uint64_t hash = mHash ^ c_SideToMoveZobristHash;

    hash ^= GetPieceZobristHash(mSideToMove, move.GetPiece(), move.FromSquare().Index());
    hash ^= GetPieceZobristHash(mSideToMove, move.GetPiece(), move.ToSquare().Index());

    if (move.IsCapture() && !move.IsEnPassant())
    {
        const Piece capturedPiece = GetOpponentSide().GetPieceAtSquare(move.ToSquare());
        hash ^= GetPieceZobristHash(mSideToMove ^ 1, capturedPiece, move.ToSquare().Index());
    }

    return hash;
}

void Position::SetSideToMove(Color color)
{
    ASSERT(color == White || color == Black);

    if (mSideToMove != color)
    {
        mHash ^= c_SideToMoveZobristHash;
        mSideToMove = color;
    }
}

void Position::SetCastlingRights(Color color, uint8_t rightsMask)
{
    ASSERT(PopCount(rightsMask) <= 2);

    if (const uint8_t difference = mCastlingRights[(uint32_t)color] ^ rightsMask)
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (difference & (1 << i))
            {
                mHash ^= GetCastlingRightsZobristHash(color, i);
            }
        }

        mCastlingRights[(uint32_t)color] = rightsMask;
    }
}

void Position::SetEnPassantSquare(const Square square)
{
    if (mEnPassantSquare != square)
    {
        uint64_t hashDiff = 0;

        if (mEnPassantSquare.IsValid())
        {
            hashDiff = GetEnPassantFileZobristHash(mEnPassantSquare.File());
        }
        if (square.IsValid())
        {
            hashDiff ^= GetEnPassantFileZobristHash(square.File());
        }

        mHash ^= hashDiff;
        mEnPassantSquare = square;
    }
}

void Position::ClearEnPassantSquare()
{
    if (mEnPassantSquare.IsValid())
    {
        mHash ^= GetEnPassantFileZobristHash(mEnPassantSquare.File());
    }

    mEnPassantSquare = Square::Invalid();
}

void Position::SetHalfMoveCount(uint16_t halfMoveCount)
{
    //if (mHalfMoveCount != halfMoveCount)
    {
        mHash ^= GetHalfMoveZobristHash(mHalfMoveCount);
        mHash ^= GetHalfMoveZobristHash(halfMoveCount);
        mHalfMoveCount = halfMoveCount;
    }
}

Bitboard Position::GetAttackedSquares(Color side) const
{
    const SidePosition& currentSide = mColors[(uint8_t)side];
    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    Bitboard bitboard{ 0 };

    if (currentSide.pawns)
    {
        if (side == White)
        {
            bitboard |= Bitboard::GetPawnsAttacks<White>(currentSide.pawns);
        }
        else
        {
            bitboard |= Bitboard::GetPawnsAttacks<Black>(currentSide.pawns);
        }
    }

    currentSide.knights.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        bitboard |= Bitboard::GetKnightAttacks(Square(fromIndex));
    });

    const Bitboard rooks = currentSide.rooks | currentSide.queens;
    const Bitboard bishops = currentSide.bishops | currentSide.queens;

    rooks.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        bitboard |= Bitboard::GenerateRookAttacks(Square(fromIndex), occupiedSquares);
    });

    bishops.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        bitboard |= Bitboard::GenerateBishopAttacks(Square(fromIndex), occupiedSquares);
    });

    bitboard |= Bitboard::GetKingAttacks(Square(FirstBitSet(currentSide.king)));

    return bitboard;
}

Square Position::GetLongCastleRookSquare(const Square kingSquare, uint8_t castlingRights)
{
    constexpr uint8_t mask[] = { 0b00000000, 0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111 };
    const uint32_t longCastleMask = castlingRights & mask[kingSquare.File()];
    if (longCastleMask)
    {
        ASSERT(PopCount(longCastleMask) == 1);
        const uint8_t longCastleBitIndex = (uint8_t)FirstBitSet(longCastleMask);
        return Square(longCastleBitIndex, kingSquare.Rank());
    }

    return Square::Invalid();
}

Square Position::GetShortCastleRookSquare(const Square kingSquare, uint8_t castlingRights)
{
    constexpr uint8_t mask[] = { 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000, 0b00000000 };
    const uint32_t shortCastleMask = castlingRights & mask[kingSquare.File()];
    if (shortCastleMask)
    {
        ASSERT(PopCount(shortCastleMask) == 1);
        const uint8_t shortCastleBitIndex = (uint8_t)FirstBitSet(shortCastleMask);
        return Square(shortCastleBitIndex, kingSquare.Rank());
    }

    return Square::Invalid();
}

const Bitboard Position::GetAttackers(const Square square, const Bitboard occupied) const
{
    const Bitboard knights  = Whites().knights | Blacks().knights;
    const Bitboard bishops  = Whites().bishops | Blacks().bishops;
    const Bitboard rooks    = Whites().rooks | Blacks().rooks;
    const Bitboard queens   = Whites().queens | Blacks().queens;
    const Bitboard kings    = Whites().king | Blacks().king;

    Bitboard bitboard       = Bitboard::GetKingAttacks(square) & kings;
    if (knights)            bitboard |= Bitboard::GetKnightAttacks(square) & knights;
    if (rooks | queens)     bitboard |= Bitboard::GenerateRookAttacks(square, occupied) & (rooks | queens);
    if (bishops | queens)   bitboard |= Bitboard::GenerateBishopAttacks(square, occupied) & (bishops | queens);
    if (Whites().pawns)     bitboard |= Bitboard::GetPawnAttacks(square, Black) & Whites().pawns;
    if (Blacks().pawns)     bitboard |= Bitboard::GetPawnAttacks(square, White) & Blacks().pawns;

    return bitboard;
}

const Bitboard Position::GetAttackers(const Square square, const Color color) const
{
    const SidePosition& side = GetSide(color);
    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    Bitboard bitboard = Bitboard::GetKingAttacks(square) & side.king;

    if (side.knights)               bitboard |= Bitboard::GetKnightAttacks(square) & side.knights;
    if (side.rooks | side.queens)   bitboard |= Bitboard::GenerateRookAttacks(square, occupiedSquares) & (side.rooks | side.queens);
    if (side.bishops | side.queens) bitboard |= Bitboard::GenerateBishopAttacks(square, occupiedSquares) & (side.bishops | side.queens);
    if (side.pawns)                 bitboard |= Bitboard::GetPawnAttacks(square, color ^ 1) & side.pawns;

    return bitboard;
}

bool Position::IsSquareVisible(const Square square, const Color color) const
{
    const SidePosition& side = GetSide(color);

    if (Bitboard::GetKingAttacks(square) & side.king) return true;
    if (Bitboard::GetKnightAttacks(square) & side.knights) return true;
    if (Bitboard::GetPawnAttacks(square, color ^ 1) & side.pawns) return true;

    const Bitboard potentialBishopAttacks = Bitboard::GetBishopAttacks(square) & (side.bishops | side.queens);
    const Bitboard potentialRookAttacks = Bitboard::GetRookAttacks(square) & (side.rooks | side.queens);
    if (potentialBishopAttacks || potentialRookAttacks)
    {
        const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();
        if (potentialBishopAttacks && Bitboard::GenerateBishopAttacks(square, occupiedSquares) & potentialBishopAttacks) return true;
        if (potentialRookAttacks && Bitboard::GenerateRookAttacks(square, occupiedSquares) & potentialRookAttacks) return true;
    }

    return false;
}

bool Position::IsInCheck() const
{
    const SidePosition& currentSide = GetCurrentSide();

    const uint32_t kingSquareIndex = FirstBitSet(currentSide.king);
    return IsSquareVisible(Square(kingSquareIndex), mSideToMove ^ 1);
}

bool Position::IsInCheck(Color color) const
{
    const SidePosition& currentSide = GetSide(color);

    const uint32_t kingSquareIndex = FirstBitSet(currentSide.king);
    return IsSquareVisible(Square(kingSquareIndex), color ^ 1);
}

bool Position::GivesCheck_Approx(const Move move) const
{
    ASSERT(move.IsValid());

    const Bitboard kingBitboard = GetOpponentSide().king;
    const Square kingSq(FirstBitSet(kingBitboard));

    if ((move.GetPiece() == Piece::Knight) &&
        (Bitboard::GetKnightAttacks(move.ToSquare()) & kingBitboard))
    {
        return true;
    }

    if ((move.GetPiece() == Piece::Pawn) &&
        (Bitboard::GetPawnAttacks(move.ToSquare(), mSideToMove) & kingBitboard))
    {
        return true;
    }

    if (move.GetPiece() == Piece::Rook || move.GetPiece() == Piece::Queen)
    {
        if (move.ToSquare().File() == kingSq.File() ||
            move.ToSquare().Rank() == kingSq.Rank())
        {
            if ((Bitboard::GetBetween(kingSq, move.ToSquare()) & Occupied()) == 0)
            {
                return true;
            }
        }
    }

    if (move.GetPiece() == Piece::Bishop || move.GetPiece() == Piece::Queen)
    {
        if (move.ToSquare().Diagonal() == kingSq.Diagonal() ||
            move.ToSquare().AntiDiagonal() == kingSq.AntiDiagonal())
        {
            if ((Bitboard::GetBetween(kingSq, move.ToSquare()) & Occupied()) == 0)
            {
                return true;
            }
        }
    }

    // TODO discovered attacks

    return false;
}

uint32_t Position::GetNumLegalMoves(std::vector<Move>* outMoves) const
{
    MoveList moves;
    GenerateMoveList(*this, Bitboard::GetKingAttacks(GetOpponentSide().GetKingSquare()), moves);

    if (moves.Size() == 0)
    {
        return 0;
    }

    uint32_t numLegalMoves = 0u;
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.GetMove(i);
        ASSERT(move.IsValid());

        Position childPosition = *this;
        if (childPosition.DoMove(move))
        {
            numLegalMoves++;

            if (outMoves)
            {
                outMoves->push_back(move);
            }
        }
    }

    return numLegalMoves;
}

bool Position::IsMate() const
{
    return IsInCheck(mSideToMove) && (GetNumLegalMoves() == 0u);
}

bool Position::IsStalemate() const
{
    return !IsInCheck(mSideToMove) && (GetNumLegalMoves() == 0u);
}

bool Position::IsFiftyMoveRuleDraw() const
{
    if (mHalfMoveCount >= 100)
    {
        if (IsInCheck())
        {
            return GetNumLegalMoves() > 0u;
        }
        return true;
    }
    return false;
}

bool Position::IsMoveLegal(const Move& move) const
{
    ASSERT(IsMoveValid(move));

    Position positionAfterMove{ *this };
    return positionAfterMove.DoMove(move);
}

Piece Position::GetCapturedPiece(const Move move) const
{
    return move.IsEnPassant() ? Piece::Pawn : GetOpponentSide().GetPieceAtSquare(move.ToSquare());
}

Square Position::ExtractEnPassantSquareFromMove(const Move& move) const
{
    ASSERT(move.GetPiece() == Piece::Pawn);

    const Bitboard oponentPawns = GetOpponentSide().pawns;
    const Square from = move.FromSquare();
    const Square to = move.ToSquare();

    if (from.Rank() == 1u && to.Rank() == 3u)
    {
        ASSERT(from.File() == to.File());
        ASSERT(mSideToMove == White);

        if ((to.File() > 0 && (to.West_Unsafe().GetBitboard() & oponentPawns)) ||
            (to.File() < 7 && (to.East_Unsafe().GetBitboard() & oponentPawns)))
        {
            return Square(move.FromSquare().File(), 2u);
        }
    }

    if (from.Rank() == 6u && to.Rank() == 4u)
    {
        ASSERT(from.File() == to.File());
        ASSERT(mSideToMove == Black);

        if ((to.File() > 0 && (to.West_Unsafe().GetBitboard() & oponentPawns)) ||
            (to.File() < 7 && (to.East_Unsafe().GetBitboard() & oponentPawns)))
        {
            return Square(from.File(), 5u);
        }
    }

    return Square::Invalid();
}

void Position::ClearRookCastlingRights(const Square affectedSquare)
{
    if (affectedSquare.Rank() == 0)
    {
        if (mCastlingRights[0] & (1 << affectedSquare.File()))
        {
            mHash ^= GetCastlingRightsZobristHash(White, affectedSquare.File());
            mCastlingRights[0] &= ~(1 << affectedSquare.File());
        }
    }
    else if (affectedSquare.Rank() == 7)
    {
        if (mCastlingRights[1] & (1 << affectedSquare.File()))
        {
            mHash ^= GetCastlingRightsZobristHash(Black, affectedSquare.File());
            mCastlingRights[1] &= ~(1 << affectedSquare.File());
        }
    }
}

bool Position::DoMove(const Move& move, NNEvaluatorContext& nnContext)
{
    ASSERT(IsMoveValid(move));  // move must be valid
    ASSERT(IsValid());          // board position must be valid

    // move piece & mark NN accumulator as dirty
    {
        RemovePiece(move.FromSquare(), move.GetPiece(), mSideToMove);

        nnContext.MarkAsDirty();
        nnContext.dirtyPieces[0] = { move.GetPiece(), mSideToMove, move.FromSquare(), move.ToSquare() };
        nnContext.numDirtyPieces = 1;
    }

    // remove captured piece
    if (move.IsCapture())
    {
        if (move.IsEnPassant()) [[unlikely]]
        {
            Square captureSquare = Square::Invalid();
            if (move.ToSquare().Rank() == 5)  captureSquare = Square(move.ToSquare().File(), 4u);
            if (move.ToSquare().Rank() == 2)  captureSquare = Square(move.ToSquare().File(), 3u);
            ASSERT(captureSquare.IsValid());

            RemovePiece(captureSquare, Piece::Pawn, mSideToMove ^ 1);

            nnContext.dirtyPieces[nnContext.numDirtyPieces++] = { Piece::Pawn, (Color)(mSideToMove ^ 1), captureSquare, Square::Invalid() };
        }
        else // regular piece capture
        {
            const Piece capturedPiece = GetOpponentSide().GetPieceAtSquare(move.ToSquare());
            const Color capturedColor = mSideToMove ^ 1;
            RemovePiece(move.ToSquare(), capturedPiece, capturedColor);

            nnContext.dirtyPieces[nnContext.numDirtyPieces++] = { capturedPiece, capturedColor, move.ToSquare(), Square::Invalid() };

            if (capturedPiece == Piece::Rook)
            {
                // clear specific castling right after capturing a rook
                ClearRookCastlingRights(move.ToSquare());
            }
        }
    }

    // put moved piece
    if (!move.IsCastling()) [[likely]]
    {
        const bool isPromotion = move.GetPromoteTo() != Piece::None;
        const Piece targetPiece = isPromotion ? move.GetPromoteTo() : move.GetPiece();
        SetPiece(move.ToSquare(), targetPiece, mSideToMove);

        if (isPromotion)
        {
            ASSERT(move.GetPiece() == Piece::Pawn);
            nnContext.dirtyPieces[0].toSquare = Square::Invalid();
            nnContext.dirtyPieces[nnContext.numDirtyPieces++] = { targetPiece, mSideToMove, Square::Invalid(), move.ToSquare() };
        }
    }

    SetEnPassantSquare(move.GetPiece() == Piece::Pawn ? ExtractEnPassantSquareFromMove(move) : Square::Invalid());

    if (move.GetPiece() == Piece::King)
    {
        if (move.IsCastling()) [[unlikely]]
        {
            const uint8_t currentSideCastlingRights = mCastlingRights[(uint32_t)mSideToMove];

            ASSERT(currentSideCastlingRights != 0);
            ASSERT(move.FromSquare().Rank() == 0 || move.FromSquare().Rank() == 7);
            ASSERT(move.FromSquare().Rank() == move.ToSquare().Rank());

            const Square oldKingSquare = move.FromSquare();
            Square oldRookSquare = Square::Invalid();
            Square newRookSquare = Square::Invalid();
            Square newKingSquare = Square::Invalid();

            if (move.IsShortCastle())
            {
                oldRookSquare = GetShortCastleRookSquare(oldKingSquare, currentSideCastlingRights);
                newRookSquare = Square(5u, move.FromSquare().Rank());
                newKingSquare = Square(6u, move.FromSquare().Rank());
            }
            else if (move.IsLongCastle())
            {
                oldRookSquare = GetLongCastleRookSquare(oldKingSquare, currentSideCastlingRights);
                newRookSquare = Square(3u, move.FromSquare().Rank());
                newKingSquare = Square(2u, move.FromSquare().Rank());
            }
            else // invalid castle
            {
                ASSERT(false);
            }

            RemovePiece(oldRookSquare, Piece::Rook, mSideToMove);
            SetPiece(newKingSquare, Piece::King, mSideToMove);
            SetPiece(newRookSquare, Piece::Rook, mSideToMove);

            ASSERT(nnContext.numDirtyPieces == 1);
            nnContext.dirtyPieces[0].toSquare = newKingSquare; // adjust king movement
            nnContext.dirtyPieces[nnContext.numDirtyPieces++] = { Piece::Rook, mSideToMove, oldRookSquare, newRookSquare }; // move the rook
        }

        // clear all castling rights after moving a king
        SetCastlingRights(mSideToMove, 0);
    }
    else if (move.GetPiece() == Piece::Rook)
    {
        // clear specific castling right after moving a rook
        ClearRookCastlingRights(move.FromSquare());
    }

    if (mSideToMove == Black)
        mMoveCount++;

    SetHalfMoveCount((move.GetPiece() == Piece::Pawn || move.IsCapture()) ? 0 : (mHalfMoveCount + 1));

    const Color prevToMove = mSideToMove;
    mSideToMove = mSideToMove ^ 1;
    mHash ^= c_SideToMoveZobristHash;

    // board position after the move must be valid
    ASSERT(IsValid());

    // validate hash
    ASSERT(ComputeHash() == GetHash());

    ASSERT(nnContext.numDirtyPieces > 0 && nnContext.numDirtyPieces <= MaxNumDirtyPieces);

    // can't be in check after move
    return !IsInCheck(prevToMove);
}

bool Position::DoMove(const Move& move)
{
    NNEvaluatorContext dummyContext;
    return DoMove(move, dummyContext);
}

bool Position::DoNullMove()
{
    ASSERT(IsValid());          // board position must be valid
    ASSERT(!IsInCheck(mSideToMove));

    SetEnPassantSquare(Square::Invalid());

    if (mSideToMove == Black)
    {
        mMoveCount++;
    }

    SetHalfMoveCount(mHalfMoveCount + 1);

    mSideToMove = mSideToMove ^ 1;
    mHash ^= c_SideToMoveZobristHash;

    ASSERT(IsValid());  // board position after the move must be valid

    // validate hash
    ASSERT(ComputeHash() == GetHash());

    return true;
}

Position Position::SwappedColors() const
{
    Position result;

    result.mColors[0].king          = mColors[1].king.MirroredVertically();
    result.mColors[0].queens        = mColors[1].queens.MirroredVertically();
    result.mColors[0].rooks         = mColors[1].rooks.MirroredVertically();
    result.mColors[0].bishops       = mColors[1].bishops.MirroredVertically();
    result.mColors[0].knights       = mColors[1].knights.MirroredVertically();
    result.mColors[0].pawns         = mColors[1].pawns.MirroredVertically();

    result.mColors[1].king          = mColors[0].king.MirroredVertically();
    result.mColors[1].queens        = mColors[0].queens.MirroredVertically();
    result.mColors[1].rooks         = mColors[0].rooks.MirroredVertically();
    result.mColors[1].bishops       = mColors[0].bishops.MirroredVertically();
    result.mColors[1].knights       = mColors[0].knights.MirroredVertically();
    result.mColors[1].pawns         = mColors[0].pawns.MirroredVertically();

    // flip pieces
    for (uint32_t rank = 0; rank < 8; ++rank)
    {
        for (uint32_t file = 0; file < 8; ++file)
        {
            result.mColors[1].pieces[rank * 8 + file] = mColors[0].pieces[(7 - rank) * 8 + file];
            result.mColors[0].pieces[rank * 8 + file] = mColors[1].pieces[(7 - rank) * 8 + file];
        }
    }

    result.mCastlingRights[0]       = mCastlingRights[1];
    result.mCastlingRights[1]       = mCastlingRights[0];
    result.mSideToMove              = mSideToMove ^ 1;
    result.mMoveCount               = mMoveCount;
    result.mHalfMoveCount           = mHalfMoveCount;
    result.mHash                    = 0;
    result.mPawnsHash               = 0;
    result.mNonPawnsHash[0]         = 0;
    result.mNonPawnsHash[1]         = 0;

    return result;
}

void Position::MirrorVertically()
{
    mColors[0].king     = mColors[0].king.MirroredVertically();
    mColors[0].queens   = mColors[0].queens.MirroredVertically();
    mColors[0].rooks    = mColors[0].rooks.MirroredVertically();
    mColors[0].bishops  = mColors[0].bishops.MirroredVertically();
    mColors[0].knights  = mColors[0].knights.MirroredVertically();
    mColors[0].pawns    = mColors[0].pawns.MirroredVertically();

    mColors[1].king     = mColors[1].king.MirroredVertically();
    mColors[1].queens   = mColors[1].queens.MirroredVertically();
    mColors[1].rooks    = mColors[1].rooks.MirroredVertically();
    mColors[1].bishops  = mColors[1].bishops.MirroredVertically();
    mColors[1].knights  = mColors[1].knights.MirroredVertically();
    mColors[1].pawns    = mColors[1].pawns.MirroredVertically();

    mCastlingRights[0] = 0;
    mCastlingRights[1] = 0;

    mHash = ComputeHash();
    mPawnsHash = 0; // TODO
    mNonPawnsHash[0] = 0; // TODO
    mNonPawnsHash[1] = 0; // TODO
}

void Position::MirrorHorizontally()
{
    mColors[0].king     = mColors[0].king.MirroredHorizontally();
    mColors[0].queens   = mColors[0].queens.MirroredHorizontally();
    mColors[0].rooks    = mColors[0].rooks.MirroredHorizontally();
    mColors[0].bishops  = mColors[0].bishops.MirroredHorizontally();
    mColors[0].knights  = mColors[0].knights.MirroredHorizontally();
    mColors[0].pawns    = mColors[0].pawns.MirroredHorizontally();

    mColors[1].king     = mColors[1].king.MirroredHorizontally();
    mColors[1].queens   = mColors[1].queens.MirroredHorizontally();
    mColors[1].rooks    = mColors[1].rooks.MirroredHorizontally();
    mColors[1].bishops  = mColors[1].bishops.MirroredHorizontally();
    mColors[1].knights  = mColors[1].knights.MirroredHorizontally();
    mColors[1].pawns    = mColors[1].pawns.MirroredHorizontally();

    mCastlingRights[0] = ReverseBits(mCastlingRights[0]);
    mCastlingRights[1] = ReverseBits(mCastlingRights[1]);

    mHash = ComputeHash();
    mPawnsHash = 0; // TODO
    mNonPawnsHash[0] = 0; // TODO
    mNonPawnsHash[1] = 0; // TODO
}

void Position::FlipDiagonally()
{
    mColors[0].king     = mColors[0].king.FlippedDiagonally();
    mColors[0].queens   = mColors[0].queens.FlippedDiagonally();
    mColors[0].rooks    = mColors[0].rooks.FlippedDiagonally();
    mColors[0].bishops  = mColors[0].bishops.FlippedDiagonally();
    mColors[0].knights  = mColors[0].knights.FlippedDiagonally();
    mColors[0].pawns    = mColors[0].pawns.FlippedDiagonally();

    mColors[1].king     = mColors[1].king.FlippedDiagonally();
    mColors[1].queens   = mColors[1].queens.FlippedDiagonally();
    mColors[1].rooks    = mColors[1].rooks.FlippedDiagonally();
    mColors[1].bishops  = mColors[1].bishops.FlippedDiagonally();
    mColors[1].knights  = mColors[1].knights.FlippedDiagonally();
    mColors[1].pawns    = mColors[1].pawns.FlippedDiagonally();

    mCastlingRights[0] = 0;
    mCastlingRights[1] = 0;

    mHash = ComputeHash();
    mPawnsHash = 0; // TODO
    mNonPawnsHash[0] = 0; // TODO
    mNonPawnsHash[1] = 0; // TODO
}

Position Position::MirroredVertically() const
{
    Position ret = *this;
    ret.MirrorVertically();
    return ret;
}

Position Position::MirroredHorizontally() const
{
    Position ret = *this;
    ret.MirrorHorizontally();
    return ret;
}

bool Position::HasNonPawnMaterial(Color color) const
{
    const SidePosition& side = GetSide(color);

    return
        side.queens     != 0 ||
        side.rooks      != 0 ||
        side.bishops    != 0 ||
        side.knights    != 0;
}

const MaterialKey Position::GetMaterialKey() const
{
    MaterialKey key;

    key.numWhiteQueens = mColors[0].queens.Count();
    key.numWhiteRooks = mColors[0].rooks.Count();
    key.numWhiteBishops = mColors[0].bishops.Count();
    key.numWhiteKnights = mColors[0].knights.Count();
    key.numWhitePawns = mColors[0].pawns.Count();

    key.numBlackQueens = mColors[1].queens.Count();
    key.numBlackRooks = mColors[1].rooks.Count();
    key.numBlackBishops = mColors[1].bishops.Count();
    key.numBlackKnights = mColors[1].knights.Count();
    key.numBlackPawns = mColors[1].pawns.Count();

    return key;
}

static const int32_t pawnValue = 100;
static const int32_t knightValue = 300;
static const int32_t bishopValue = 300;
static const int32_t rookValue = 500;
static const int32_t queenValue = 900;
static const int32_t kingValue = INT32_MAX;

int32_t Position::BestPossibleMoveValue() const
{
    int32_t value = 0;

    const SidePosition& side = GetOpponentSide();

    // can capture most valuable piece
         if (side.queens)   value = std::max(c_queenValue.mg, c_queenValue.eg);
    else if (side.rooks)    value = std::max(c_rookValue.mg, c_rookValue.eg);
    else if (side.knights)  value = std::max(c_knightValue.mg, c_knightValue.eg);
    else if (side.bishops)  value = std::max(c_bishopValue.mg, c_bishopValue.eg);
    else if (side.pawns)    value = std::max(c_pawnValue.mg, c_pawnValue.eg);

    // can promote to queen
    if (GetCurrentSide().pawns & (mSideToMove == White ? Bitboard::RankBitboard<6>() : Bitboard::RankBitboard<1>()))
    {
        value += std::max(c_queenValue.mg, c_queenValue.eg) - std::min(c_pawnValue.mg, c_pawnValue.eg);
    }

    return value;
}

static const int32_t c_seePieceValues[] =
{
    0, // none
    pawnValue,
    knightValue,
    bishopValue,
    rookValue,
    queenValue,
    kingValue,
};

bool Position::StaticExchangeEvaluation(const Move& move, int32_t treshold) const
{
    const Square toSquare = move.ToSquare();
    const Square fromSquare = move.FromSquare();

    int32_t balance = -treshold;

    if (move.IsCapture())
    {
        const Piece capturedPiece = GetCapturedPiece(move);
        balance += c_seePieceValues[(uint32_t)capturedPiece];
        if (balance < 0) return false;
    }

    {
        ASSERT(move.GetPiece() == GetCurrentSide().GetPieceAtSquare(fromSquare));
        balance = c_seePieceValues[(uint32_t)move.GetPiece()] - balance;
        if (balance <= 0) return true;
    }

    const Bitboard whiteOccupied = Whites().Occupied();
    const Bitboard blackOccupied = Blacks().Occupied();
    Bitboard occupied = whiteOccupied | blackOccupied;

    // "do" move
    occupied &= ~fromSquare.GetBitboard();
    occupied |= toSquare.GetBitboard();

    const Bitboard bishopsAndQueens = Whites().bishops | Blacks().bishops | Whites().queens | Blacks().queens;
    const Bitboard rooksAndQueens = Whites().rooks | Blacks().rooks | Whites().queens | Blacks().queens;

    Bitboard allAttackers = GetAttackers(toSquare, occupied);

    Color sideToMove = mSideToMove;
    int32_t result = 1;

    for (;;)
    {
        sideToMove ^= 1;
        allAttackers &= occupied;

        const SidePosition& side = GetSide(sideToMove);
        const Bitboard ourAttackers = allAttackers & (sideToMove == White ? whiteOccupied : blackOccupied);
        const Bitboard theirAttackers = allAttackers & (sideToMove == White ? blackOccupied : whiteOccupied);

        // no more attackers - side to move loses
        if (ourAttackers == 0) break;

        result ^= 1;

        // TODO filter out pinned pieces

        // find attacking piece
        Piece piece = Piece::Pawn;
        for (; piece != Piece::King; piece = NextPiece(piece))
            if (side.GetPieceBitBoard(piece) & ourAttackers)
                break;

        if (piece == Piece::King)
        {
            // if capturing with the king, but opponent still has attacker, return the result (can't be in check)
            if (theirAttackers) result ^= 1;
            break;
        }

        balance = c_seePieceValues[(uint32_t)piece] - balance;
        if (balance < result) break;

        // remove one attacker from occupied squares
        occupied ^= (1ull << FirstBitSet(side.GetPieceBitBoard(piece) & ourAttackers));

        // update diagonal attackers
        if (piece == Piece::Pawn || piece == Piece::Bishop || piece == Piece::Queen)
            allAttackers |= Bitboard::GenerateBishopAttacks(toSquare, occupied) & bishopsAndQueens;

        // update horizontal/vertical attackers
        if (piece == Piece::Rook || piece == Piece::Queen)
            allAttackers |= Bitboard::GenerateRookAttacks(toSquare, occupied) & rooksAndQueens;
    }

    return result != 0;
}

void Position::ComputeThreats(Threats& outThreats) const
{
    Bitboard attackedByPawns = 0;
    Bitboard attackedByMinors = 0;
    Bitboard attackedByRooks = 0;
    Bitboard allThreats = 0;

    const SidePosition& opponentSide = GetOpponentSide();
    const Bitboard occupied = Occupied();

    attackedByPawns = Bitboard::GetPawnsAttacks(opponentSide.pawns, GetSideToMove());

    attackedByMinors = attackedByPawns |
        Bitboard::GetKnightAttacks(opponentSide.knights);
    opponentSide.bishops.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA {
        attackedByMinors |= Bitboard::GenerateBishopAttacks(Square(fromIndex), occupied); });

    attackedByRooks = attackedByMinors;
    opponentSide.rooks.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA {
        attackedByRooks |= Bitboard::GenerateRookAttacks(Square(fromIndex), occupied); });

    allThreats = attackedByRooks;
    opponentSide.queens.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA {
        allThreats |= Bitboard::GenerateQueenAttacks(Square(fromIndex), occupied); });
    allThreats |= Bitboard::GetKingAttacks(opponentSide.GetKingSquare());

    outThreats.attackedByPawns = attackedByPawns;
    outThreats.attackedByMinors = attackedByMinors;
    outThreats.attackedByRooks = attackedByRooks;
    outThreats.allThreats = allThreats;
}

bool Position::IsQuiet() const
{
    if (IsInCheck(mSideToMove))
    {
        return false;
    }

    MoveList moves;
    GenerateMoveList<MoveGenerationMode::Captures>(*this, Bitboard::GetKingAttacks(GetOpponentSide().GetKingSquare()), moves);

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.GetMove(i);

        Position posCopy = *this;
        if (!posCopy.DoMove(move))
        {
            continue;
        }

        if (StaticExchangeEvaluation(move, 0))
        {
            return false;
        }
    }

    return true;
}