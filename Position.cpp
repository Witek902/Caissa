#include "Position.hpp"
#include "Move.hpp"
#include "Bitboard.hpp"

SidePosition::SidePosition()
    : castlingRights(CastlingRights_All)
{}

void SidePosition::SetPieceAtSquare(const Square square, Piece piece)
{
    const uint64_t mask = square.Bitboard();

    pawns &= ~mask;
    knights &= ~mask;
    bishops &= ~mask;
    rooks &= ~mask;
    queens &= ~mask;
    king &= ~mask;

    switch (piece)
    {
    case Piece::Pawn:   pawns |= mask;      break;
    case Piece::Knight: knights |= mask;    break;
    case Piece::Bishop: bishops |= mask;    break;
    case Piece::Rook:   rooks |= mask;      break;
    case Piece::Queen:  queens |= mask;     break;
    case Piece::King:   king |= mask;       break;
    }
}

Piece SidePosition::GetPieceAtSquare(const Square square) const
{
    ASSERT(square.IsValid());

    const Bitboard squareBitboard = square.Bitboard();

    if (pawns & squareBitboard)     return Piece::Pawn;
    if (knights & squareBitboard)   return Piece::Knight;
    if (bishops & squareBitboard)   return Piece::Bishop;
    if (rooks & squareBitboard)     return Piece::Rook;
    if (queens & squareBitboard)    return Piece::Queen;
    if (king & squareBitboard)      return Piece::King;

    return Piece::None;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

Position::Position()
    : mSideToMove(Color::White)
    , mHalfMoveCount(0u)
    , mMoveCount(1u)
{}

void Position::SetPieceAtSquare(const Square square, Piece piece, Color color)
{
    ASSERT(square.IsValid());

    SidePosition& pos = color == Color::White ? mWhites : mBlacks;
    pos.SetPieceAtSquare(square, piece);
}

float Position::Evaluate() const
{
    return 0.0f;
}

Bitboard Position::GetAttackedSquares(Color side) const
{
    const SidePosition& currentSide = side == Color::White ? mWhites : mBlacks;
    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    Bitboard bitboard{ 0 };

    const int32_t pawnDirection = side == Color::White ? 1 : -1;

    currentSide.pawns.Iterate([&](uint32_t fromIndex)
    {
        const Square fromSquare(fromIndex);
        const Square squareForward(fromSquare.Index() + pawnDirection * 8); // next rank

        // there should be no pawn in first or last rank
        ASSERT(fromSquare.Rank() > 0u && fromSquare.Rank() < 7u);

        // capture on the left
        if (fromSquare.File() > 0u)
        {
            bitboard |= Square(fromSquare.Index() + pawnDirection * 8 - 1).Bitboard();
        }

        // capture on the right
        if (fromSquare.File() < 7u)
        {
            bitboard |= Square(fromSquare.Index() + pawnDirection * 8 + 1).Bitboard();
        }
    });

    currentSide.knights.Iterate([&](uint32_t fromIndex)
    {
        bitboard |= Bitboard::GetKnightAttacks(Square(fromIndex));
    });

    currentSide.rooks.Iterate([&](uint32_t fromIndex)
    {
        bitboard |= Bitboard::GenerateRookAttacks(Square(fromIndex), occupiedSquares);
    });

    currentSide.bishops.Iterate([&](uint32_t fromIndex)
    {
        bitboard |= Bitboard::GenerateBishopAttacks(Square(fromIndex), occupiedSquares);
    });

    currentSide.queens.Iterate([&](uint32_t fromIndex)
    {
        bitboard |= Bitboard::GenerateRookAttacks(Square(fromIndex), occupiedSquares);
        bitboard |= Bitboard::GenerateBishopAttacks(Square(fromIndex), occupiedSquares);
    });

    {
        unsigned long kingSquareIndex;
        if (_BitScanForward64(&kingSquareIndex, currentSide.king))
        {
            bitboard |= Bitboard::GetKingAttacks(Square(kingSquareIndex));
        }
    }

    return bitboard;
}

void Position::GenerateMoveList(MoveList& outMoveList) const
{
    outMoveList.mNumMoves = 0u;

    GeneratePawnMoveList(outMoveList);
    GenerateKnightMoveList(outMoveList);
    GenerateRookMoveList(outMoveList);
    GenerateBishopMoveList(outMoveList);
    GenerateQueenMoveList(outMoveList);
    GenerateKingMoveList(outMoveList);
}

void Position::GeneratePawnMoveList(MoveList& outMoveList) const
{
    const int32_t pawnDirection = mSideToMove == Color::White ? 1 : -1;
    const SidePosition& currentSide  = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;
    const uint32_t pawnStartingRank = mSideToMove == Color::White ? 1u : 6u;
    const uint32_t enPassantRank = mSideToMove == Color::White ? 5u : 2u;
    const uint32_t pawnFinalRank = mSideToMove == Color::White ? 6u : 1u;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    const auto generatePawnMove = [&outMoveList, pawnFinalRank](const Square fromSquare, const Square toSquare, bool isCapture, bool enPassant)
    {
        if (fromSquare.Rank() == pawnFinalRank)
        {
            // TODO promotion to rook/bishop should have very low priority when sorting moves
            // or not generate it at all if it wouldn't lead to stallmate?
            const Piece promotionList[] = { Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop };
            for (const Piece promoteTo : promotionList)
            {
                Move& move = outMoveList.PushMove();
                move.fromSquare = fromSquare;
                move.toSquare = toSquare;
                move.piece = Piece::Pawn;
                move.promoteTo = promoteTo;
                move.isCapture = isCapture;
                move.isEnPassant = enPassant;
            }
        }
        else // simple move forward
        {
            Move& move = outMoveList.PushMove();
            move.fromSquare = fromSquare;
            move.toSquare = toSquare;
            move.piece = Piece::Pawn;
            move.promoteTo = Piece::None;
            move.isCapture = isCapture;
            move.isEnPassant = enPassant;
        }
    };

    currentSide.pawns.Iterate([&](uint32_t fromIndex)
    {
        const Square fromSquare(fromIndex);
        const Square squareForward(fromSquare.Index() + pawnDirection * 8); // next rank

        // there should be no pawn in first or last rank
        ASSERT(fromSquare.Rank() > 0u && fromSquare.Rank() < 7u);

        // can move forward only to non-occupied squares
        if ((occupiedSquares & squareForward.Bitboard()) == 0u)
        {
            generatePawnMove(fromSquare, squareForward, false, false);

            if (fromSquare.Rank() == pawnStartingRank) // move by two ranks
            {
                const Square twoSquaresForward(fromSquare.Index() + pawnDirection * 16); // two ranks up

                // can move forward only to non-occupied squares
                if ((occupiedSquares & twoSquaresForward.Bitboard()) == 0u)
                {
                    Move& move = outMoveList.PushMove();
                    move.fromSquare = fromSquare;
                    move.toSquare = twoSquaresForward;
                    move.piece = Piece::Pawn;
                    move.promoteTo = Piece::None;
                    move.isCapture = false;
                    move.isEnPassant = false;
                }
            }
        }

        // capture on the left
        if (fromSquare.File() > 0u)
        {
            const Square toSquare(fromSquare.Index() + pawnDirection * 8 - 1);
            if (toSquare.Bitboard() & opponentSide.OccupiedExcludingKing())
            {
                generatePawnMove(fromSquare, toSquare, true, false);
            }
            if (toSquare == mEnPassantSquare && toSquare.Rank() == enPassantRank)
            {
                generatePawnMove(fromSquare, toSquare, true, true);
            }
        }

        // capture on the right
        if (fromSquare.File() < 7u)
        {
            const Square toSquare(fromSquare.Index() + pawnDirection * 8 + 1);
            if (toSquare.Bitboard() & opponentSide.OccupiedExcludingKing())
            {
                generatePawnMove(fromSquare, toSquare, true, false);
            }
            if (toSquare == mEnPassantSquare && toSquare.Rank() == enPassantRank)
            {
                generatePawnMove(fromSquare, toSquare, true, true);
            }
        }
    });
}

void Position::GenerateKnightMoveList(MoveList& outMoveList) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    currentSide.knights.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = Bitboard::GetKnightAttacks(square);
        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            // can't capture king
            if (opponentSide.king & targetSquare.Bitboard()) return;

            Move& move = outMoveList.PushMove();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Knight;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        });
    });
}

void Position::GenerateRookMoveList(MoveList& outMoveList) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    currentSide.rooks.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = Bitboard::GenerateRookAttacks(square, occupiedSquares);
        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move& move = outMoveList.PushMove();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Rook;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        });
    });
}

void Position::GenerateBishopMoveList(MoveList& outMoveList) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    currentSide.bishops.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = Bitboard::GenerateBishopAttacks(square, occupiedSquares);
        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move& move = outMoveList.PushMove();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Bishop;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        });
    });
}

void Position::GenerateQueenMoveList(MoveList& outMoveList) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    currentSide.queens.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard =
            Bitboard::GenerateRookAttacks(square, occupiedSquares) |
            Bitboard::GenerateBishopAttacks(square, occupiedSquares);

        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move& move = outMoveList.PushMove();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Queen;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        });
    });
}

void Position::GenerateKingMoveList(MoveList& outMoveList) const
{
    const uint32_t numKingOffsets = 8u;
    const int32_t kingFileOffsets[numKingOffsets] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const int32_t kingRankOffsets[numKingOffsets] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    unsigned long kingSquareIndex;
    if (0 == _BitScanForward64(&kingSquareIndex, currentSide.king))
    {
        return;
    }

    const Square square(kingSquareIndex);

    const Bitboard attackBitboard = Bitboard::GetKingAttacks(square);
    attackBitboard.Iterate([&](uint32_t toIndex)
    {
        const Square targetSquare(toIndex);

        // can't capture own piece
        if (currentSide.Occupied() & targetSquare.Bitboard()) return;

        Move& move = outMoveList.PushMove();
        move.fromSquare = square;
        move.toSquare = targetSquare;
        move.piece = Piece::King;
        move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        move.isCastling = false;
    });

    if (currentSide.castlingRights & CastlingRights_All)
    {
        const Bitboard opponentAttacks = GetAttackedSquares(GetOppositeColor(mSideToMove));

        // TODO simplify this
        const Bitboard longCastleKingCrossedSquares = (1ull << (kingSquareIndex - 1)) | (1ull << (kingSquareIndex - 2));
        const Bitboard shortCastleKingCrossedSquares = (1ull << (kingSquareIndex + 1)) | (1ull << (kingSquareIndex + 2));
        const Bitboard longCastleCrossedSquares = longCastleKingCrossedSquares | Bitboard(1ull << (kingSquareIndex - 3));
        const Bitboard shortCastleCrossedSquares = shortCastleKingCrossedSquares;

        // king can't be in check
        if ((currentSide.king & opponentAttacks) == 0u)
        {
            if ((currentSide.castlingRights & CastlingRights_LongCastleAllowed) &&
                ((occupiedSquares & longCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & longCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?

                Move& move = outMoveList.PushMove();
                move.fromSquare = square;
                move.toSquare = Square(2u, square.Rank());
                move.piece = Piece::King;
                move.isCapture = false;
                move.isCastling = true;
            }

            if ((currentSide.castlingRights & CastlingRights_ShortCastleAllowed) &&
                ((occupiedSquares & shortCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & shortCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?

                Move& move = outMoveList.PushMove();
                move.fromSquare = square;
                move.toSquare = Square(6u, square.Rank());
                move.piece = Piece::King;
                move.isCapture = false;
                move.isCastling = true;
            }
        }
    }
}

bool Position::IsInCheck(Color sideColor) const
{
    const SidePosition& currentSide = sideColor == Color::White ? mWhites : mBlacks;
    const Color opponentColor = sideColor == Color::White ? Color::Black : Color::White;

    const Bitboard attackedSquares = GetAttackedSquares(opponentColor);

    return currentSide.king & attackedSquares;
}

bool Position::IsMoveLegal(const Move& move) const
{
    ASSERT(IsMoveValid(move));

    Position positionAfterMove{ *this };
    positionAfterMove.DoMove(move);

    // can't be in check after move
    return !positionAfterMove.IsInCheck(mSideToMove);
}

static Square ExtractEnPassantSquareFromMove(const Move& move)
{
    ASSERT(move.piece == Piece::Pawn);

    if (move.fromSquare.Rank() == 1u && move.toSquare.Rank() == 3u)
    {
        ASSERT(move.fromSquare.File() == move.toSquare.File());
        return Square(move.fromSquare.File(), 2u);
    }

    if (move.fromSquare.Rank() == 6u && move.toSquare.Rank() == 4u)
    {
        ASSERT(move.fromSquare.File() == move.toSquare.File());
        return Square(move.fromSquare.File(), 5u);
    }

    return Square();
}

void Position::ClearRookCastlingRights(const Square affectedSquare)
{
    switch (affectedSquare.mIndex)
    {
    case Square_a1: mWhites.castlingRights = CastlingRights(mWhites.castlingRights & ~CastlingRights_LongCastleAllowed); break;
    case Square_h1: mWhites.castlingRights = CastlingRights(mWhites.castlingRights & ~CastlingRights_ShortCastleAllowed); break;
    case Square_a8: mBlacks.castlingRights = CastlingRights(mBlacks.castlingRights & ~CastlingRights_LongCastleAllowed); break;
    case Square_h8: mBlacks.castlingRights = CastlingRights(mBlacks.castlingRights & ~CastlingRights_ShortCastleAllowed); break;
    };
}

bool Position::DoMove(const Move& move)
{
    ASSERT(IsMoveValid(move));  // move must be valid
    ASSERT(IsValid());          // board position must be valid

    SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    // move piece
    Bitboard& pieceBitboard = currentSide.GetPieceBitBoard(move.piece);
    ASSERT(pieceBitboard & move.fromSquare.Bitboard()); // expected moved piece
    pieceBitboard &= ~move.fromSquare.Bitboard();

    // handle promotion by updating different bitboard
    const bool isPromotion = move.piece == Piece::Pawn && move.promoteTo != Piece::None;
    Bitboard& targetPieceBitboard = isPromotion ? currentSide.GetPieceBitBoard(move.promoteTo) : pieceBitboard;
    targetPieceBitboard |= move.toSquare.Bitboard();

    if (move.isCapture)
    {
        if (move.piece != Piece::Pawn || !move.isEnPassant)
        {
            ASSERT(opponentSide.Occupied() & move.toSquare.Bitboard());
        }
        opponentSide.SetPieceAtSquare(move.toSquare, Piece::None);

        // clear specific castling right after capturing a rook
        ClearRookCastlingRights(move.toSquare);
    }

    if (move.piece == Piece::Pawn && move.isEnPassant)
    {
        Square captureSquare;
        if (move.toSquare.Rank() == 5)  captureSquare = Square(move.toSquare.File(), 4u);
        if (move.toSquare.Rank() == 2)  captureSquare = Square(move.toSquare.File(), 3u);
        ASSERT(captureSquare.IsValid());

        ASSERT(opponentSide.pawns & captureSquare.Bitboard()); // expected pawn
        opponentSide.pawns &= ~captureSquare.Bitboard();
    }

    if (move.piece == Piece::Pawn)
    {
        mEnPassantSquare = ExtractEnPassantSquareFromMove(move);
    }
    else
    {
        mEnPassantSquare = Square();
    }

    if (move.piece == Piece::King)
    {
        if (move.isCastling)
        {
            Bitboard& rooksBitboard = currentSide.rooks;

            ASSERT(move.fromSquare.Rank() == 0 || move.fromSquare.Rank() == 7);
            ASSERT(move.fromSquare.Rank() == move.toSquare.Rank());

            // short castle
            if (move.fromSquare.File() == 4u && move.toSquare.File() == 6u)
            {
                const Square oldRookSquare(7u, move.fromSquare.Rank());
                const Square newRookSquare(5u, move.fromSquare.Rank());

                ASSERT((rooksBitboard & oldRookSquare.Bitboard()) == oldRookSquare.Bitboard());
                ASSERT((occupiedSquares & newRookSquare.Bitboard()) == 0u);

                rooksBitboard &= ~oldRookSquare.Bitboard();
                rooksBitboard |= newRookSquare.Bitboard();
            }
            // long castle
            else if (move.fromSquare.File() == 4u && move.toSquare.File() == 2u)
            {
                const Square oldRookSquare(0u, move.fromSquare.Rank());
                const Square newRookSquare(3u, move.fromSquare.Rank());

                ASSERT((rooksBitboard & oldRookSquare.Bitboard()) == oldRookSquare.Bitboard());
                ASSERT((occupiedSquares & newRookSquare.Bitboard()) == 0u);

                rooksBitboard &= ~oldRookSquare.Bitboard();
                rooksBitboard |= newRookSquare.Bitboard();
            }
            else // invalid castle
            {
                ASSERT(false);
            }
        }

        // clear all castling rights after moving a king
        currentSide.castlingRights = CastlingRights(0);
    }

    // clear specific castling right after moving a rook
    if (move.piece == Piece::Rook)
    {
        ClearRookCastlingRights(move.fromSquare);
    }

    ASSERT(IsValid());  // board position after the move must be valid

    const Color prevToMove = mSideToMove;

    mSideToMove = GetOppositeColor(mSideToMove);

    // can't be in check after move
    return !IsInCheck(prevToMove);
}