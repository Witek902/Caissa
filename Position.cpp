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

    uint32_t pieceIndex = 0;

    pieceIndex |= ((pawns >> square.Index()) & 1) * (uint32_t)Piece::Pawn;
    pieceIndex |= ((knights >> square.Index()) & 1) * (uint32_t)Piece::Knight;
    pieceIndex |= ((bishops >> square.Index()) & 1) * (uint32_t)Piece::Bishop;
    pieceIndex |= ((rooks >> square.Index()) & 1) * (uint32_t)Piece::Rook;
    pieceIndex |= ((queens >> square.Index()) & 1) * (uint32_t)Piece::Queen;
    pieceIndex |= ((king >> square.Index()) & 1) * (uint32_t)Piece::King;

    return (Piece)pieceIndex;
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

Bitboard Position::GetAttackedSquares(Color side) const
{
    const SidePosition& currentSide = side == Color::White ? mWhites : mBlacks;
    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    Bitboard bitboard{ 0 };

    const int32_t pawnDirection = side == Color::White ? 1 : -1;

    if (side == Color::White)
    {
        bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<0u>()) << 7u;
        bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<7u>()) << 9u;
    }
    else
    {
        bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<0u>()) >> 9u;
        bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<7u>()) >> 7u;
    }

    currentSide.knights.Iterate([&](uint32_t fromIndex)
    {
        bitboard |= Bitboard::GetKnightAttacks(Square(fromIndex));
    });

    const Bitboard rooks = currentSide.rooks | currentSide.queens;
    const Bitboard bishops = currentSide.bishops | currentSide.queens;

    rooks.Iterate([&](uint32_t fromIndex)
    {
        bitboard |= Bitboard::GenerateRookAttacks(Square(fromIndex), occupiedSquares);
    });

    bishops.Iterate([&](uint32_t fromIndex)
    {
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

void Position::GenerateMoveList(MoveList& outMoveList, uint32_t flags) const
{
    outMoveList.numMoves = 0u;

    GeneratePawnMoveList(outMoveList, flags);
    GenerateKnightMoveList(outMoveList, flags);
    GenerateRookMoveList(outMoveList, flags);
    GenerateBishopMoveList(outMoveList, flags);
    GenerateQueenMoveList(outMoveList, flags);
    GenerateKingMoveList(outMoveList, flags);
}

static const int32_t c_MvvLvaScoreBaseValue = 10000000;

static int32_t ComputeMvvLvaScore(const Piece attackingPiece, const Piece capturedPiece)
{
    return c_MvvLvaScoreBaseValue + 10 * (int32_t)capturedPiece - (int32_t)attackingPiece;
}

void Position::PushMove(const Move move, MoveList& outMoveList) const
{
    int32_t score = 0;

    if (move.piece == Piece::Pawn && move.isEnPassant)
    {
        score += c_MvvLvaScoreBaseValue;
    }
    else if (move.isCapture)
    {
        const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

        const Piece attackingPiece = move.piece;
        const Piece capturedPiece = opponentSide.GetPieceAtSquare(move.toSquare);
        score += ComputeMvvLvaScore(attackingPiece, capturedPiece);
    }

    outMoveList.PushMove(move, score);
}

void Position::GeneratePawnMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const bool onlyCaptures = flags & MOVE_GEN_ONLY_CAPTURES;
    const int32_t pawnDirection = mSideToMove == Color::White ? 1 : -1;
    const SidePosition& currentSide  = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;
    const uint32_t pawnStartingRank = mSideToMove == Color::White ? 1u : 6u;
    const uint32_t enPassantRank = mSideToMove == Color::White ? 5u : 2u;
    const uint32_t pawnFinalRank = mSideToMove == Color::White ? 6u : 1u;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    const auto generatePawnMove = [&](const Square fromSquare, const Square toSquare, bool isCapture, bool enPassant)
    {
        if (fromSquare.Rank() == pawnFinalRank)
        {
            // TODO promotion to rook/bishop should have very low priority when sorting moves
            // or not generate it at all if it wouldn't lead to stallmate?
            const Piece promotionList[] = { Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop };
            for (const Piece promoteTo : promotionList)
            {
                Move move;
                move.fromSquare = fromSquare;
                move.toSquare = toSquare;
                move.piece = Piece::Pawn;
                move.promoteTo = promoteTo;
                move.isCapture = isCapture;
                move.isEnPassant = enPassant;
                PushMove(move, outMoveList);
            }
        }
        else
        {
            Move move;
            move.fromSquare = fromSquare;
            move.toSquare = toSquare;
            move.piece = Piece::Pawn;
            move.promoteTo = Piece::None;
            move.isCapture = isCapture;
            move.isEnPassant = enPassant;
            PushMove(move, outMoveList);
        }
    };

    currentSide.pawns.Iterate([&](uint32_t fromIndex)
    {
        const Square fromSquare(fromIndex);
        const Square squareForward(fromSquare.Index() + pawnDirection * 8); // next rank

        // there should be no pawn in first or last rank
        ASSERT(fromSquare.Rank() > 0u && fromSquare.Rank() < 7u);

        if (!onlyCaptures)
        {
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
                        Move move;
                        move.fromSquare = fromSquare;
                        move.toSquare = twoSquaresForward;
                        move.piece = Piece::Pawn;
                        move.promoteTo = Piece::None;
                        move.isCapture = false;
                        move.isEnPassant = false;
                        PushMove(move, outMoveList);
                    }
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

void Position::GenerateKnightMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    currentSide.knights.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);

        Bitboard attackBitboard = Bitboard::GetKnightAttacks(square);
        if (flags & MOVE_GEN_ONLY_CAPTURES)
        {
            attackBitboard &= opponentSide.OccupiedExcludingKing();
        }

        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            // can't capture king
            if (opponentSide.king & targetSquare.Bitboard()) return;

            Move move;
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Knight;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
            PushMove(move, outMoveList);
        });
    });
}

void Position::GenerateRookMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    currentSide.rooks.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);

        Bitboard attackBitboard = Bitboard::GenerateRookAttacks(square, occupiedSquares);
        if (flags & MOVE_GEN_ONLY_CAPTURES)
        {
            attackBitboard &= opponentSide.OccupiedExcludingKing();
        }

        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move move;
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Rook;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
            PushMove(move, outMoveList);
        });
    });
}

void Position::GenerateBishopMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    currentSide.bishops.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);

        Bitboard attackBitboard = Bitboard::GenerateBishopAttacks(square, occupiedSquares);
        if (flags & MOVE_GEN_ONLY_CAPTURES)
        {
            attackBitboard &= opponentSide.OccupiedExcludingKing();
        }

        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move move;
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Bishop;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
            PushMove(move, outMoveList);
        });
    });
}

void Position::GenerateQueenMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    currentSide.queens.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);

        Bitboard attackBitboard =
            Bitboard::GenerateRookAttacks(square, occupiedSquares) |
            Bitboard::GenerateBishopAttacks(square, occupiedSquares);
        if (flags & MOVE_GEN_ONLY_CAPTURES)
        {
            attackBitboard &= opponentSide.OccupiedExcludingKing();
        }

        attackBitboard.Iterate([&](uint32_t toIndex)
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move move;
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Queen;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
            PushMove(move, outMoveList);
        });
    });
}

void Position::GenerateKingMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const bool onlyCaptures = flags & MOVE_GEN_ONLY_CAPTURES;
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

    Bitboard attackBitboard = Bitboard::GetKingAttacks(square);
    if (onlyCaptures)
    {
        attackBitboard &= opponentSide.OccupiedExcludingKing();
    }

    attackBitboard.Iterate([&](uint32_t toIndex)
    {
        const Square targetSquare(toIndex);

        // can't capture own piece
        if (currentSide.Occupied() & targetSquare.Bitboard()) return;

        Move move;
        move.fromSquare = square;
        move.toSquare = targetSquare;
        move.piece = Piece::King;
        move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        move.isCastling = false;
        PushMove(move, outMoveList);
    });

    if (!onlyCaptures && (currentSide.castlingRights & CastlingRights_All))
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

                Move move;
                move.fromSquare = square;
                move.toSquare = Square(2u, square.Rank());
                move.piece = Piece::King;
                move.isCapture = false;
                move.isCastling = true;
                PushMove(move, outMoveList);
            }

            if ((currentSide.castlingRights & CastlingRights_ShortCastleAllowed) &&
                ((occupiedSquares & shortCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & shortCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?

                Move move;
                move.fromSquare = square;
                move.toSquare = Square(6u, square.Rank());
                move.piece = Piece::King;
                move.isCapture = false;
                move.isCastling = true;
                PushMove(move, outMoveList);
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