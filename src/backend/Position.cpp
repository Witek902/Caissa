#include "Position.hpp"
#include "MoveList.hpp"
#include "Bitboard.hpp"
#include "Material.hpp"
#include "PositionHash.hpp"
#include "Evaluate.hpp"
#include "NeuralNetworkEvaluator.hpp"

#include <random>

const char* Position::InitPositionFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

bool Position::s_enableChess960 = false;

uint64_t Position::ComputeHash() const
{
    uint64_t hash = mSideToMove == Color::Black ? GetSideToMoveZobristHash() : 0llu;

    for (uint32_t colorIdx = 0; colorIdx < 2; ++colorIdx)
    {
        const Color color = (Color)colorIdx;
        const SidePosition& pos = mColors[colorIdx];

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
        if (mWhitesCastlingRights & (1 << i))
        {
            hash ^= GetCastlingRightsZobristHash(Color::White, i);
        }
        if (mBlacksCastlingRights & (1 << i))
        {
            hash ^= GetCastlingRightsZobristHash(Color::Black, i);
        }
    }

    return hash;
}

Piece SidePosition::GetPieceAtSquare(const Square square) const
{
    ASSERT(square.IsValid());

    const Bitboard squareBitboard = square.GetBitboard();

    Piece piece = Piece::None;

    if (pawns & squareBitboard)     piece = Piece::Pawn;
    if (knights & squareBitboard)   piece = Piece::Knight;
    if (bishops & squareBitboard)   piece = Piece::Bishop;
    if (rooks & squareBitboard)     piece = Piece::Rook;
    if (queens & squareBitboard)    piece = Piece::Queen;
    if (king & squareBitboard)      piece = Piece::King;

    return piece;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

Position::Position()
    : mSideToMove(Color::White)
    , mEnPassantSquare(Square::Invalid())
    , mWhitesCastlingRights(0)
    , mBlacksCastlingRights(0)
    , mHalfMoveCount(0u)
    , mMoveCount(1u)
    , mPieceSquareValueMG(0)
    , mPieceSquareValueEG(0)
    , mHash(0u)
{}

void Position::UpdatePieceSquareValue(Square square, const Piece piece, const Color color, bool remove)
{
    if (color != Color::White)
    {
        square = square.FlippedRank();
        remove = !remove;
    }

    (void)piece;

    //const PieceScore pieceScore = PSQT[(uint32_t)piece - (uint32_t)Piece::Pawn][square.mIndex];

    //if (remove)
    //{
    //    mPieceSquareValueMG -= pieceScore.mg;
    //    mPieceSquareValueEG -= pieceScore.eg;
    //}
    //else
    //{
    //    mPieceSquareValueMG += pieceScore.mg;
    //    mPieceSquareValueEG += pieceScore.eg;
    //}
}

void Position::SetPiece(const Square square, const Piece piece, const Color color)
{
    ASSERT(square.IsValid());
    ASSERT((uint8_t)piece <= (uint8_t)Piece::King);
    ASSERT(color == Color::White || color == Color::Black);

    const Bitboard mask = square.GetBitboard();
    SidePosition& pos = GetSide(color);

    ASSERT((pos.pawns & mask) == 0);
    ASSERT((pos.knights & mask) == 0);
    ASSERT((pos.bishops & mask) == 0);
    ASSERT((pos.rooks & mask) == 0);
    ASSERT((pos.queens & mask) == 0);
    ASSERT((pos.king & mask) == 0);

    UpdatePieceSquareValue(square, piece, color, false);

    mHash ^= GetPieceZobristHash(color, piece, square.Index());

    pos.GetPieceBitBoard(piece) |= mask;
}

void Position::RemovePiece(const Square square, const Piece piece, const Color color)
{
    const Bitboard mask = square.GetBitboard();
    SidePosition& pos = GetSide(color);
    Bitboard& targetBitboard = pos.GetPieceBitBoard(piece);

    ASSERT((targetBitboard & mask) == mask);
    targetBitboard &= ~mask;

    UpdatePieceSquareValue(square, piece, color, true);

    mHash ^= GetPieceZobristHash(color, piece, square.Index());
}

void Position::SetSideToMove(Color color)
{
    ASSERT(color == Color::White || color == Color::Black);

    if (mSideToMove != color)
    {
        mHash ^= GetSideToMoveZobristHash();
        mSideToMove = color;
    }
}

void Position::SetWhitesCastlingRights(uint8_t rightsMask)
{
    ASSERT(PopCount(rightsMask) <= 2);

    if (const uint8_t difference = mWhitesCastlingRights ^ rightsMask)
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (difference & (1 << i))
            {
                mHash ^= GetCastlingRightsZobristHash(Color::White, i);
            }
        }

        mWhitesCastlingRights = rightsMask;
    }
}

void Position::SetBlacksCastlingRights(uint8_t rightsMask)
{
    ASSERT(PopCount(rightsMask) <= 2);

    if (const uint8_t difference = mBlacksCastlingRights ^ rightsMask)
    {
        for (uint32_t i = 0; i < 8; ++i)
        {
            if (difference & (1 << i))
            {
                mHash ^= GetCastlingRightsZobristHash(Color::Black, i);
            }
        }

        mBlacksCastlingRights = rightsMask;
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

Bitboard Position::GetAttackedSquares(Color side) const
{
    const SidePosition& currentSide = mColors[(uint8_t)side];
    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    Bitboard bitboard{ 0 };

    if (currentSide.pawns)
    {
        if (side == Color::White)
        {
            bitboard |= Bitboard::GetPawnAttacks<Color::White>(currentSide.pawns);
        }
        else
        {
            bitboard |= Bitboard::GetPawnAttacks<Color::Black>(currentSide.pawns);
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

void Position::GeneratePawnMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Bitboard occupiedByCurrent = currentSide.Occupied();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;

    if (currentSide.pawns)
    {
        const int32_t pawnDirection = mSideToMove == Color::White ? 1 : -1;

        const uint32_t pawnStartingRank = mSideToMove == Color::White ? 1u : 6u;
        const uint32_t enPassantRank = mSideToMove == Color::White ? 5u : 2u;
        const uint32_t pawnFinalRank = mSideToMove == Color::White ? 6u : 1u;

        const auto generatePawnMove = [&](const Square fromSquare, const Square toSquare, bool isCapture, bool enPassant)
        {
            if (fromSquare.Rank() == pawnFinalRank) // pawn promotion
            {
                if (isCapture || (flags & MOVE_GEN_MASK_PROMOTIONS))
                {
                    const Piece promotionTargets[] = { Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop };
                    for (uint32_t i = 0; i < 4; ++i)
                    {
                        outMoveList.Push(Move::Make(fromSquare, toSquare, Piece::Pawn, promotionTargets[i], isCapture, false));
                    }
                }
            }
            else if ((flags & MOVE_GEN_MASK_QUIET) || isCapture)
            {
                outMoveList.Push(Move::Make(fromSquare, toSquare, Piece::Pawn, Piece::None, isCapture, enPassant));
            }
        };

        currentSide.pawns.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
        {
            const Square fromSquare(fromIndex);
            const Square squareForward(fromSquare.Index() + pawnDirection * 8); // next rank

            // there should be no pawn on the last rank
            ASSERT(fromSquare.RelativeRank(mSideToMove) < 7u);

            if (flags & MOVE_GEN_MASK_CAPTURES)
            {
                // capture on the left
                if (fromSquare.File() > 0u)
                {
                    const Square toSquare(fromSquare.Index() + pawnDirection * 8 - 1);
                    if (toSquare.GetBitboard() & opponentSide.OccupiedExcludingKing())
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
                    if (toSquare.GetBitboard() & opponentSide.OccupiedExcludingKing())
                    {
                        generatePawnMove(fromSquare, toSquare, true, false);
                    }
                    if (toSquare == mEnPassantSquare && toSquare.Rank() == enPassantRank)
                    {
                        generatePawnMove(fromSquare, toSquare, true, true);
                    }
                }
            }

            // can move forward only to non-occupied squares
            if ((occupiedSquares & squareForward.GetBitboard()) == 0u)
            {
                generatePawnMove(fromSquare, squareForward, false, false);

                if (fromSquare.Rank() == pawnStartingRank && (flags & MOVE_GEN_MASK_QUIET)) // move by two ranks
                {
                    const Square twoSquaresForward(fromSquare.Index() + pawnDirection * 16); // two ranks up

                    // can move forward only to non-occupied squares
                    if ((occupiedSquares & twoSquaresForward.GetBitboard()) == 0u)
                    {
                        outMoveList.Push(Move::Make(fromSquare, twoSquaresForward, Piece::Pawn, Piece::None));
                    }
                }
            }
        });
    }
}

void Position::GenerateMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Bitboard occupiedByCurrent = currentSide.Occupied();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;

    Bitboard filter = ~currentSide.Occupied(); // can't capture own piece
    filter &= ~opponentSide.king; // can't capture king
    if ((flags & MOVE_GEN_MASK_QUIET) == 0) filter &= occupiedByOpponent;
    if ((flags & MOVE_GEN_MASK_CAPTURES) == 0) filter &= ~occupiedByOpponent;

    GeneratePawnMoveList(outMoveList, flags);

    currentSide.knights.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = Bitboard::GetKnightAttacks(square) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);
            const bool isCapture = occupiedByOpponent & targetSquare.GetBitboard();
            outMoveList.Push(Move::Make(square, targetSquare, Piece::Knight, Piece::None, isCapture));
        });
    });

    currentSide.rooks.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = Bitboard::GenerateRookAttacks(square, occupiedSquares) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);
            const bool isCapture = occupiedByOpponent & targetSquare.GetBitboard();
            outMoveList.Push(Move::Make(square, targetSquare, Piece::Rook, Piece::None, isCapture));
        });
    });

    currentSide.bishops.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = Bitboard::GenerateBishopAttacks(square, occupiedSquares) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);
            const bool isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.GetBitboard();
            outMoveList.Push(Move::Make(square, targetSquare, Piece::Bishop, Piece::None, isCapture));
        });
    });

    currentSide.queens.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);
        const Bitboard attackBitboard = filter &
            (Bitboard::GenerateRookAttacks(square, occupiedSquares) |
            Bitboard::GenerateBishopAttacks(square, occupiedSquares));
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);
            const bool isCapture = occupiedByOpponent & targetSquare.GetBitboard();
            outMoveList.Push(Move::Make(square, targetSquare, Piece::Queen, Piece::None, isCapture));
        });
    });

    GenerateKingMoveList(outMoveList, flags);
}

Square Position::GetLongCastleRookSquare(const Square kingSquare, uint8_t castlingRights)
{
    constexpr uint8_t mask[] = { 0b00000000, 0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111 };
    const uint32_t longCastleMask = castlingRights & mask[kingSquare.File()];
    const uint8_t longCastleBitIndex = (uint8_t)FirstBitSet(longCastleMask);

    if (longCastleMask)
    {
        ASSERT(PopCount(longCastleMask) == 1);
        return Square(longCastleBitIndex, kingSquare.Rank());
    }

    return Square::Invalid();
}

Square Position::GetShortCastleRookSquare(const Square kingSquare, uint8_t castlingRights)
{
    constexpr uint8_t mask[] = { 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000, 0b00000000 };
    const uint32_t shortCastleMask = castlingRights & mask[kingSquare.File()];
    const uint8_t shortCastleBitIndex = (uint8_t)FirstBitSet(shortCastleMask);

    if (shortCastleMask)
    {
        ASSERT(PopCount(shortCastleMask) == 1);
        return Square(shortCastleBitIndex, kingSquare.Rank());
    }

    return Square::Invalid();
}

void Position::GenerateKingMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const uint8_t currentSideCastlingRights = (mSideToMove == Color::White) ? mWhitesCastlingRights : mBlacksCastlingRights;
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    ASSERT(currentSide.king);
    const uint32_t kingSquareIndex = FirstBitSet(currentSide.king);
    const Square kingSquare(kingSquareIndex);
    const Square opponentKingSquare(FirstBitSet(opponentSide.king));

    const Bitboard occupiedBySideToMove = currentSide.Occupied();
    const Bitboard occupiedByOponent = opponentSide.Occupied();

    Bitboard attackBitboard = Bitboard::GetKingAttacks(kingSquare);
    attackBitboard &= ~occupiedBySideToMove; // can't capture own piece
    attackBitboard &= ~Bitboard::GetKingAttacks(opponentKingSquare); // can't move to piece controlled by opponent's king
    if ((flags & MOVE_GEN_MASK_QUIET) == 0) attackBitboard &= occupiedByOponent;
    if ((flags & MOVE_GEN_MASK_CAPTURES) == 0) attackBitboard &= ~occupiedByOponent;

    attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
    {
        const Square targetSquare(toIndex);
        const bool isCapture = occupiedByOponent & targetSquare.GetBitboard();
        outMoveList.Push(Move::Make(kingSquare, targetSquare, Piece::King, Piece::None, isCapture));
    });

    if ((flags & MOVE_GEN_MASK_QUIET) && currentSideCastlingRights)
    {
        // TODO this is expensive
        const Bitboard opponentAttacks = GetAttackedSquares(GetOppositeColor(mSideToMove));

        // king can't be in check
        if ((currentSide.king & opponentAttacks) == 0u)
        {
            const Square longCastleRookSquare = GetLongCastleRookSquare(kingSquare, currentSideCastlingRights);
            const Square shortCastleRookSquare = GetShortCastleRookSquare(kingSquare, currentSideCastlingRights);

            if (longCastleRookSquare.IsValid() && shortCastleRookSquare.IsValid())
            {
                ASSERT(longCastleRookSquare.File() < shortCastleRookSquare.File());
            }

            // "long" castle
            if (longCastleRookSquare.IsValid())
            {
                ASSERT(longCastleRookSquare.File() < kingSquare.File());
                ASSERT(currentSide.rooks & longCastleRookSquare.GetBitboard());

                const Square targetKingSquare(2u, kingSquare.Rank());
                const Square targetRookSquare(3u, kingSquare.Rank());

                const Bitboard kingCrossedSquares = Bitboard::GetBetween(kingSquare, targetKingSquare) | targetKingSquare.GetBitboard();
                const Bitboard rookCrossedSquares = Bitboard::GetBetween(longCastleRookSquare, targetRookSquare) | targetRookSquare.GetBitboard();
                const Bitboard occupiedSquares = (occupiedBySideToMove | occupiedByOponent) & ~longCastleRookSquare.GetBitboard() & ~kingSquare.GetBitboard();

                if (0u == (opponentAttacks & kingCrossedSquares) &&
                    0u == (kingCrossedSquares & occupiedSquares) &&
                    0u == (rookCrossedSquares & occupiedSquares))
                {
                    outMoveList.Push(Move::Make(kingSquare, longCastleRookSquare, Piece::King, Piece::None, false, false, true, false));
                }
            }

            if (shortCastleRookSquare.IsValid())
            {
                ASSERT(kingSquare.File() < shortCastleRookSquare.File());
                ASSERT(currentSide.rooks & shortCastleRookSquare.GetBitboard());

                const Square targetKingSquare(6u, kingSquare.Rank());
                const Square targetRookSquare(5u, kingSquare.Rank());

                const Bitboard kingCrossedSquares = Bitboard::GetBetween(kingSquare, targetKingSquare) | targetKingSquare.GetBitboard();
                const Bitboard rookCrossedSquares = Bitboard::GetBetween(shortCastleRookSquare, targetRookSquare) | targetRookSquare.GetBitboard();
                const Bitboard occupiedSquares = (occupiedBySideToMove | occupiedByOponent) & ~shortCastleRookSquare.GetBitboard() & ~kingSquare.GetBitboard();

                if (0u == (opponentAttacks & kingCrossedSquares) &&
                    0u == (kingCrossedSquares & occupiedSquares) &&
                    0u == (rookCrossedSquares & occupiedSquares))
                {
                    outMoveList.Push(Move::Make(kingSquare, shortCastleRookSquare, Piece::King, Piece::None, false, false, false, true));
                }
            }
        }


/*
        // TODO simplify this
        const Bitboard longCastleKingCrossedSquares = (1ull << (kingSquareIndex - 1)) | (1ull << (kingSquareIndex - 2));
        const Bitboard shortCastleKingCrossedSquares = (1ull << (kingSquareIndex + 1)) | (1ull << (kingSquareIndex + 2));
        const Bitboard longCastleCrossedSquares = longCastleKingCrossedSquares | Bitboard(1ull << (kingSquareIndex - 3));
        const Bitboard shortCastleCrossedSquares = shortCastleKingCrossedSquares;

        // king can't be in check
        if ((currentSide.king & opponentAttacks) == 0u)
        {
            const Bitboard occupiedSquares = occupiedBySideToMove | occupiedByOponent;

            if ((currentSideCastlingRights & CastlingRights_LongCastleAllowed) &&
                ((occupiedSquares & longCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & longCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?
                outMoveList.Push(Move::Make(kingSquare, Square(2u, kingSquare.Rank()), Piece::King, Piece::None, false, false, true));
            }

            if ((currentSideCastlingRights & CastlingRights_ShortCastleAllowed) &&
                ((occupiedSquares & shortCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & shortCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?
                outMoveList.Push(Move::Make(kingSquare, Square(6u, kingSquare.Rank()), Piece::King, Piece::None, false, false, true));
            }
        }
*/
    }
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
    if (Whites().pawns)     bitboard |= Bitboard::GetPawnAttacks(square, Color::Black) & Whites().pawns;
    if (Blacks().pawns)     bitboard |= Bitboard::GetPawnAttacks(square, Color::White) & Blacks().pawns;

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
    if (side.pawns)                 bitboard |= Bitboard::GetPawnAttacks(square, GetOppositeColor(color)) & side.pawns;

    return bitboard;
}

bool Position::IsSquareVisible(const Square square, const Color color) const
{
    const SidePosition& side = GetSide(color);

    if (Bitboard::GetKingAttacks(square) & side.king) return true;

    if (Bitboard::GetKnightAttacks(square) & side.knights) return true;

    if (Bitboard::GetPawnAttacks(square, GetOppositeColor(color)) & side.pawns) return true;

    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    if (side.bishops | side.queens)
    {
        if (Bitboard::GenerateBishopAttacks(square, occupiedSquares) & (side.bishops | side.queens)) return true;
    }

    if (side.rooks | side.queens)
    {
        if (Bitboard::GenerateRookAttacks(square, occupiedSquares) & (side.rooks | side.queens)) return true;
    }

    return false;
}

bool Position::IsInCheck() const
{
    const SidePosition& currentSide = GetCurrentSide();

    const uint32_t kingSquareIndex = FirstBitSet(currentSide.king);
    return IsSquareVisible(Square(kingSquareIndex), GetOppositeColor(mSideToMove));
}

bool Position::IsInCheck(Color color) const
{
    const SidePosition& currentSide = GetSide(color);

    const uint32_t kingSquareIndex = FirstBitSet(currentSide.king);
    return IsSquareVisible(Square(kingSquareIndex), GetOppositeColor(color));
}

uint32_t Position::GetNumLegalMoves(std::vector<Move>* outMoves) const
{
    MoveList moves;
    GenerateMoveList(moves);

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

bool Position::IsMoveLegal(const Move& move) const
{
    ASSERT(IsMoveValid(move));

    Position positionAfterMove{ *this };
    return positionAfterMove.DoMove(move);
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
        ASSERT(mSideToMove == Color::White);

        if ((to.File() > 0 && (to.West_Unsafe().GetBitboard() & oponentPawns)) ||
            (to.File() < 7 && (to.East_Unsafe().GetBitboard() & oponentPawns)))
        {
            return Square(move.FromSquare().File(), 2u);
        }
    }

    if (from.Rank() == 6u && to.Rank() == 4u)
    {
        ASSERT(from.File() == to.File());
        ASSERT(mSideToMove == Color::Black);

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
        if (mWhitesCastlingRights & (1 << affectedSquare.File()))
        {
            mHash ^= GetCastlingRightsZobristHash(Color::White, affectedSquare.File());
            mWhitesCastlingRights &= ~(1 << affectedSquare.File());
        }
    }
    else if (affectedSquare.Rank() == 7)
    {
        if (mBlacksCastlingRights & (1 << affectedSquare.File()))
        {
            mHash ^= GetCastlingRightsZobristHash(Color::Black, affectedSquare.File());
            mBlacksCastlingRights &= ~(1 << affectedSquare.File());
        }
    }
}

bool Position::DoMove(const Move& move, NNEvaluatorContext* nnContext)
{
    ASSERT(IsMoveValid(move));  // move must be valid
    ASSERT(IsValid());          // board position must be valid

    SidePosition& opponentSide = GetOpponentSide();

    // move piece & mark NN accumulator as dirty
    {
        RemovePiece(move.FromSquare(), move.GetPiece(), mSideToMove);

        if (nnContext)
        {
            nnContext->MarkAsDirty();
            nnContext->removedPieces[0] = { move.GetPiece(), mSideToMove, move.FromSquare() };
            nnContext->numRemovedPieces = 1;
        }
    }

    // remove captured piece
    if (move.IsCapture())
    {
        if (!move.IsEnPassant())
        {
            const Piece capturedPiece = opponentSide.GetPieceAtSquare(move.ToSquare());
            const Color capturedColor = GetOppositeColor(mSideToMove);
            RemovePiece(move.ToSquare(), capturedPiece, capturedColor);

            if (nnContext)
            {
                nnContext->removedPieces[nnContext->numRemovedPieces++] = { capturedPiece, capturedColor, move.ToSquare() };
            }
        }

        // clear specific castling right after capturing a rook
        ClearRookCastlingRights(move.ToSquare());
    }

    // put moved piece
    if (!move.IsCastling())
    {
        const bool isPromotion = move.GetPiece() == Piece::Pawn && move.GetPromoteTo() != Piece::None;
        const Piece targetPiece = isPromotion ? move.GetPromoteTo() : move.GetPiece();
        SetPiece(move.ToSquare(), targetPiece, mSideToMove);

        if (nnContext)
        {
            nnContext->addedPieces[0] = { targetPiece, mSideToMove, move.ToSquare() };
            nnContext->numAddedPieces = 1;
        }
    }

    if (move.IsEnPassant())
    {
        Square captureSquare = Square::Invalid();
        if (move.ToSquare().Rank() == 5)  captureSquare = Square(move.ToSquare().File(), 4u);
        if (move.ToSquare().Rank() == 2)  captureSquare = Square(move.ToSquare().File(), 3u);
        ASSERT(captureSquare.IsValid());

        RemovePiece(captureSquare, Piece::Pawn, GetOppositeColor(mSideToMove));

        if (nnContext)
        {
            nnContext->removedPieces[nnContext->numRemovedPieces++] = { Piece::Pawn, GetOppositeColor(mSideToMove), captureSquare };
        }
    }

    SetEnPassantSquare(move.GetPiece() == Piece::Pawn ? ExtractEnPassantSquareFromMove(move) : Square::Invalid());

    if (move.GetPiece() == Piece::King)
    {
        if (move.IsCastling())
        {
            const uint8_t currentSideCastlingRights = (mSideToMove == Color::White) ? mWhitesCastlingRights : mBlacksCastlingRights;

            ASSERT(currentSideCastlingRights != 0);
            ASSERT(move.FromSquare().Rank() == 0 || move.FromSquare().Rank() == 7);
            ASSERT(move.FromSquare().Rank() == move.ToSquare().Rank());

            const Square oldKingSquare = move.FromSquare();
            Square oldRookSquare, newRookSquare, newKingSquare;

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

            if (nnContext)
            {
                nnContext->removedPieces[nnContext->numRemovedPieces++] = { Piece::Rook, mSideToMove, oldRookSquare };
                nnContext->addedPieces[nnContext->numAddedPieces++] = { Piece::King, mSideToMove, newKingSquare };
                nnContext->addedPieces[nnContext->numAddedPieces++] = { Piece::Rook, mSideToMove, newRookSquare };
            }
        }

        // clear all castling rights after moving a king
        if (mSideToMove == Color::White)
        {
            SetWhitesCastlingRights(0);
        }
        else
        {
            SetBlacksCastlingRights(0);
        }
    }

    // clear specific castling right after moving a rook
    if (move.GetPiece() == Piece::Rook)
    {
        ClearRookCastlingRights(move.FromSquare());
    }

    if (mSideToMove == Color::Black)
    {
        mMoveCount++;
    }

    if (move.GetPiece() == Piece::Pawn || move.IsCapture())
    {
        mHalfMoveCount = 0;
    }
    else
    {
        mHalfMoveCount++;
    }

    const Color prevToMove = mSideToMove;

    mSideToMove = GetOppositeColor(mSideToMove);
    mHash ^= GetSideToMoveZobristHash();

    ASSERT(IsValid());  // board position after the move must be valid

    // validate hash
    ASSERT(ComputeHash() == GetHash());

    if (nnContext)
    {
        ASSERT(nnContext->numRemovedPieces > 0 && nnContext->numAddedPieces > 0);
        ASSERT(nnContext->numRemovedPieces <= 2 && nnContext->numAddedPieces <= 2);
    }

    // can't be in check after move
    return !IsInCheck(prevToMove);
}

bool Position::DoNullMove()
{
    ASSERT(IsValid());          // board position must be valid
    ASSERT(!IsInCheck(mSideToMove));

    SetEnPassantSquare(Square::Invalid());

    if (mSideToMove == Color::Black)
    {
        mMoveCount++;
    }

    mHalfMoveCount++;

    mSideToMove = GetOppositeColor(mSideToMove);
    mHash ^= GetSideToMoveZobristHash();

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
    result.mBlacksCastlingRights    = mWhitesCastlingRights;
    result.mWhitesCastlingRights    = mBlacksCastlingRights;
    result.mSideToMove              = GetOppositeColor(mSideToMove);
    result.mMoveCount               = mMoveCount;
    result.mHalfMoveCount           = mHalfMoveCount;
    result.mHash                    = 0;
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

    mWhitesCastlingRights = 0;
    mBlacksCastlingRights = 0;

    mHash = ComputeHash();
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

    mWhitesCastlingRights = ReverseBits(mWhitesCastlingRights);
    mBlacksCastlingRights = ReverseBits(mBlacksCastlingRights);

    mHash = ComputeHash();
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

uint32_t Position::ToFeaturesVector(uint16_t* outFeatures, const NetworkInputMapping mapping) const
{
    uint32_t numFeatures = 0;

    Square whiteKingSquare = Square(FirstBitSet(Whites().king));
    Square blackKingSquare = Square(FirstBitSet(Blacks().king));

    uint32_t numInputs = 0;

    const auto writePieceFeatures = [&](const Bitboard bitboard, const uint32_t bitFlipMask) INLINE_LAMBDA
    {
        bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA { outFeatures[numFeatures++] = (uint16_t)(numInputs + (square ^ bitFlipMask)); });
        numInputs += 64;
    };

    const auto writePawnFeatures = [&](const Bitboard bitboard, const uint32_t bitFlipMask) INLINE_LAMBDA
    {
        // pawns cannot stand on first or last rank
        constexpr Bitboard mask = ~(Bitboard::RankBitboard<0>() | Bitboard::RankBitboard<7>());
        (bitboard & mask).Iterate([&](uint32_t square) INLINE_LAMBDA { outFeatures[numFeatures++] = (uint16_t)(numInputs + (square ^ bitFlipMask) - 8u); });
        numInputs += 48;
    };

    const auto writeKingRelativePieceFeatures = [&](const Bitboard bitboard, const uint32_t bitFlipMask, const uint32_t kingSquareIndex) INLINE_LAMBDA
    {
        bitboard.Iterate([&](uint32_t square) INLINE_LAMBDA
        {
            outFeatures[numFeatures++] = (uint16_t)(numInputs + 32u * kingSquareIndex + (square ^ bitFlipMask));
        });
        numInputs += 32 * 64;
    };

    if (mapping == NetworkInputMapping::Full)
    {
        writePawnFeatures(Whites().pawns, 0);
        writePieceFeatures(Whites().knights, 0);
        writePieceFeatures(Whites().bishops, 0);
        writePieceFeatures(Whites().rooks, 0);
        writePieceFeatures(Whites().queens, 0);

        // white king
        {
            outFeatures[numFeatures++] = (uint16_t)(numInputs + whiteKingSquare.Index());
            numInputs += 64;
        }

        writePawnFeatures(Blacks().pawns, 0);
        writePieceFeatures(Blacks().knights, 0);
        writePieceFeatures(Blacks().bishops, 0);
        writePieceFeatures(Blacks().rooks, 0);
        writePieceFeatures(Blacks().queens, 0);

        // black king
        {
            outFeatures[numFeatures++] = (uint16_t)(numInputs + blackKingSquare.Index());
            numInputs += 64;
        }

        ASSERT(numInputs == (2 * 5 * 64 + 2 * 48));
    }
    else if (mapping == NetworkInputMapping::Full_Symmetrical)
    {
        uint32_t bitFlipMask = 0;

        if (whiteKingSquare.File() >= 4)
        {
            whiteKingSquare = whiteKingSquare.FlippedFile();
            blackKingSquare = blackKingSquare.FlippedFile();

            // flip file
            bitFlipMask = 0b000111;
        }

        writePawnFeatures(Whites().pawns, bitFlipMask);
        writePieceFeatures(Whites().knights, bitFlipMask);
        writePieceFeatures(Whites().bishops, bitFlipMask);
        writePieceFeatures(Whites().rooks, bitFlipMask);
        writePieceFeatures(Whites().queens, bitFlipMask);

        // white king
        {
            const uint32_t whiteKingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
            ASSERT(whiteKingIndex < 32);
            outFeatures[numFeatures++] = (uint16_t)(numInputs + whiteKingIndex);
            numInputs += 32;
        }

        writePawnFeatures(Blacks().pawns, bitFlipMask);
        writePieceFeatures(Blacks().knights, bitFlipMask);
        writePieceFeatures(Blacks().bishops, bitFlipMask);
        writePieceFeatures(Blacks().rooks, bitFlipMask);
        writePieceFeatures(Blacks().queens, bitFlipMask);

        // black king
        {
            outFeatures[numFeatures++] = (uint16_t)(numInputs + blackKingSquare.Index());
            numInputs += 64;
        }

        ASSERT(numInputs == (32 + 64 + 2 * (4 * 64 + 48)));
    }
    else if (mapping == NetworkInputMapping::MaterialPacked_Symmetrical)
    {
        const bool pawnless = Whites().pawns == 0 && Blacks().pawns == 0;

        uint32_t bitFlipMask = 0;

        if (whiteKingSquare.File() >= 4)
        {
            whiteKingSquare = whiteKingSquare.FlippedFile();
            blackKingSquare = blackKingSquare.FlippedFile();

            // flip file
            bitFlipMask = 0b000111;
        }

        // vertical symmetry in pawnless positions
        // TODO diagonal symmetry
        if (pawnless && whiteKingSquare.Rank() >= 4)
        {
            whiteKingSquare = whiteKingSquare.FlippedRank();
            blackKingSquare = blackKingSquare.FlippedRank();

            // flip rank
            bitFlipMask |= 0b111000;
        }

        // white king
        if (pawnless)
        {
            const uint32_t whiteKingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
            ASSERT(whiteKingIndex < 16);
            outFeatures[numFeatures++] = (uint16_t)whiteKingIndex;
            numInputs += 16;
        }
        else
        {
            const uint32_t whiteKingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
            ASSERT(whiteKingIndex < 32);
            outFeatures[numFeatures++] = (uint16_t)whiteKingIndex;
            numInputs += 32;
        }

        // black king
        {
            outFeatures[numFeatures++] = (uint16_t)(numInputs + blackKingSquare.Index());
            numInputs += 64;
        }

        if (Whites().pawns)     writePawnFeatures(Whites().pawns, bitFlipMask);
        if (Whites().knights)   writePieceFeatures(Whites().knights, bitFlipMask);
        if (Whites().bishops)   writePieceFeatures(Whites().bishops, bitFlipMask);
        if (Whites().rooks)     writePieceFeatures(Whites().rooks, bitFlipMask);
        if (Whites().queens)    writePieceFeatures(Whites().queens, bitFlipMask);
        if (Blacks().pawns)     writePawnFeatures(Blacks().pawns, bitFlipMask);
        if (Blacks().knights)   writePieceFeatures(Blacks().knights, bitFlipMask);
        if (Blacks().bishops)   writePieceFeatures(Blacks().bishops, bitFlipMask);
        if (Blacks().rooks)     writePieceFeatures(Blacks().rooks, bitFlipMask);
        if (Blacks().queens)    writePieceFeatures(Blacks().queens, bitFlipMask);

        ASSERT(numInputs <= UINT16_MAX);
        ASSERT(numInputs == GetMaterialKey().GetNeuralNetworkInputsNumber());
    }
    else if (mapping == NetworkInputMapping::KingPiece_Symmetrical)
    {
        // white perspective
        {
            uint32_t bitFlipMask = 0;

            if (whiteKingSquare.File() >= 4)
            {
                whiteKingSquare = whiteKingSquare.FlippedFile();
                bitFlipMask = 0b000111; // flip file
            }

            const uint32_t kingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
            ASSERT(kingIndex < 32);

            writeKingRelativePieceFeatures(Whites().pawns, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().knights, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().bishops, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().rooks, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().queens, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().pawns, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().knights, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().bishops, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().rooks, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().queens, bitFlipMask, kingIndex);
        }

        // black perspective
        {
            uint32_t bitFlipMask = 0;

            if (blackKingSquare.File() >= 4)
            {
                blackKingSquare = blackKingSquare.FlippedFile();
                bitFlipMask = 0b000111; // flip file
            }

            const uint32_t kingIndex = 4 * blackKingSquare.Rank() + blackKingSquare.File();
            ASSERT(kingIndex < 32);

            writeKingRelativePieceFeatures(Whites().pawns, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().knights, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().bishops, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().rooks, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Whites().queens, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().pawns, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().knights, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().bishops, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().rooks, bitFlipMask, kingIndex);
            writeKingRelativePieceFeatures(Blacks().queens, bitFlipMask, kingIndex);
        }

        // 2 perspectives, 10 pieces, 32 king locations, 64 piece locations
        ASSERT(numInputs == 2 * 10 * 32 * 64);
    }
    else if (mapping == NetworkInputMapping::MaterialPacked_KingPiece_Symmetrical)
    {
        // white perspective
        {
            uint32_t bitFlipMask = 0;

            if (whiteKingSquare.File() >= 4)
            {
                whiteKingSquare = whiteKingSquare.FlippedFile();
                bitFlipMask = 0b000111; // flip file
            }

            const uint32_t kingIndex = 4 * whiteKingSquare.Rank() + whiteKingSquare.File();
            ASSERT(kingIndex < 32);

            if (Whites().pawns)     writeKingRelativePieceFeatures(Whites().pawns, bitFlipMask, kingIndex);
            if (Whites().knights)   writeKingRelativePieceFeatures(Whites().knights, bitFlipMask, kingIndex);
            if (Whites().bishops)   writeKingRelativePieceFeatures(Whites().bishops, bitFlipMask, kingIndex);
            if (Whites().rooks)     writeKingRelativePieceFeatures(Whites().rooks, bitFlipMask, kingIndex);
            if (Whites().queens)    writeKingRelativePieceFeatures(Whites().queens, bitFlipMask, kingIndex);
            if (Blacks().pawns)     writeKingRelativePieceFeatures(Blacks().pawns, bitFlipMask, kingIndex);
            if (Blacks().knights)   writeKingRelativePieceFeatures(Blacks().knights, bitFlipMask, kingIndex);
            if (Blacks().bishops)   writeKingRelativePieceFeatures(Blacks().bishops, bitFlipMask, kingIndex);
            if (Blacks().rooks)     writeKingRelativePieceFeatures(Blacks().rooks, bitFlipMask, kingIndex);
            if (Blacks().queens)    writeKingRelativePieceFeatures(Blacks().queens, bitFlipMask, kingIndex);
        }

        // black perspective
        {
            uint32_t bitFlipMask = 0;

            if (blackKingSquare.File() >= 4)
            {
                blackKingSquare = blackKingSquare.FlippedFile();
                bitFlipMask = 0b000111; // flip file
            }

            const uint32_t kingIndex = 4 * blackKingSquare.Rank() + blackKingSquare.File();
            ASSERT(kingIndex < 32);

            if (Whites().pawns)     writeKingRelativePieceFeatures(Whites().pawns, bitFlipMask, kingIndex);
            if (Whites().knights)   writeKingRelativePieceFeatures(Whites().knights, bitFlipMask, kingIndex);
            if (Whites().bishops)   writeKingRelativePieceFeatures(Whites().bishops, bitFlipMask, kingIndex);
            if (Whites().rooks)     writeKingRelativePieceFeatures(Whites().rooks, bitFlipMask, kingIndex);
            if (Whites().queens)    writeKingRelativePieceFeatures(Whites().queens, bitFlipMask, kingIndex);
            if (Blacks().pawns)     writeKingRelativePieceFeatures(Blacks().pawns, bitFlipMask, kingIndex);
            if (Blacks().knights)   writeKingRelativePieceFeatures(Blacks().knights, bitFlipMask, kingIndex);
            if (Blacks().bishops)   writeKingRelativePieceFeatures(Blacks().bishops, bitFlipMask, kingIndex);
            if (Blacks().rooks)     writeKingRelativePieceFeatures(Blacks().rooks, bitFlipMask, kingIndex);
            if (Blacks().queens)    writeKingRelativePieceFeatures(Blacks().queens, bitFlipMask, kingIndex);
        }

        // 2 perspectives, X pieces, 32 king locations, 64 piece locations
        ASSERT(numInputs == 2 * GetMaterialKey().GetActivePiecesCount() * 32 * 64);
    }
    else
    {
        DEBUG_BREAK();
    }

    return numFeatures;
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
         if (side.queens)   value = c_queenValue.eg;
    else if (side.rooks)    value = c_rookValue.eg;
    else if (side.knights)  value = c_knightValue.eg;
    else if (side.bishops)  value = c_bishopValue.eg;
    else if (side.pawns)    value = c_pawnValue.eg;

    // can promote to queen
    if (GetCurrentSide().pawns & (mSideToMove == Color::White ? Bitboard::RankBitboard<6>() : Bitboard::RankBitboard<1>()))
    {
        value += c_queenValue.eg - c_pawnValue.eg;
    }

    return value;
}

bool Position::StaticExchangeEvaluation(const Move& move, int32_t treshold) const
{
    const Square toSquare = move.ToSquare();
    const Square fromSquare = move.FromSquare();

    const int32_t c_seePieceValues[] =
    {
        0, // none
        pawnValue,
        knightValue,
        bishopValue,
        rookValue,
        queenValue,
        kingValue,
    };

    Bitboard occupied = Whites().Occupied() | Blacks().Occupied();

    int32_t balance = 0;

    {
        const Piece capturedPiece = GetOpponentSide().GetPieceAtSquare(toSquare);
        balance = c_seePieceValues[(uint32_t)capturedPiece] - treshold;
        if (balance < 0) return false;
    }

    {
        const Piece movedPiece = GetCurrentSide().GetPieceAtSquare(fromSquare);
        balance = c_seePieceValues[(uint32_t)movedPiece] - balance;
        if (balance <= 0) return true;
    }

    // "do" move
    occupied &= ~fromSquare.GetBitboard();
    occupied |= toSquare.GetBitboard();

    Bitboard allAttackers = GetAttackers(toSquare, occupied);

    Color sideToMove = mSideToMove;
    int32_t result = 1;

    for (;;)
    {
        sideToMove = GetOppositeColor(sideToMove);
        allAttackers &= occupied;

        const SidePosition& side = mColors[(uint8_t)sideToMove];
        const Bitboard ourAttackers = allAttackers & side.Occupied();
        const Bitboard theirAttackers = allAttackers & ~side.Occupied();

        // no more attackers - side to move loses
        if (ourAttackers == 0) break;

        result ^= 1;

        // TODO filter out pinned pieces

        Bitboard pieceBitboard;

        if ((pieceBitboard = ourAttackers & side.pawns))
        {
            // remove attacker from occupied squares
            const uint32_t attackerSquare = FirstBitSet(pieceBitboard.value);
            const Bitboard mask = 1ull << attackerSquare;
            ASSERT((occupied & mask) != 0);
            occupied ^= mask;

            // update diagonal attackers
            allAttackers |= Bitboard::GenerateBishopAttacks(move.ToSquare(), occupied) & (Whites().bishops | Blacks().bishops | Whites().queens | Blacks().queens);

            balance = pawnValue - balance;
            if (balance < result) break;
        }
        else if ((pieceBitboard = ourAttackers & side.knights))
        {
            // remove attacker from occupied squares
            const uint32_t attackerSquare = FirstBitSet(pieceBitboard.value);
            const Bitboard mask = 1ull << attackerSquare;
            ASSERT((occupied & mask) != 0);
            occupied ^= mask;

            balance = knightValue - balance;
            if (balance < result) break;
        }
        else if ((pieceBitboard = ourAttackers & side.bishops))
        {
            // remove attacker from occupied squares
            const uint32_t attackerSquare = FirstBitSet(pieceBitboard.value);
            const Bitboard mask = 1ull << attackerSquare;
            ASSERT((occupied & mask) != 0);
            occupied ^= mask;

            // update diagonal attackers
            allAttackers |= Bitboard::GenerateBishopAttacks(toSquare, occupied) & (Whites().bishops | Blacks().bishops | Whites().queens | Blacks().queens);

            balance = bishopValue - balance;
            if (balance < result) break;
        }
        else if ((pieceBitboard = ourAttackers & side.rooks))
        {
            // remove attacker from occupied squares
            const uint32_t attackerSquare = FirstBitSet(pieceBitboard.value);
            const Bitboard mask = 1ull << attackerSquare;
            ASSERT((occupied & mask) != 0);
            occupied ^= mask;

            // update hirozontal/vertical attackers
            allAttackers |= Bitboard::GenerateRookAttacks(toSquare, occupied) & (Whites().rooks | Blacks().rooks | Whites().queens | Blacks().queens);

            balance = rookValue - balance;
            if (balance < result) break;
        }
        else if ((pieceBitboard = ourAttackers & side.queens))
        {
            // remove attacker from occupied squares
            const uint32_t attackerSquare = FirstBitSet(pieceBitboard.value);
            const Bitboard mask = 1ull << attackerSquare;
            ASSERT((occupied & mask) != 0);
            occupied ^= mask;

            // update hirozontal/vertical/diagonal attackers
            allAttackers |= Bitboard::GenerateBishopAttacks(toSquare, occupied) & (Whites().bishops | Blacks().bishops | Whites().queens | Blacks().queens);
            allAttackers |= Bitboard::GenerateRookAttacks(toSquare, occupied) & (Whites().rooks | Blacks().rooks | Whites().queens | Blacks().queens);

            balance = rookValue - balance;
            if (balance < result) break;
        }
        else // king
        {
            // if capturing with the king, but oponent still has attacker, return the result (can't be in check)
            if (theirAttackers)
            {
                result ^= 1;
            }

            break;
        }
    }

    return result != 0;
}

bool Position::IsQuiet() const
{
    if (IsInCheck(mSideToMove))
    {
        return false;
    }

    MoveList moves;
    GenerateMoveList(moves, MOVE_GEN_MASK_CAPTURES|MOVE_GEN_MASK_PROMOTIONS);

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