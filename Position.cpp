#include "Position.hpp"
#include "MoveList.hpp"
#include "Bitboard.hpp"

#include <random>

const char* Position::InitPositionFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static uint64_t s_BlackToMoveHash;
static uint64_t s_PiecePositionHash[2][6][64];
static uint64_t s_CastlingRightsHash[2][2];
static uint64_t s_EnPassantFileHash[8];

void InitZobristHash()
{
    std::mt19937_64 mt(0x06db3aa64a37b526LLU);
    std::uniform_int_distribution<uint64_t> distr;

    s_BlackToMoveHash = distr(mt);

    for (uint32_t color = 0; color < 2; ++color)
    {
        for (uint32_t piece = 0; piece < 6; ++piece)
        {
            for (uint32_t square = 0; square < 64; ++square)
            {
                s_PiecePositionHash[color][piece][square] = distr(mt);
            }
        }
    }

    for (uint32_t file = 0; file < 8; ++file)
    {
        s_EnPassantFileHash[file] = distr(mt);
    }

    s_CastlingRightsHash[0][0] = distr(mt);
    s_CastlingRightsHash[0][1] = distr(mt);
    s_CastlingRightsHash[1][0] = distr(mt);
    s_CastlingRightsHash[1][1] = distr(mt);
}

uint64_t Position::ComputeHash() const
{
    uint64_t hash = mSideToMove == Color::Black ? s_BlackToMoveHash : 0llu;

    for (uint32_t color = 0; color < 2; ++color)
    {
        const SidePosition& pos = mColors[color];

        pos.pawns.Iterate([&](uint32_t square)      { hash ^= s_PiecePositionHash[color][0][square]; });
        pos.knights.Iterate([&](uint32_t square)    { hash ^= s_PiecePositionHash[color][1][square]; });
        pos.bishops.Iterate([&](uint32_t square)    { hash ^= s_PiecePositionHash[color][2][square]; });
        pos.rooks.Iterate([&](uint32_t square)      { hash ^= s_PiecePositionHash[color][3][square]; });
        pos.queens.Iterate([&](uint32_t square)     { hash ^= s_PiecePositionHash[color][4][square]; });
        pos.king.Iterate([&](uint32_t square)       { hash ^= s_PiecePositionHash[color][5][square]; });
    }

    if (mWhitesCastlingRights & CastlingRights_ShortCastleAllowed)  hash ^= s_CastlingRightsHash[0][0];
    if (mWhitesCastlingRights & CastlingRights_LongCastleAllowed)   hash ^= s_CastlingRightsHash[0][1];
    if (mBlacksCastlingRights & CastlingRights_ShortCastleAllowed)  hash ^= s_CastlingRightsHash[1][0];
    if (mBlacksCastlingRights & CastlingRights_LongCastleAllowed)   hash ^= s_CastlingRightsHash[1][1];

    if (mEnPassantSquare.IsValid()) hash ^= s_EnPassantFileHash[mEnPassantSquare.File()];

    return hash;
}

NO_INLINE Piece SidePosition::GetPieceAtSquare(const Square square) const
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
    : mHash(0)
    , mSideToMove(Color::White)
    , mEnPassantSquare(Square::Invalid())
    , mWhitesCastlingRights(CastlingRights_All)
    , mBlacksCastlingRights(CastlingRights_All)
    , mHalfMoveCount(0u)
    , mMoveCount(1u)
{}

void Position::SetPiece(const Square square, const Piece piece, const Color color)
{
    const Bitboard mask = square.Bitboard();
    SidePosition& pos = mColors[(uint8_t)color];

    ASSERT((pos.pawns & mask) == 0);
    ASSERT((pos.knights & mask) == 0);
    ASSERT((pos.bishops & mask) == 0);
    ASSERT((pos.rooks & mask) == 0);
    ASSERT((pos.queens & mask) == 0);
    ASSERT((pos.king & mask) == 0);

    const uint32_t colorIndex = (uint32_t)color;
    const uint32_t pieceIndex = (uint32_t)piece - 1;
    mHash ^= s_PiecePositionHash[colorIndex][pieceIndex][square.Index()];

    pos.GetPieceBitBoard(piece) |= mask;
    pos.occupied |= mask;
}

void Position::RemovePiece(const Square square, const Piece piece, const Color color)
{
    const Bitboard mask = square.Bitboard();
    SidePosition& pos = mColors[(uint8_t)color];
    Bitboard& targetBitboard = pos.GetPieceBitBoard(piece);

    ASSERT((targetBitboard & mask) == mask);
    ASSERT((pos.occupied & mask) == mask);
    targetBitboard &= ~mask;
    pos.occupied &= ~mask;

    const uint32_t colorIndex = (uint32_t)color;
    const uint32_t pieceIndex = (uint32_t)piece - 1;
    mHash ^= s_PiecePositionHash[colorIndex][pieceIndex][square.Index()];
}

void Position::SetEnPassantSquare(const Square square)
{
    if (mEnPassantSquare.IsValid())
    {
        mHash ^= s_EnPassantFileHash[mEnPassantSquare.File()];
    }
    if (square.IsValid())
    {
        mHash ^= s_EnPassantFileHash[square.File()];
    }

    mEnPassantSquare = square;
}

void Position::ClearEnPassantSquare()
{
    if (mEnPassantSquare.IsValid())
    {
        mHash ^= s_EnPassantFileHash[mEnPassantSquare.File()];
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
            bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<0u>()) << 7u;
            bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<7u>()) << 9u;
        }
        else
        {
            bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<0u>()) >> 9u;
            bitboard |= (currentSide.pawns & ~Bitboard::FileBitboard<7u>()) >> 7u;
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

void Position::GeneratePawnMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const bool onlyTactical = flags & MOVE_GEN_ONLY_TACTICAL;
    const int32_t pawnDirection = mSideToMove == Color::White ? 1 : -1;
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();
    const uint32_t pawnStartingRank = mSideToMove == Color::White ? 1u : 6u;
    const uint32_t enPassantRank = mSideToMove == Color::White ? 5u : 2u;
    const uint32_t pawnFinalRank = mSideToMove == Color::White ? 6u : 1u;

    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    const auto generatePawnMove = [&](const Square fromSquare, const Square toSquare, bool isCapture, bool enPassant)
    {
        if (fromSquare.Rank() == pawnFinalRank) // pawn promotion
        {
            const Piece promotionList[] = { Piece::Queen, Piece::Knight, Piece::Rook, Piece::Bishop };
            for (const Piece promoteTo : promotionList)
            {
                Move move = Move::Invalid();
                move.fromSquare = fromSquare;
                move.toSquare = toSquare;
                move.piece = Piece::Pawn;
                move.promoteTo = promoteTo;
                move.isCapture = isCapture;
                move.isEnPassant = enPassant;
                outMoveList.Push(move);
            }
        }
        else if (!onlyTactical || isCapture)
        {
            Move move = Move::Invalid();
            move.fromSquare = fromSquare;
            move.toSquare = toSquare;
            move.piece = Piece::Pawn;
            move.promoteTo = Piece::None;
            move.isCapture = isCapture;
            move.isEnPassant = enPassant;
            outMoveList.Push(move);
        }
    };

    currentSide.pawns.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square fromSquare(fromIndex);
        const Square squareForward(fromSquare.Index() + pawnDirection * 8); // next rank

        // there should be no pawn in first or last rank
        ASSERT(fromSquare.Rank() > 0u && fromSquare.Rank() < 7u);

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

        // can move forward only to non-occupied squares
        if ((occupiedSquares & squareForward.Bitboard()) == 0u)
        {
            generatePawnMove(fromSquare, squareForward, false, false);

            if (fromSquare.Rank() == pawnStartingRank && !onlyTactical) // move by two ranks
            {
                const Square twoSquaresForward(fromSquare.Index() + pawnDirection * 16); // two ranks up

                // can move forward only to non-occupied squares
                if ((occupiedSquares & twoSquaresForward.Bitboard()) == 0u)
                {
                    Move move = Move::Invalid();
                    move.fromSquare = fromSquare;
                    move.toSquare = twoSquaresForward;
                    move.piece = Piece::Pawn;
                    move.promoteTo = Piece::None;
                    outMoveList.Push(move);
                }
            }
        }
    });
}

void Position::GenerateKnightMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    currentSide.knights.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);

        Bitboard attackBitboard = Bitboard::GetKnightAttacks(square);
        attackBitboard &= ~currentSide.Occupied(); // can't capture own piece
        if (flags & MOVE_GEN_ONLY_TACTICAL) attackBitboard &= opponentSide.occupied;
        attackBitboard &= ~opponentSide.king; // can't capture king

        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);

            Move move = Move::Invalid();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Knight;
            move.isCapture = opponentSide.occupied & targetSquare.Bitboard();
            outMoveList.Push(move);
        });
    });
}

void Position::GenerateRookMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    currentSide.rooks.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);

        Bitboard attackBitboard = Bitboard::GenerateRookAttacks(square, occupiedSquares);
        attackBitboard &= ~currentSide.Occupied(); // can't capture own piece
        if (flags & MOVE_GEN_ONLY_TACTICAL) attackBitboard &= opponentSide.occupied;
        attackBitboard &= ~opponentSide.king; // can't capture king

        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);

            Move move = Move::Invalid();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Rook;
            move.isCapture = opponentSide.occupied & targetSquare.Bitboard();
            outMoveList.Push(move);
        });
    });
}

void Position::GenerateBishopMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    currentSide.bishops.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);

        Bitboard attackBitboard = Bitboard::GenerateBishopAttacks(square, occupiedSquares);
        if (flags & MOVE_GEN_ONLY_TACTICAL)
        {
            attackBitboard &= opponentSide.OccupiedExcludingKing();
        }

        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move move = Move::Invalid();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Bishop;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
            outMoveList.Push(move);
        });
    });
}

void Position::GenerateQueenMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    currentSide.queens.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square square(fromIndex);

        Bitboard attackBitboard =
            Bitboard::GenerateRookAttacks(square, occupiedSquares) |
            Bitboard::GenerateBishopAttacks(square, occupiedSquares);
        if (flags & MOVE_GEN_ONLY_TACTICAL)
        {
            attackBitboard &= opponentSide.OccupiedExcludingKing();
        }

        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            const Square targetSquare(toIndex);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) return;

            Move move = Move::Invalid();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Queen;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
            outMoveList.Push(move);
        });
    });
}

void Position::GenerateKingMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const bool onlyTactical = flags & MOVE_GEN_ONLY_TACTICAL;
    const uint32_t numKingOffsets = 8u;
    const int32_t kingFileOffsets[numKingOffsets] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const int32_t kingRankOffsets[numKingOffsets] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    const CastlingRights& currentSideCastlingRights = (mSideToMove == Color::White) ? mWhitesCastlingRights : mBlacksCastlingRights;
    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    unsigned long kingSquareIndex;
    if (0 == _BitScanForward64(&kingSquareIndex, currentSide.king))
    {
        return;
    }

    const Square square(kingSquareIndex);

    Bitboard attackBitboard = Bitboard::GetKingAttacks(square);
    if (onlyTactical)
    {
        attackBitboard &= opponentSide.OccupiedExcludingKing();
    }

    attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
    {
        const Square targetSquare(toIndex);

        // can't capture own piece
        if (currentSide.Occupied() & targetSquare.Bitboard()) return;

        Move move = Move::Invalid();
        move.fromSquare = square;
        move.toSquare = targetSquare;
        move.piece = Piece::King;
        move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        outMoveList.Push(move);
    });

    if (!onlyTactical && (currentSideCastlingRights & CastlingRights_All))
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
            if ((currentSideCastlingRights & CastlingRights_LongCastleAllowed) &&
                ((occupiedSquares & longCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & longCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?

                Move move = Move::Invalid();
                move.fromSquare = square;
                move.toSquare = Square(2u, square.Rank());
                move.piece = Piece::King;
                move.isCastling = true;
                outMoveList.Push(move);
            }

            if ((currentSideCastlingRights & CastlingRights_ShortCastleAllowed) &&
                ((occupiedSquares & shortCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & shortCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?

                Move move = Move::Invalid();
                move.fromSquare = square;
                move.toSquare = Square(6u, square.Rank());
                move.piece = Piece::King;
                move.isCastling = true;
                outMoveList.Push(move);
            }
        }
    }
}

const Bitboard Position::GetAttackers(const Square square, const Color sideColor) const
{
    const SidePosition& side = mColors[(uint8_t)sideColor];
    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    Bitboard bitboard = Bitboard::GetKingAttacks(square) & side.king;

    if (side.knights)
    {
        bitboard |= Bitboard::GetKnightAttacks(square) & side.knights;
    }

    if (side.rooks | side.queens)
    {
        bitboard |= Bitboard::GenerateRookAttacks(square, occupiedSquares) & (side.rooks | side.queens);
    }

    if (side.bishops | side.queens)
    {
        bitboard |= Bitboard::GenerateBishopAttacks(square, occupiedSquares) & (side.bishops | side.queens);
    }

    if (side.pawns)
    {
        bitboard |= Bitboard::GetPawnAttacks(square, GetOppositeColor(sideColor)) & side.pawns;
    }

    return bitboard;
}

bool Position::IsSquareVisible(const Square square, const Color sideColor) const
{
    return GetAttackers(square, sideColor) != 0;
}

bool Position::IsInCheck(Color sideColor) const
{
    const SidePosition& currentSide = mColors[(uint8_t)sideColor];

    unsigned long kingSquareIndex;
    _BitScanForward64(&kingSquareIndex, currentSide.king);

    return IsSquareVisible(Square(kingSquareIndex), GetOppositeColor(sideColor));
}

uint32_t Position::GetNumLegalMoves(std::vector<Move>* outMoves) const
{
    MoveList moves;
    GenerateMoveList(moves);

    if (moves.numMoves == 0)
    {
        return 0;
    }

    uint32_t numLegalMoves = 0u;
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves[i].move;
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
    return GetNumLegalMoves() > 0u && IsInCheck(mSideToMove);
}

bool Position::IsStalemate() const
{
    return GetNumLegalMoves() > 0u && !IsInCheck(mSideToMove);
}

bool Position::IsMoveLegal(const Move& move) const
{
    ASSERT(IsMoveValid(move));

    Position positionAfterMove{ *this };
    return positionAfterMove.DoMove(move);
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

    return Square::Invalid();
}

void Position::ClearRookCastlingRights(const Square affectedSquare)
{
    switch (affectedSquare.mIndex)
    {
    case Square_h1:
        if (mWhitesCastlingRights & CastlingRights_ShortCastleAllowed) mHash ^= s_CastlingRightsHash[0][0];
        mWhitesCastlingRights = CastlingRights(mWhitesCastlingRights & ~CastlingRights_ShortCastleAllowed);
        break;
    case Square_a1:
        if (mWhitesCastlingRights & CastlingRights_LongCastleAllowed) mHash ^= s_CastlingRightsHash[0][1];
        mWhitesCastlingRights = CastlingRights(mWhitesCastlingRights & ~CastlingRights_LongCastleAllowed);
        break;
    case Square_h8:
        if (mBlacksCastlingRights & CastlingRights_ShortCastleAllowed) mHash ^= s_CastlingRightsHash[1][0];
        mBlacksCastlingRights = CastlingRights(mBlacksCastlingRights & ~CastlingRights_ShortCastleAllowed);
        break;
    case Square_a8:
        if (mBlacksCastlingRights & CastlingRights_LongCastleAllowed) mHash ^= s_CastlingRightsHash[1][1];
        mBlacksCastlingRights = CastlingRights(mBlacksCastlingRights & ~CastlingRights_LongCastleAllowed);
        break;
    };
}

bool Position::DoMove(const Move& move)
{
    ASSERT(IsMoveValid(move));  // move must be valid
    ASSERT(IsValid());          // board position must be valid

    SidePosition& opponentSide = GetOpponentSide();
    const Bitboard occupiedSquares = Whites().Occupied() | Blacks().Occupied();

    // move piece
    RemovePiece(move.fromSquare, move.piece, mSideToMove);

    if (move.isCapture)
    {
        if (!move.isEnPassant)
        {
            const Piece capturedPiece = opponentSide.GetPieceAtSquare(move.toSquare);
            RemovePiece(move.toSquare, capturedPiece, GetOppositeColor(mSideToMove));
        }

        // clear specific castling right after capturing a rook
        ClearRookCastlingRights(move.toSquare);
    }

    // move piece
    const bool isPromotion = move.piece == Piece::Pawn && move.promoteTo != Piece::None;
    SetPiece(move.toSquare, isPromotion ? move.promoteTo : move.piece, mSideToMove);

    if (move.isEnPassant)
    {
        Square captureSquare = Square::Invalid();
        if (move.toSquare.Rank() == 5)  captureSquare = Square(move.toSquare.File(), 4u);
        if (move.toSquare.Rank() == 2)  captureSquare = Square(move.toSquare.File(), 3u);
        ASSERT(captureSquare.IsValid());

        RemovePiece(captureSquare, Piece::Pawn, GetOppositeColor(mSideToMove));
    }

    SetEnPassantSquare(move.piece == Piece::Pawn ? ExtractEnPassantSquareFromMove(move) : Square::Invalid());

    if (move.piece == Piece::King)
    {
        if (move.isCastling)
        {
            ASSERT(move.fromSquare.Rank() == 0 || move.fromSquare.Rank() == 7);
            ASSERT(move.fromSquare.Rank() == move.toSquare.Rank());

            Square oldRookSquare, newRookSquare;

            // short castle
            if (move.fromSquare.File() == 4u && move.toSquare.File() == 6u)
            {
                oldRookSquare = Square(7u, move.fromSquare.Rank());
                newRookSquare = Square(5u, move.fromSquare.Rank());
            }
            // long castle
            else if (move.fromSquare.File() == 4u && move.toSquare.File() == 2u)
            {
                oldRookSquare = Square(0u, move.fromSquare.Rank());
                newRookSquare = Square(3u, move.fromSquare.Rank());
            }
            else // invalid castle
            {
                ASSERT(false);
            }

            RemovePiece(oldRookSquare, Piece::Rook, mSideToMove);
            SetPiece(newRookSquare, Piece::Rook, mSideToMove);
        }

        // clear all castling rights after moving a king
        CastlingRights& currentSideCastlingRights = (mSideToMove == Color::White) ? mWhitesCastlingRights : mBlacksCastlingRights;
        if (currentSideCastlingRights & CastlingRights_ShortCastleAllowed)  mHash ^= s_CastlingRightsHash[(uint32_t)mSideToMove][0];
        if (currentSideCastlingRights & CastlingRights_LongCastleAllowed)   mHash ^= s_CastlingRightsHash[(uint32_t)mSideToMove][1];
        currentSideCastlingRights = CastlingRights(0);
    }

    // clear specific castling right after moving a rook
    if (move.piece == Piece::Rook)
    {
        ClearRookCastlingRights(move.fromSquare);
    }

    if (mSideToMove == Color::Black)
    {
        mMoveCount++;
    }

    if (move.piece == Piece::Pawn || move.isCapture)
    {
        mHalfMoveCount = 0;
    }
    else
    {
        mHalfMoveCount++;
    }

    const Color prevToMove = mSideToMove;

    mSideToMove = GetOppositeColor(mSideToMove);
    mHash ^= s_BlackToMoveHash;

    ASSERT(IsValid());  // board position after the move must be valid

    // validate hash
    ASSERT(ComputeHash() == GetHash());

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
    mHash ^= s_BlackToMoveHash;

    ASSERT(IsValid());  // board position after the move must be valid

    // validate hash
    ASSERT(ComputeHash() == GetHash());

    return true;
}

static const int16_t c_seePieceValues[] =
{
    0,      // none
    100,    // pawn
    320,    // knight
    330,    // bishop
    500,    // rook
    900,    // queen
};

int32_t Position::StaticExchangeEvaluation(const Move& move) const
{
    Color sideToMove = mSideToMove;
    Bitboard occupied = Whites().Occupied() | Blacks().Occupied();
    Bitboard allAttackers = GetAttackers(move.toSquare, Color::White) | GetAttackers(move.toSquare, Color::Black);

    int32_t balance = 0;

    {
        const SidePosition& opponentSide = GetOpponentSide();
        const Piece capturedPiece = opponentSide.GetPieceAtSquare(move.toSquare);
        balance = c_seePieceValues[(uint32_t)capturedPiece];
    }

    {
        const SidePosition& side = GetCurrentSide();
        const Piece movedPiece = side.GetPieceAtSquare(move.fromSquare);
        balance -= c_seePieceValues[(uint32_t)movedPiece];
    }

    // If the balance is positive even if losing the moved piece,
    // the exchange is guaranteed to beat the threshold.
    if (balance >= 0)
    {
        return 1;
    }

    // "do" move
    occupied &= ~move.fromSquare.Bitboard();
    occupied |= move.toSquare.Bitboard();
    allAttackers &= occupied;

    sideToMove = GetOppositeColor(sideToMove);

    for (;;)
    {
        const SidePosition& side = mColors[(uint8_t)sideToMove];
        const Bitboard ourAttackers = allAttackers & side.Occupied();

        // no more attackers
        if (ourAttackers == 0) break;

        // find weakest attacker
        for (uint32_t pieceType = 1; pieceType <= 6u; ++pieceType)
        {
            const Bitboard pieceBitboard = ourAttackers & side.GetPieceBitBoard((Piece)pieceType);
            if (pieceBitboard)
            {
                uint32_t attackerSquare = UINT32_MAX;
                pieceBitboard.BitScanForward(attackerSquare);
                ASSERT(attackerSquare != UINT32_MAX);

                // remove attacker from occupied squares
                const Bitboard mask = 1ull << attackerSquare;
                ASSERT((occupied & mask) != 0);
                occupied &= ~mask;
                allAttackers &= occupied;

                // TODO update diagonal/vertical attackers

                balance = -balance - 1 - c_seePieceValues[pieceType];

                break;
            }
        }

        sideToMove = GetOppositeColor(sideToMove);

        // If the balance is non negative after giving away our piece then we win
        if (balance >= 0) 
        {
            break;
        }
    }

    // Side to move after the loop loses
    return mSideToMove != sideToMove;
}