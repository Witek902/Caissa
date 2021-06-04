#include "Position.hpp"
#include "Move.hpp"
#include "Bitboard.hpp"
#include "Evaluate.hpp"

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
        const SidePosition& pos = color == 0u ? mWhites : mBlacks;

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
    : mHash(0)
    , mSideToMove(Color::White)
    , mWhitesCastlingRights(CastlingRights_All)
    , mBlacksCastlingRights(CastlingRights_All)
    , mHalfMoveCount(0u)
    , mMoveCount(1u)
{}

void Position::SetPiece(const Square square, const Piece piece, const Color color)
{
    const Bitboard mask = square.Bitboard();
    SidePosition& pos = color == Color::White ? mWhites : mBlacks;

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
}

void Position::RemovePiece(const Square square, const Piece piece, const Color color)
{
    const Bitboard mask = square.Bitboard();
    SidePosition& pos = color == Color::White ? mWhites : mBlacks;
    Bitboard& targetBitboard = pos.GetPieceBitBoard(piece);

    const uint32_t colorIndex = (uint32_t)color;
    const uint32_t pieceIndex = (uint32_t)piece - 1;
    mHash ^= s_PiecePositionHash[colorIndex][pieceIndex][square.Index()];

    ASSERT((targetBitboard & mask) == mask);
    targetBitboard &= ~mask;
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

    mEnPassantSquare = Square();
}

Bitboard Position::GetAttackedSquares(Color side) const
{
    const SidePosition& currentSide = side == Color::White ? mWhites : mBlacks;
    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

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

static const int32_t c_PromotionScores[] =
{
    0,          // none
    0,          // pawn
    1000,       // knight
    1000,       // bishop
    1000,       // rook
    9000001,    // queen
};

static const int16_t c_PieceValues[] =
{
    0,      // none
    100,    // pawn
    320,    // knight
    330,    // bishop
    500,    // rook
    900,    // queen
};

static int32_t ComputeMvvLvaScore(const Piece attackingPiece, const Piece capturedPiece)
{
    return c_MvvLvaScoreBaseValue + 100 * (int32_t)capturedPiece - (int32_t)attackingPiece;
}

void Position::PushMove(const Move move, MoveList& outMoveList) const
{
    int32_t score = 0;

    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    if (move.isEnPassant)
    {
        score += c_MvvLvaScoreBaseValue;
    }
    else if (move.isCapture)
    {
        const Piece attackingPiece = move.piece;
        const Piece capturedPiece = opponentSide.GetPieceAtSquare(move.toSquare);
        score += ComputeMvvLvaScore(attackingPiece, capturedPiece);
    }
    else
    {
        score += ScoreQuietMove(*this, move);

        //// bonus for threats
        {
            Bitboard attacked = 0;
            if (move.piece == Piece::King)          attacked = Bitboard::GetKingAttacks(move.toSquare);
            else if (move.piece == Piece::Knight)   attacked = Bitboard::GetKnightAttacks(move.toSquare);
            else if (move.piece == Piece::Rook)     attacked = Bitboard::GetRookAttacks(move.toSquare);
            else if (move.piece == Piece::Bishop)   attacked = Bitboard::GetBishopAttacks(move.toSquare);
            else if (move.piece == Piece::Queen)    attacked = Bitboard::GetRookAttacks(move.toSquare) | Bitboard::GetBishopAttacks(move.toSquare);
            else if (move.piece == Piece::Pawn)     attacked = Bitboard::GetPawnAttacks(move.toSquare, mSideToMove);
            score += (opponentSide.Occupied() & attacked).Count();
        }
    }

    if (move.piece == Piece::Pawn && move.promoteTo != Piece::None)
    {
        const uint32_t pieceIndex = (uint32_t)move.promoteTo;
        ASSERT(pieceIndex > 1 && pieceIndex < 6);
        score += c_PromotionScores[pieceIndex];
    }

    outMoveList.PushMove(move, score);
}

void Position::GeneratePawnMoveList(MoveList& outMoveList, uint32_t flags) const
{
    const bool onlyTactical = flags & MOVE_GEN_ONLY_TACTICAL;
    const int32_t pawnDirection = mSideToMove == Color::White ? 1 : -1;
    const SidePosition& currentSide  = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;
    const uint32_t pawnStartingRank = mSideToMove == Color::White ? 1u : 6u;
    const uint32_t enPassantRank = mSideToMove == Color::White ? 5u : 2u;
    const uint32_t pawnFinalRank = mSideToMove == Color::White ? 6u : 1u;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    const auto generatePawnMove = [&](const Square fromSquare, const Square toSquare, bool isCapture, bool enPassant)
    {
        if (fromSquare.Rank() == pawnFinalRank) // pawn promotion
        {
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
        else if (!onlyTactical || isCapture)
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
                    Move move;
                    move.fromSquare = fromSquare;
                    move.toSquare = twoSquaresForward;
                    move.piece = Piece::Pawn;
                    move.promoteTo = Piece::None;
                    PushMove(move, outMoveList);
                }
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
        if (flags & MOVE_GEN_ONLY_TACTICAL)
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
        if (flags & MOVE_GEN_ONLY_TACTICAL)
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
        if (flags & MOVE_GEN_ONLY_TACTICAL)
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
        if (flags & MOVE_GEN_ONLY_TACTICAL)
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
    const bool onlyTactical = flags & MOVE_GEN_ONLY_TACTICAL;
    const uint32_t numKingOffsets = 8u;
    const int32_t kingFileOffsets[numKingOffsets] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const int32_t kingRankOffsets[numKingOffsets] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    const CastlingRights& currentSideCastlingRights = (mSideToMove == Color::White) ? mWhitesCastlingRights : mBlacksCastlingRights;
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
    if (onlyTactical)
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
        PushMove(move, outMoveList);
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

                Move move;
                move.fromSquare = square;
                move.toSquare = Square(2u, square.Rank());
                move.piece = Piece::King;
                move.isCastling = true;
                PushMove(move, outMoveList);
            }

            if ((currentSideCastlingRights & CastlingRights_ShortCastleAllowed) &&
                ((occupiedSquares & shortCastleCrossedSquares) == 0u) &&
                ((opponentAttacks & shortCastleKingCrossedSquares) == 0u))
            {
                // TODO Chess960 support?

                Move move;
                move.fromSquare = square;
                move.toSquare = Square(6u, square.Rank());
                move.piece = Piece::King;
                move.isCastling = true;
                PushMove(move, outMoveList);
            }
        }
    }
}

bool Position::IsSquareVisible(const Square square, const Color sideColor) const
{
    const SidePosition& side = sideColor == Color::White ? mWhites : mBlacks;
    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

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

    return bitboard != 0;
}

bool Position::IsInCheck(Color sideColor) const
{
    const SidePosition& currentSide = sideColor == Color::White ? mWhites : mBlacks;

    unsigned long kingSquareIndex;
    _BitScanForward64(&kingSquareIndex, currentSide.king);

    return IsSquareVisible(Square(kingSquareIndex), GetOppositeColor(sideColor));
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

    return Square();
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

    SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

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
        Square captureSquare;
        if (move.toSquare.Rank() == 5)  captureSquare = Square(move.toSquare.File(), 4u);
        if (move.toSquare.Rank() == 2)  captureSquare = Square(move.toSquare.File(), 3u);
        ASSERT(captureSquare.IsValid());

        RemovePiece(captureSquare, Piece::Pawn, GetOppositeColor(mSideToMove));
    }

    SetEnPassantSquare(move.piece == Piece::Pawn ? ExtractEnPassantSquareFromMove(move) : Square());

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

    const Color prevToMove = mSideToMove;

    mSideToMove = GetOppositeColor(mSideToMove);
    mHash ^= s_BlackToMoveHash;

    ASSERT(IsValid());  // board position after the move must be valid

    // validate hash
    ASSERT(ComputeHash() == GetHash());

    // can't be in check after move
    return !IsInCheck(prevToMove);
}