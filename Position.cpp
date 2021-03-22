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
    assert(square.IsValid());

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
    , mMoveCount(1u)
{}

void Position::SetPieceAtSquare(const Square square, Piece piece, Color color)
{
    assert(square.IsValid());

    SidePosition& pos = color == Color::White ? mWhites : mBlacks;
    pos.SetPieceAtSquare(square, piece);
}

int Position::IsCheck() const
{
    return 0;
}

float Position::Evaluate() const
{
    return 0.0f;
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
    const uint32_t pawnFinalRank = mSideToMove == Color::White ? 6u : 1u;

    const Bitboard occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

    const auto generatePawnMove = [&](const Square fromSquare, const Square toSquare, bool isCapture)
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
            }
        }
        else // simple move forward
        {
            Move& move = outMoveList.PushMove();
            move.fromSquare = fromSquare;
            move.toSquare = toSquare;
            move.piece = Piece::Pawn;
            move.isCapture = isCapture;
        }
    };

    currentSide.pawns.Iterate([&](uint32_t fromIndex)
    {
        const Square fromSquare(fromIndex);
        const Square squareForward(fromSquare.Index() + pawnDirection * 8); // next rank

        // there should be no pawn in first or last rank
        assert(fromSquare.Rank() > 0u && fromSquare.Rank() < 7u);

        // can move forward only to non-occupied squares
        if ((occupiedSquares & squareForward.Bitboard()) == 0u)
        {
            generatePawnMove(fromSquare, squareForward, false);

            if (fromSquare.Rank() == pawnStartingRank) // move by two ranks
            {
                Move& move = outMoveList.PushMove();
                move.fromSquare = fromSquare;
                move.toSquare = Square(fromSquare.Index() + pawnDirection * 16); // two ranks up
                move.piece = Piece::Pawn;
                move.isCapture = false;
            }
        }

        // capture on the left
        if (fromSquare.Rank() > 0u)
        {
            const Square toSquare(fromSquare.Index() + pawnDirection * 8 - 1);
            if (toSquare.Bitboard() & opponentSide.OccupiedExcludingKing())
            {
                generatePawnMove(fromSquare, toSquare, true);
            }
        }

        // capture on the right
        if (fromSquare.Rank() < 7u)
        {
            const Square toSquare(fromSquare.Index() + pawnDirection * 8 + 1);
            if (toSquare.Bitboard() & opponentSide.OccupiedExcludingKing())
            {
                generatePawnMove(fromSquare, toSquare, true);
            }
        }

        // TODO! en passant
    });
}

void Position::GenerateKnightMoveList(MoveList& outMoveList) const
{
    const uint32_t numKnightOffsets = 8u;
    const int32_t knightFileOffsets[numKnightOffsets] = { 1, 2, 2, 1, -1, -2, -2, -1 };
    const int32_t knightRankOffsets[numKnightOffsets] = { 2, 1, -1, -2, -2, -1, 1, 2 };

    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    currentSide.knights.Iterate([&](uint32_t fromIndex)
    {
        const Square square(fromIndex);

        // TODO use bitboards to generate jumps
        // can then skip out-of-board check, etc.
        for (uint32_t i = 0; i < numKnightOffsets; ++i)
        {
            const int32_t targetFile = (int32_t)square.File() + knightFileOffsets[i];
            const int32_t targetRank = (int32_t)square.Rank() + knightRankOffsets[i];

            // out of board
            if (targetFile < 0 || targetRank < 0 || targetFile >= 8 || targetRank >= 8) continue;

            const Square targetSquare((uint8_t)targetFile, (uint8_t)targetRank);

            // can't capture own piece
            if (currentSide.Occupied() & targetSquare.Bitboard()) continue;

            // can't capture king
            if (opponentSide.king & targetSquare.Bitboard()) continue;

            Move& move = outMoveList.PushMove();
            move.fromSquare = square;
            move.toSquare = targetSquare;
            move.piece = Piece::Knight;
            move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        }
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

__declspec(noinline)
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

    const uint64_t occupiedSquares = mWhites.Occupied() | mBlacks.Occupied();

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

    unsigned long kingSquareIndex;
    if (0 == _BitScanForward64(&kingSquareIndex, currentSide.king))
    {
        return;
    }

    const Square square(kingSquareIndex);

    // TODO use bitboards to generate jumps - can then skip out-of-board check, etc.
    for (uint32_t i = 0; i < numKingOffsets; ++i)
    {
        const int32_t targetFile = (int32_t)square.File() + kingFileOffsets[i];
        const int32_t targetRank = (int32_t)square.Rank() + kingRankOffsets[i];

        // out of board
        if (targetFile < 0 || targetRank < 0 || targetFile >= 8 || targetRank >= 8) continue;

        const Square targetSquare((uint8_t)targetFile, (uint8_t)targetRank);

        // can't capture own piece
        if (currentSide.Occupied() & targetSquare.Bitboard()) continue;

        Move& move = outMoveList.PushMove();
        move.fromSquare = square;
        move.toSquare = targetSquare;
        move.piece = Piece::King;
        move.isCapture = opponentSide.OccupiedExcludingKing() & targetSquare.Bitboard();
        move.isCastling = false;
    }

    if (currentSide.castlingRights & CastlingRights_All)
    {
        // TODO simplify this
        const Bitboard longCastleCrossedSquares = (1ull << (kingSquareIndex - 1)) | (1ull << (kingSquareIndex - 2)) | (1ull << (kingSquareIndex - 3));
        const Bitboard shortCastleCrossedSquares = (1ull << (kingSquareIndex + 1)) | (1ull << (kingSquareIndex + 2));

        if ((currentSide.castlingRights & CastlingRights_LongCastleAllowed) &&
            ((currentSide.Occupied() & longCastleCrossedSquares) == 0u))
        {
            // TODO! check if any square that king crosses is not checked
            // TODO Chess960 support?

            Move& move = outMoveList.PushMove();
            move.fromSquare = square;
            move.toSquare = Square(2u, square.Rank());
            move.piece = Piece::King;
            move.isCapture = false;
            move.isCastling = true;
        }

        if ((currentSide.castlingRights & CastlingRights_ShortCastleAllowed) &&
            ((currentSide.Occupied() & shortCastleCrossedSquares) == 0u))
        {
            // TODO! check if any square that king crosses is not checked
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

bool Position::IsMoveValid(const Move& move) const
{
    if (move.fromSquare.Index() == move.toSquare.Index())
    {
        return false;
    }

    return false;
}

bool Position::DoMove(const Move& move)
{
    assert(move.fromSquare.IsValid());
    assert(move.toSquare.IsValid());
    assert(move.fromSquare.Index() != move.toSquare.Index());

    SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    assert(move.piece == currentSide.GetPieceAtSquare(move.fromSquare));
    assert(Piece::None == currentSide.GetPieceAtSquare(move.toSquare)); // can't take own piece

    // move piece
    Bitboard& pieceBitboard = currentSide.GetPieceBitBoard(move.piece);
    pieceBitboard |= move.toSquare.Bitboard();
    pieceBitboard &= ~move.fromSquare.Bitboard();

    // TODO skip this if not capture
    opponentSide.SetPieceAtSquare(move.toSquare, Piece::None);

    // TODO! castling
    // TODO! update en passant state
    // TODO! en passant capture
    // TODO! promotion

    return false;
}