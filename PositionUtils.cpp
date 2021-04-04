#include "Position.hpp"
#include "Move.hpp"

Position::Position(const std::string& fenString)
    : Position()
{
    FromFEN(fenString);
}

bool Position::IsValid() const
{
    // validate piece counts
    if ((mWhites.pawns.Count() + mWhites.knights.Count() + mWhites.bishops.Count() + mWhites.rooks.Count() + mWhites.queens.Count() > 15u)) return false;
    if ((mBlacks.pawns.Count() + mBlacks.knights.Count() + mBlacks.bishops.Count() + mBlacks.rooks.Count() + mBlacks.queens.Count() > 15u)) return false;
    if (mWhites.pawns.Count() > 8u || mBlacks.pawns.Count() > 8u) return false;
    if (mWhites.knights.Count() > 9u || mBlacks.knights.Count() > 9u) return false;
    if (mWhites.bishops.Count() > 9u || mBlacks.bishops.Count() > 9u) return false;
    if (mWhites.rooks.Count() > 9u || mBlacks.rooks.Count() > 9u) return false;
    if (mWhites.queens.Count() > 9u || mBlacks.queens.Count() > 9u) return false;
    if (mWhites.king.Count() != 1u || mBlacks.king.Count() != 1u) return false;

    // validate pawn locations
    {
        bool pawnsValid = true;
        mWhites.pawns.Iterate([&](uint32_t index)
        {
            uint8_t pawnRank = Square(index).Rank();
            pawnsValid &= pawnRank >= 1u;   // pawns can't go backward
            pawnsValid &= pawnRank < 7u;    // unpromoted pawn
        });
        mBlacks.pawns.Iterate([&](uint32_t index)
        {
            uint8_t pawnRank = Square(index).Rank();
            pawnsValid &= pawnRank >= 1u;   // unpromoted pawn
            pawnsValid &= pawnRank < 7u;    // pawns can't go backward
        });
        if (!pawnsValid)
        {
            return false;
        }
    }

    if (mWhitesCastlingRights & CastlingRights_ShortCastleAllowed)
    {
        if (((mWhites.king & Bitboard(1ull << Square_e1)) == 0) ||
            ((mWhites.rooks & Bitboard(1ull << Square_h1)) == 0))
        {
            return false;
        }
    }
    if (mWhitesCastlingRights & CastlingRights_LongCastleAllowed)
    {
        if (((mWhites.king & Bitboard(1ull << Square_e1)) == 0) ||
            ((mWhites.rooks & Bitboard(1ull << Square_a1)) == 0))
        {
            return false;
        }
    }

    if (mBlacksCastlingRights & CastlingRights_ShortCastleAllowed)
    {
        if (((mBlacks.king & Bitboard(1ull << Square_e8)) == 0) ||
            ((mBlacks.rooks & Bitboard(1ull << Square_h8)) == 0))
        {
            return false;
        }
    }
    if (mBlacksCastlingRights & CastlingRights_LongCastleAllowed)
    {
        if (((mBlacks.king & Bitboard(1ull << Square_e8)) == 0) ||
            ((mBlacks.rooks & Bitboard(1ull << Square_a8)) == 0))
        {
            return false;
        }
    }

    return true;
}

bool Position::FromFEN(const std::string& fenString)
{
    int numSpaces = 0;
    int numRows = 1;
    for (const char p : fenString)
    {
        if (p == ' ')
        {
            ++numSpaces;
        }
        else if (p == '/')
        {
            ++numRows;
        }
    }

    if (numSpaces != 5 || numRows != 8)
    {
        fprintf(stderr, "Invalid FEN\n");
        return false;
    }

    const char* p = fenString.c_str();

    // board
    {
        uint8_t rank = 7;
        uint8_t file = 0;
        for (; *p != ' '; ++p)
        {
            const char ch = *p;

            if (isdigit(ch))
            {
                uint8_t skipCount = ch - '0';
                if (skipCount > 0 && skipCount <= 9)
                {
                    file += skipCount;
                }
                else
                {
                    fprintf(stderr, "Invalid FEN\n");
                    return false;
                }
            }
            else if (ch == '/')
            {
                file = 0;
                rank--;
            }
            else
            {
                const Square square(file, rank);
                const Color color = ch <= 90 ? Color::White : Color::Black;

                Piece piece;
                if (!CharToPiece(ch, piece))
                {
                    fprintf(stderr, "Invalid FEN\n");
                    return false;
                }

                SidePosition& side = color == Color::White ? mWhites : mBlacks;
                side.GetPieceBitBoard(piece) |= square.Bitboard();

                file++;
            }
        }
    }

    // next to move
    {
        const char nextToMove = (char)tolower(*++p);
        if (nextToMove == 'w')
        {
            mSideToMove = Color::White;
        }
        else if (nextToMove == 'b')
        {
            mSideToMove = Color::Black;
        }
        else
        {
            fprintf(stderr, "Invalid FEN: invalid next to move\n");
            return false;
        }
    }

    // castling rights
    {
        mWhitesCastlingRights = (CastlingRights)0;
        mBlacksCastlingRights = (CastlingRights)0;
        for (p += 2; *p != ' '; ++p)
        {
            switch (*p)
            {
            case 'K':
                mWhitesCastlingRights = CastlingRights(mWhitesCastlingRights | CastlingRights_ShortCastleAllowed);
                break;
            case 'Q':
                mWhitesCastlingRights = CastlingRights(mWhitesCastlingRights | CastlingRights_LongCastleAllowed);
                break;
            case 'k':
                mBlacksCastlingRights = CastlingRights(mBlacksCastlingRights | CastlingRights_ShortCastleAllowed);
                break;
            case 'q':
                mBlacksCastlingRights = CastlingRights(mBlacksCastlingRights | CastlingRights_LongCastleAllowed);
                break;
            case '-':
                break;
            default:
                fprintf(stderr, "Invalid FEN: invalid castling rights\n");
                return false;
            }
        }
    }

    std::string enPassantSquare;
    for (++p; *p != ' '; ++p)
    {
        enPassantSquare += *p;
    }

    if (enPassantSquare != "-")
    {
        mEnPassantSquare = Square::FromString(enPassantSquare);
        if (!mEnPassantSquare.IsValid())
        {
            fprintf(stderr, "Invalid FEN: invalid en passant square\n");
            return false;
        }
    }
    else
    {
        mEnPassantSquare = Square();
    }

    // TODO!
    // half-moves
    // move number

    //printf("\nhalf-moves since last pawn move/capture: "); for (++p; *p != ' '; ++p) putchar(*p);
    //printf("\n(full) move number: %s\n", ++p);

    mHash = ComputeHash();
    mAttackedByWhites = GetAttackedSquares(Color::White);
    mAttackedByBlacks = GetAttackedSquares(Color::Black);

    return IsValid();
}

std::string Position::ToFEN() const
{
    std::string str;

    for (uint8_t rank = 8u; rank-- > 0u; )
    {
        uint32_t numEmptySquares = 0u;
        for (uint8_t file = 0; file < 8u; ++file)
        {
            const Square square(file, rank);

            const Piece whitePiece = mWhites.GetPieceAtSquare(square);
            const Piece blackPiece = mBlacks.GetPieceAtSquare(square);

            if (whitePiece != Piece::None)
            {
                if (numEmptySquares)
                {
                    str += std::to_string(numEmptySquares);
                    numEmptySquares = 0;
                }
                str += PieceToChar(whitePiece);
            }
            else if (blackPiece != Piece::None)
            {
                if (numEmptySquares)
                {
                    str += std::to_string(numEmptySquares);
                    numEmptySquares = 0;
                }
                str += PieceToChar(blackPiece, false);
            }
            else // empty square
            {
                numEmptySquares++;
            }
        }

        if (numEmptySquares)
        {
            str += std::to_string(numEmptySquares);
            numEmptySquares = 0;
        }

        if (rank > 0)
        {
            str += '/';
        }
    }
    
    // side to move
    {
        str += ' ';
        str += mSideToMove == Color::White ? 'w' : 'b';
    }

    // castling rights
    {
        bool anyCastlingRights = false;
        str += ' ';
        if (mWhitesCastlingRights & CastlingRights_ShortCastleAllowed) str += 'K', anyCastlingRights = true;
        if (mWhitesCastlingRights & CastlingRights_LongCastleAllowed) str += 'Q', anyCastlingRights = true;
        if (mBlacksCastlingRights & CastlingRights_ShortCastleAllowed) str += 'k', anyCastlingRights = true;
        if (mBlacksCastlingRights & CastlingRights_LongCastleAllowed) str += 'q', anyCastlingRights = true;
        if (!anyCastlingRights) str += '-';
    }

    // en passant square
    {
        str += ' ';
        str += mEnPassantSquare.IsValid() ? mEnPassantSquare.ToString() : "-";
    }

    // half-moves since last pawn move/capture
    {
        str += ' ';
        str += std::to_string(mHalfMoveCount);
    }

    // full moves
    {
        str += ' ';
        str += std::to_string(mMoveCount);
    }

    return str;
}

std::string Position::Print() const
{
    std::string str;

    str += "   ---------------\n";

    // reverse, because first are printed higher ranks
    for (uint8_t rank = 8u; rank-- > 0u; )
    {
        str += '1' + rank;
        str += " |";

        for (uint8_t file = 0; file < 8u; ++file)
        {
            const Square square(file, rank);

            const Piece whitePiece = mWhites.GetPieceAtSquare(square);
            const Piece blackPiece = mBlacks.GetPieceAtSquare(square);

            if (whitePiece != Piece::None)
            {
                str += PieceToChar(whitePiece);
            }
            else if (blackPiece != Piece::None)
            {
                str += PieceToChar(blackPiece, false);
            }
            else // empty square
            {
                str += '.';
            }

            if (file < 7u)
            {
                str += " ";
            }
        }

        str += "|\n";
    }
    str += "   ---------------\n";
    str += "   a b c d e f g h\n";

    return str;
}

std::string Position::MoveToString(const Move& move) const
{
    ASSERT(move.piece != Piece::None);

    std::string str;

    str += move.fromSquare.ToString();
    str += move.toSquare.ToString();

    /*
    if (move.piece == Piece::Pawn)
    {
        str = move.toSquare.ToString();
        if (move.toSquare.Rank() == 7u && move.promoteTo != Piece::None)
        {
            str += "=";
            str += PieceToChar(move.promoteTo);
        }
    }
    else
    {
        if (move.piece == Piece::King && move.isCastling)
        {
            if (move.toSquare.File() == 2u)
            {
                str = "O-O-O";
            }
            else if (move.toSquare.File() == 6u)
            {
                str = "O-O";
            }
            else
            {
                str = "?";
            }
        }
        else
        {
            str = PieceToChar(move.piece);

            {
                const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
                const Bitboard movedPieceBitboard = currentSide.GetPieceBitBoard(move.piece);
                
                if (movedPieceBitboard.Count() > 1u)
                {
                    str += move.fromSquare.ToString();
                }
            }

            if (move.isCapture)
            {
                str += 'x';
            }
            str += move.toSquare.ToString();
        }
    }

    if (move.isCapture && move.isEnPassant)
    {
        str += " e.p.";
    }

    // TODO! check / checkmate
    // TODO! disambiguation
    */

    return str;
}

static bool IsMoveCastling(const Square& from, const Square& to)
{
    if (from == Square_e1)
    {
        return to == Square_c1 || to == Square_g1;
    }

    if (from == Square_e8)
    {
        return to == Square_c8 || to == Square_g8;
    }

    return false;
}

// parse move from string
Move Position::MoveFromString(const std::string& str) const
{
    if (str.length() < 4)
    {
        fprintf(stderr, "MoveFromString: Move string too short\n");
        return {};
    }

    const Square fromSquare = Square::FromString(str.substr(0, 2));
    const Square toSquare = Square::FromString(str.substr(2, 2));

    if (!fromSquare.IsValid() || !toSquare.IsValid())
    {
        fprintf(stderr, "MoveFromString: Failed to parse square\n");
        return {};
    }

    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Piece movedPiece = currentSide.GetPieceAtSquare(fromSquare);
    const Piece targetPiece = opponentSide.GetPieceAtSquare(toSquare);

    Move move;
    move.fromSquare = fromSquare;
    move.toSquare = toSquare;
    move.piece = movedPiece;
    move.isCapture = targetPiece != Piece::None;
    move.isEnPassant = false;
    move.isCastling = movedPiece == Piece::King && IsMoveCastling(fromSquare, toSquare);

    if (movedPiece == Piece::Pawn && toSquare == mEnPassantSquare)
    {
        move.isCapture = true;
        move.isEnPassant = true;
    }

    Piece promoteTo = Piece::None;
    if (str.length() > 4)
    {
        if (!CharToPiece(str[4], promoteTo))
        {
            fprintf(stderr, "MoveFromString: Failed to parse promotion\n");
            return {};
        }
    }
    move.promoteTo = promoteTo;

    return move;
}

bool Position::IsMoveValid(const Move& move) const
{
    ASSERT(move.IsValid());

    if (move.fromSquare == move.toSquare)
    {
        fprintf(stderr, "IsMoveValid: Cannot move piece to the same square\n");
        return {};
    }

    const SidePosition& currentSide = mSideToMove == Color::White ? mWhites : mBlacks;
    const SidePosition& opponentSide = mSideToMove == Color::White ? mBlacks : mWhites;

    const Piece movedPiece = currentSide.GetPieceAtSquare(move.fromSquare);
    const Piece targetPiece = opponentSide.GetPieceAtSquare(move.toSquare);

    if (movedPiece == Piece::None)
    {
        fprintf(stderr, "IsMoveValid: 'From' square does not contain a piece\n");
        return false;
    }

    if (opponentSide.GetPieceAtSquare(move.fromSquare) != Piece::None)
    {
        fprintf(stderr, "IsMoveValid: Cannot move opponent's piece\n");
        return false;
    }

    if (currentSide.GetPieceAtSquare(move.toSquare) != Piece::None)
    {
        fprintf(stderr, "IsMoveValid: Cannot capture own piece\n");
        return false;
    }

    if (targetPiece == Piece::King)
    {
        fprintf(stderr, "IsMoveValid: Cannot capture king\n");
        return false;
    }

    if (move.isEnPassant && move.piece != Piece::Pawn)
    {
        fprintf(stderr, "IsMoveValid: Only pawn can do en passant capture");
        return false;
    }

    MoveList moveList;

    if (move.piece == Piece::Pawn)
    {
        if ((mSideToMove == Color::White && move.toSquare.Rank() == 7) ||
            (mSideToMove == Color::Black && move.toSquare.Rank() == 0))
        {
            if (move.promoteTo != Piece::Queen && move.promoteTo != Piece::Rook &&
                move.promoteTo != Piece::Bishop && move.promoteTo != Piece::Knight)
            {
                fprintf(stderr, "IsMoveValid: Invalid promotion\n");
                return false;
            }
        }

        GeneratePawnMoveList(moveList);
    }
    else if (move.piece == Piece::Knight)
    {
        GenerateKnightMoveList(moveList);
    }
    else if (move.piece == Piece::Bishop)
    {
        GenerateBishopMoveList(moveList);
    }
    else if (move.piece == Piece::Rook)
    {
        GenerateRookMoveList(moveList);
    }
    else if (move.piece == Piece::Queen)
    {
        GenerateQueenMoveList(moveList);
    }
    else if (move.piece == Piece::King)
    {
        GenerateKingMoveList(moveList);
    }

    bool isMoveValid = false;

    for (uint32_t i = 0; i < moveList.Size(); ++i)
    {
        const Move refMove = moveList.GetMove(i);

        bool isSame =
            refMove.fromSquare == move.fromSquare &&
            refMove.toSquare == move.toSquare &&
            refMove.piece == move.piece &&
            refMove.isCapture == move.isCapture;

        if (move.piece == Piece::King)
        {
            isSame &= refMove.isCastling == move.isCastling;
        }

        if (move.piece == Piece::Pawn)
        {
            isSame &= refMove.promoteTo == move.promoteTo;
            isSame &= refMove.isEnPassant == move.isEnPassant;
        }

        if (isSame)
        {
            isMoveValid = true;
            break;
        }
    }

    return isMoveValid;
}
