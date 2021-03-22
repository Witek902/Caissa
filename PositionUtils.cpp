#include "Position.hpp"
#include "Move.hpp"

Position::Position(const std::string& fenString)
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

                SetPieceAtSquare(square, piece, color);

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
        mWhites.castlingRights = (CastlingRights)0;
        mBlacks.castlingRights = (CastlingRights)0;
        for (p += 2; *p != ' '; ++p)
        {
            switch (*p)
            {
            case 'K':
                mWhites.castlingRights = CastlingRights(mWhites.castlingRights | CastlingRights_ShortCastleAllowed);
                break;
            case 'Q':
                mWhites.castlingRights = CastlingRights(mWhites.castlingRights | CastlingRights_LongCastleAllowed);
                break;
            case 'k':
                mBlacks.castlingRights = CastlingRights(mBlacks.castlingRights | CastlingRights_ShortCastleAllowed);
                break;
            case 'q':
                mBlacks.castlingRights = CastlingRights(mBlacks.castlingRights | CastlingRights_LongCastleAllowed);
                break;
            case '-':
                break;
            default:
                fprintf(stderr, "Invalid FEN: invalid castling rights\n");
                return false;
            }
        }
    }

    // TODO!
    // en passant
    // half-moves
    // move number

    //const char next = tolower(*++p);
    //printf("\n\n\nnext move: %s\ncastling availability: ", next == 'w' ? "white" : "black");
    //for (p += 2; *p != ' '; ++p) putchar(*p);

    //printf("\nen passant: "); for (++p; *p != ' '; ++p) putchar(*p);
    //printf("\nhalf-moves since last pawn move/capture: "); for (++p; *p != ' '; ++p) putchar(*p);
    //printf("\n(full) move number: %s\n", ++p);

    return IsValid();
}

std::string Position::ToFEN() const
{
    return "";
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
    assert(move.piece != Piece::None);

    std::string str;

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

    // TODO! disambiguation
    // TODO! en passant

    return str;
}
