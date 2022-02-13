#include "PositionUtils.hpp"

#include "Position.hpp"
#include "MoveList.hpp"
#include "Time.hpp"
#include "Material.hpp"

#include <random>

#include <immintrin.h>

static_assert(sizeof(PackedPosition) == 28, "Invalid packed position size");

bool PackPosition(const Position& inPos, PackedPosition& outPos)
{
    outPos.occupied = inPos.Whites().Occupied() | inPos.Blacks().Occupied();
    outPos.moveCount = inPos.GetMoveCount();
    outPos.sideToMove = inPos.GetSideToMove() == Color::White ? 0 : 1;
    outPos.halfMoveCount = inPos.GetHalfMoveCount();
    outPos.castlingRights = (uint8_t)inPos.GetWhitesCastlingRights() | ((uint8_t)inPos.GetBlacksCastlingRights() << 2);
    outPos.enPassantFile = inPos.GetEnPassantSquare().IsValid() ? inPos.GetEnPassantSquare().File() : 0xF;
    memset(outPos.piecesData, 0, sizeof(outPos.piecesData));

    if (outPos.occupied.Count() > 32)
    {
        return false;
    }

    uint32_t offset = 0;

    outPos.occupied.Iterate([&](uint32_t index)
    {
        Piece piece = Piece::None;
        uint8_t value = 0;

        if ((piece = inPos.Whites().GetPieceAtSquare(Square(index))) != Piece::None)
        {
            value = (uint8_t)piece - (uint8_t)Piece::Pawn;
        }
        else if ((piece = inPos.Blacks().GetPieceAtSquare(Square(index))) != Piece::None)
        {
            value = (uint8_t)piece - (uint8_t)Piece::Pawn + 8;
        }
        else
        {
            DEBUG_BREAK();
        }

        if (offset % 2 == 0)
        {
            outPos.piecesData[offset / 2] = value;
        }
        else
        {
            outPos.piecesData[offset / 2] |= (value << 4);
        }

        offset++;
    });

    return true;
}

bool UnpackPosition(const PackedPosition& inPos, Position& outPos)
{
    outPos = Position();

    uint32_t offset = 0;
    bool success = true;

    inPos.occupied.Iterate([&](uint32_t index)
    {
        const uint8_t value = (inPos.piecesData[offset / 2] >> (4 * (offset % 2))) & 0xF;

        if (value <= (uint8_t)Piece::King)
        {
            const Piece piece = (Piece)(value + (uint8_t)Piece::Pawn);
            outPos.SetPiece(Square(index), piece, Color::White);
        }
        else if (value >= 8 && value <= 8 + (uint8_t)Piece::King)
        {
            const Piece piece = (Piece)(value - 8 + (uint8_t)Piece::Pawn);
            outPos.SetPiece(Square(index), piece, Color::Black);
        }
        else
        {
            success = false;
        }

        offset++;
    });

    outPos.SetSideToMove((Color)inPos.sideToMove);
    outPos.SetMoveCount(inPos.moveCount);
    outPos.SetHalfMoveCount(inPos.halfMoveCount);
    outPos.SetWhitesCastlingRights(CastlingRights(inPos.castlingRights & 0b11));
    outPos.SetBlacksCastlingRights(CastlingRights((inPos.castlingRights >> 2) & 0b11));

    if (inPos.enPassantFile < 8)
    {
        outPos.SetEnPassantSquare(Square(inPos.enPassantFile, inPos.sideToMove == 0 ? 5 : 2));
    }

    return success;
}

Position::Position(const std::string& fenString)
    : Position()
{
    FromFEN(fenString);
}

bool Position::operator == (const Position& rhs) const
{
    const bool result =
        Whites() == rhs.Whites() &&
        Blacks() == rhs.Blacks() &&
        mSideToMove == rhs.mSideToMove &&
        mEnPassantSquare == rhs.mEnPassantSquare &&
        mWhitesCastlingRights == rhs.mWhitesCastlingRights &&
        mBlacksCastlingRights == rhs.mBlacksCastlingRights;

    if (result)
    {
        ASSERT(mHash == rhs.mHash);
    }

    return result;
}

bool Position::IsValid(bool strict) const
{
    // validate piece counts
    if (Whites().king.Count() != 1u || Blacks().king.Count() != 1u) return false;
    if (strict)
    {
        if ((Whites().pawns.Count() + Whites().knights.Count() + Whites().bishops.Count() + Whites().rooks.Count() + Whites().queens.Count() > 15u)) return false;
        if ((Blacks().pawns.Count() + Blacks().knights.Count() + Blacks().bishops.Count() + Blacks().rooks.Count() + Blacks().queens.Count() > 15u)) return false;
        if (Whites().pawns.Count() > 8u || Blacks().pawns.Count() > 8u) return false;
        if (Whites().knights.Count() > 9u || Blacks().knights.Count() > 9u) return false;
        if (Whites().bishops.Count() > 9u || Blacks().bishops.Count() > 9u) return false;
        if (Whites().rooks.Count() > 9u || Blacks().rooks.Count() > 9u) return false;
        if (Whites().queens.Count() > 9u || Blacks().queens.Count() > 9u) return false;
    }

    // validate pawn locations
    {
        bool pawnsValid = true;
        Whites().pawns.Iterate([&](uint32_t index)
        {
            uint8_t pawnRank = Square(index).Rank();
            if (strict) pawnsValid &= pawnRank >= 1u;   // pawns can't go backward
            pawnsValid &= pawnRank < 7u;                // unpromoted pawn
        });
        Blacks().pawns.Iterate([&](uint32_t index)
        {
            uint8_t pawnRank = Square(index).Rank();
            pawnsValid &= pawnRank >= 1u;               // unpromoted pawn
            if (strict) pawnsValid &= pawnRank < 7u;    // pawns can't go backward
        });
        if (!pawnsValid)
        {
            return false;
        }
    }

    if (mWhitesCastlingRights & CastlingRights_ShortCastleAllowed)
    {
        if (((Whites().king & Bitboard(1ull << Square_e1)) == 0) ||
            ((Whites().rooks & Bitboard(1ull << Square_h1)) == 0))
        {
            return false;
        }
    }
    if (mWhitesCastlingRights & CastlingRights_LongCastleAllowed)
    {
        if (((Whites().king & Bitboard(1ull << Square_e1)) == 0) ||
            ((Whites().rooks & Bitboard(1ull << Square_a1)) == 0))
        {
            return false;
        }
    }

    if (mBlacksCastlingRights & CastlingRights_ShortCastleAllowed)
    {
        if (((Blacks().king & Bitboard(1ull << Square_e8)) == 0) ||
            ((Blacks().rooks & Bitboard(1ull << Square_h8)) == 0))
        {
            return false;
        }
    }
    if (mBlacksCastlingRights & CastlingRights_LongCastleAllowed)
    {
        if (((Blacks().king & Bitboard(1ull << Square_e8)) == 0) ||
            ((Blacks().rooks & Bitboard(1ull << Square_a8)) == 0))
        {
            return false;
        }
    }

    return true;
}

bool Position::FromFEN(const std::string& fenString)
{
    *this = Position();

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

    if (numSpaces < 3 || numSpaces > 5 || numRows != 8)
    {
        fprintf(stderr, "Invalid FEN: wrong syntax\n");
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
                    fprintf(stderr, "Invalid FEN: failed to parse board state\n");
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
                    fprintf(stderr, "Invalid FEN: failed to parse board state\n");
                    return false;
                }

                SidePosition& side = color == Color::White ? mColors[0] : mColors[1];
                side.GetPieceBitBoard(piece) |= square.GetBitboard();

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
    for (++p; *p != ' ' && *p != 0; ++p)
    {
        enPassantSquare += *p;
    }

    if (enPassantSquare != "-")
    {
        mEnPassantSquare = Square::FromString(enPassantSquare);
        if (!mEnPassantSquare.IsValid())
        {
            fprintf(stderr, "Invalid FEN: failed to parse en passant square\n");
            return false;
        }

        if (mSideToMove == Color::White)
        {
            if (mEnPassantSquare.Rank() != 5)
            {
                fprintf(stderr, "Invalid FEN: invalid en passant square\n");
                return false;
            }
            if (Blacks().GetPieceAtSquare(mEnPassantSquare) != Piece::None ||
                Blacks().GetPieceAtSquare(mEnPassantSquare.South()) != Piece::Pawn)
            {
                fprintf(stderr, "Invalid FEN: invalid en passant square\n");
                return false;
            }
        }
        else
        {
            if (mEnPassantSquare.Rank() != 2)
            {
                fprintf(stderr, "Invalid FEN: invalid en passant square\n");
                return false;
            }
            if (Whites().GetPieceAtSquare(mEnPassantSquare) != Piece::None ||
                Whites().GetPieceAtSquare(mEnPassantSquare.North()) != Piece::Pawn)
            {
                fprintf(stderr, "Invalid FEN: invalid en passant square\n");
                return false;
            }
        }
    }
    else
    {
        mEnPassantSquare = Square::Invalid();
    }

    // parse half-moves counter
    {
        std::string halfMovesStr;
        for (++p; *p != ' ' && *p != 0; ++p)
        {
            halfMovesStr += *p;
        }

        if (!halfMovesStr.empty())
        {
            mHalfMoveCount = (int16_t)atoi(halfMovesStr.c_str());
        }
        else
        {
            mHalfMoveCount = 0;
        }
    }

    // parse moves number
    {
        std::string moveNumberStr;
        for (++p; *p != ' ' && *p != 0; ++p)
        {
            moveNumberStr += *p;
        }

        if (!moveNumberStr.empty())
        {
            mMoveCount = (int16_t)std::max(1, atoi(moveNumberStr.c_str()));
        }
        else
        {
            mMoveCount = 1;
        }
    }

    mHash = ComputeHash();

    if (!IsValid())
    {
        fprintf(stderr, "Invalid FEN: invalid position\n");
        return false;
    }

    if (IsInCheck(GetOppositeColor(mSideToMove)))
    {
        fprintf(stderr, "Invalid FEN: opponent cannot be in check\n");
        return false;
    }

    return true;
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

            const Piece whitePiece = Whites().GetPieceAtSquare(square);
            const Piece blackPiece = Blacks().GetPieceAtSquare(square);

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

            const Piece whitePiece = Whites().GetPieceAtSquare(square);
            const Piece blackPiece = Blacks().GetPieceAtSquare(square);

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

std::string Position::MoveToString(const Move& move, MoveNotation notation) const
{
    ASSERT(move.GetPiece() != Piece::None);

    Position afterMove(*this);
    if (!afterMove.DoMove(move))
    {
        return "illegal move";
    }

    std::string str;

    if (notation == MoveNotation::LAN)
    {
        str = move.ToString();
    }
    else if (notation == MoveNotation::SAN)
    {
        if (move.GetPiece() == Piece::Pawn)
        {
            if (move.IsCapture())
            {
                str += 'a' + move.FromSquare().File();
                str += 'x';
            }

            str += move.ToSquare().ToString();
            if (move.GetPromoteTo() != Piece::None)
            {
                str += "=";
                str += PieceToChar(move.GetPromoteTo());
            }
        }
        else
        {
            if (move.GetPiece() == Piece::King && move.IsCastling())
            {
                if (move.ToSquare().File() == 2u)
                {
                    str = "O-O-O";
                }
                else if (move.ToSquare().File() == 6u)
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
                str = PieceToChar(move.GetPiece());

                {
                    const SidePosition& currentSide = GetCurrentSide();
                    const Bitboard movedPieceBitboard = currentSide.GetPieceBitBoard(move.GetPiece());

                    bool ambiguousFile = false;
                    bool ambiguousRank = false;
                    bool ambiguousPiece = false;
                    {
                        MoveList moves;
                        GenerateMoveList(moves);

                        for (uint32_t i = 0; i < moves.numMoves; ++i)
                        {
                            const Move& otherMove = moves[i].move;
                            if (otherMove.GetPiece() == move.GetPiece() &&
                                otherMove.ToSquare() == move.ToSquare() &&
                                otherMove.FromSquare() != move.FromSquare())
                            {
                                if (otherMove.FromSquare().File() == move.FromSquare().File())  ambiguousFile = true;
                                if (otherMove.FromSquare().Rank() == move.FromSquare().Rank())  ambiguousRank = true;
                                ambiguousPiece = true;
                            }
                        }
                    }

                    if (ambiguousPiece)
                    {
                        if (ambiguousFile && ambiguousRank)
                        {
                            str += move.FromSquare().ToString();
                        }
                        else if (ambiguousFile)
                        {
                            str += '1' + move.FromSquare().Rank();
                        }
                        else
                        {
                            str += 'a' + move.FromSquare().File();
                        }
                    }
                }

                if (move.IsCapture())
                {
                    str += 'x';
                }
                str += move.ToSquare().ToString();
            }
        }

        if (afterMove.IsMate())
        {
            str += '#';
        }
        else if (afterMove.IsInCheck(afterMove.GetSideToMove()))
        {
            str += '+';
        }
    }
    else
    {
        DEBUG_BREAK();
    }

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

Move Position::MoveFromPacked(const PackedMove& packedMove) const
{
    if (!packedMove.FromSquare().IsValid())
    {
        return Move();
    }

    const Piece movedPiece = GetCurrentSide().GetPieceAtSquare(packedMove.FromSquare());
    if (movedPiece == Piece::None)
    {
        return Move();
    }

    MoveList moves;

    // TODO
    // instead of generating moves of all pieces of given type,
    // generate only moves of the moved piece
    switch (movedPiece)
    {
    case Piece::Pawn:   GeneratePawnMoveList(moves); break;
    case Piece::Knight: GenerateKnightMoveList(moves); break;
    case Piece::Rook:   GenerateRookMoveList(moves); break;
    case Piece::Bishop: GenerateBishopMoveList(moves); break;
    case Piece::Queen:  GenerateQueenMoveList(moves); break;
    case Piece::King:   GenerateKingMoveList(moves); break;
    }

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        if (moves[i].move == packedMove)
        {
            return moves[i].move;
        }
    }

    return Move();
}

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

    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Piece movedPiece = currentSide.GetPieceAtSquare(fromSquare);
    const Piece targetPiece = opponentSide.GetPieceAtSquare(toSquare);

    bool isCapture = targetPiece != Piece::None;
    bool isEnPassant = false;
    bool isCastling = movedPiece == Piece::King && IsMoveCastling(fromSquare, toSquare);

    if (movedPiece == Piece::Pawn && toSquare == mEnPassantSquare)
    {
        isCapture = true;
        isEnPassant = true;
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

    return Move::Make(fromSquare, toSquare, movedPiece, promoteTo, isCapture, isEnPassant, isCastling);
}

bool Position::IsMoveValid(const Move& move) const
{
#ifndef CONFIGURATION_FINAL
    if (!move.IsValid())
    {
        std::cout << "Invalid move for position: " << ToFEN() << std::endl;
        DEBUG_BREAK();
    }
#endif // CONFIGURATION_FINAL

    if (move.FromSquare() == move.ToSquare())
    {
        fprintf(stderr, "IsMoveValid: Cannot move piece to the same square\n");
        return {};
    }

    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Piece movedPiece = currentSide.GetPieceAtSquare(move.FromSquare());
    const Piece targetPiece = opponentSide.GetPieceAtSquare(move.ToSquare());

    if (movedPiece == Piece::None)
    {
        fprintf(stderr, "IsMoveValid: 'From' square does not contain a piece\n");
        return false;
    }

    if (opponentSide.GetPieceAtSquare(move.FromSquare()) != Piece::None)
    {
        fprintf(stderr, "IsMoveValid: Cannot move opponent's piece\n");
        return false;
    }

    if (currentSide.GetPieceAtSquare(move.ToSquare()) != Piece::None)
    {
        fprintf(stderr, "IsMoveValid: Cannot capture own piece\n");
        return false;
    }

    if (targetPiece == Piece::King)
    {
        fprintf(stderr, "IsMoveValid: Cannot capture king\n");
        return false;
    }

    if (move.IsEnPassant() && move.GetPiece() != Piece::Pawn)
    {
        fprintf(stderr, "IsMoveValid: Only pawn can do en passant capture");
        return false;
    }

    MoveList moveList;

    if (move.GetPiece() == Piece::Pawn)
    {
        if ((mSideToMove == Color::White && move.ToSquare().Rank() == 7) ||
            (mSideToMove == Color::Black && move.ToSquare().Rank() == 0))
        {
            if (move.GetPromoteTo() != Piece::Queen && move.GetPromoteTo() != Piece::Rook &&
                move.GetPromoteTo() != Piece::Bishop && move.GetPromoteTo() != Piece::Knight)
            {
                fprintf(stderr, "IsMoveValid: Invalid promotion\n");
                return false;
            }
        }

        GeneratePawnMoveList(moveList);
    }
    else if (move.GetPiece() == Piece::Knight)
    {
        GenerateKnightMoveList(moveList);
    }
    else if (move.GetPiece() == Piece::Bishop)
    {
        GenerateBishopMoveList(moveList);
    }
    else if (move.GetPiece() == Piece::Rook)
    {
        GenerateRookMoveList(moveList);
    }
    else if (move.GetPiece() == Piece::Queen)
    {
        GenerateQueenMoveList(moveList);
    }
    else if (move.GetPiece() == Piece::King)
    {
        GenerateKingMoveList(moveList);
    }

    bool isMoveValid = false;

    for (uint32_t i = 0; i < moveList.Size(); ++i)
    {
        const Move refMove = moveList.GetMove(i);

        bool isSame =
            refMove.FromSquare() == move.FromSquare() &&
            refMove.ToSquare() == move.ToSquare() &&
            refMove.GetPiece() == move.GetPiece() &&
            refMove.IsCapture() == move.IsCapture();

        if (move.GetPiece() == Piece::King)
        {
            isSame &= refMove.IsCastling() == move.IsCastling();
        }

        if (move.GetPiece() == Piece::Pawn)
        {
            isSame &= refMove.GetPromoteTo() == move.GetPromoteTo();
            isSame &= refMove.IsEnPassant() == move.IsEnPassant();
        }

        if (isSame)
        {
            isMoveValid = true;
            break;
        }
    }

    return isMoveValid;
}

bool Position::IsMoveValid_Fast(const PackedMove& move) const
{
    ASSERT(move.IsValid());
    ASSERT(move.FromSquare() != move.ToSquare());

    const SidePosition& currentSide = GetCurrentSide();
    const SidePosition& opponentSide = GetOpponentSide();

    const Piece movedPiece = currentSide.GetPieceAtSquare(move.FromSquare());
    const Piece targetPiece = opponentSide.GetPieceAtSquare(move.ToSquare());

    if (movedPiece == Piece::None)
    {
        return false;
    }

    if (opponentSide.GetPieceAtSquare(move.FromSquare()) != Piece::None)
    {
        return false;
    }

    if (currentSide.GetPieceAtSquare(move.ToSquare()) != Piece::None)
    {
        // cannot capture own piece
        return false;
    }

    if (targetPiece == Piece::King)
    {
        // cannot capture king
        return false;
    }

    return true;
}

uint64_t Position::Perft(uint32_t depth, bool print) const
{
    if (print)
    {
        std::cout << "Running Perft... depth=" << depth << std::endl;
    }

    const TimePoint startTime = TimePoint::GetCurrent();

    MoveList moveList;
    GenerateMoveList(moveList);

    uint64_t nodes = 0;
    for (uint32_t i = 0; i < moveList.Size(); i++)
    {
        const Move& move = moveList.GetMove(i);

        Position child = *this;
        if (!child.DoMove(move))
        {
            continue;
        }

        uint64_t numChildNodes = depth == 1 ? 1 : child.Perft(depth - 1, false);

        if (print)
        {
            std::cout << move.ToString() << ": " << numChildNodes << std::endl;
        }

        nodes += numChildNodes;
    }

    const TimePoint endTime = TimePoint::GetCurrent();

    if (print)
    {
        std::cout << "Total nodes: " << nodes << std::endl;
        std::cout << "Time: " << (endTime - startTime).ToSeconds() << " seconds" << std::endl;
    }

    return nodes;
}

bool GenerateRandomPosition(std::mt19937_64& randomGenerator, const MaterialKey& material, Position& outPosition)
{
    const auto genLegalSquare = [&randomGenerator](const Bitboard mask) -> Square
    {
        std::uniform_int_distribution<uint32_t> distr;

        if (!mask) return Square::Invalid();

        const uint32_t numLegalSquares = mask.Count();
        const uint32_t maskedSquareIndex = distr(randomGenerator) % numLegalSquares;

        const uint64_t squareMask = ParallelBitsDeposit(1ull << maskedSquareIndex, mask);
        ASSERT(squareMask);

        return Square(FirstBitSet(squareMask));
    };

    std::uniform_int_distribution<uint32_t> distr;

    for (;;)
    {
        outPosition = Position();

        Bitboard occupied = 0;

        // place white king on any square
        const Square whiteKingSq = Square(distr(randomGenerator) % 64);
        occupied |= whiteKingSq.GetBitboard();
        outPosition.SetPiece(whiteKingSq, Piece::King, Color::White);

        // place black king on any square
        Square blackKingSq;
        {
            const Bitboard mask = ~whiteKingSq.GetBitboard() & ~Bitboard::GetKingAttacks(whiteKingSq);
            blackKingSq = genLegalSquare(mask);
            ASSERT(blackKingSq.IsValid());
            occupied |= blackKingSq.GetBitboard();
            outPosition.SetPiece(blackKingSq, Piece::King, Color::Black);
        }

        // generate white pawns on ranks 1-7, they cannot attack black king
        for (uint32_t i = 0; i < material.numWhitePawns; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::RankBitboard(0) & ~Bitboard::RankBitboard(7) & ~Bitboard::GetPawnAttacks(blackKingSq, Color::Black);
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Pawn, Color::White);
        }

        // TODO generate en-passant square if possible

        // generate black pawns on ranks 1-7
        for (uint32_t i = 0; i < material.numBlackPawns; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::RankBitboard(0) & ~Bitboard::RankBitboard(7);
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Pawn, Color::Black);
        }

        // generate white queens, they cannot attack black king
        for (uint32_t i = 0; i < material.numWhiteQueens; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GenerateRookAttacks(blackKingSq, occupied) & ~Bitboard::GenerateBishopAttacks(blackKingSq, occupied);
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Queen, Color::White);
        }

        // generate black queens
        for (uint32_t i = 0; i < material.numBlackQueens; ++i)
        {
            const Bitboard mask = ~occupied;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Queen, Color::Black);
        }

        // generate white rooks, they cannot attack black king
        for (uint32_t i = 0; i < material.numWhiteRooks; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GenerateRookAttacks(blackKingSq, occupied);
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Rook, Color::White);
        }

        // generate black rooks
        for (uint32_t i = 0; i < material.numBlackRooks; ++i)
        {
            const Bitboard mask = ~occupied;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Rook, Color::Black);
        }

        // generate white bishops, they cannot attack black king
        for (uint32_t i = 0; i < material.numWhiteBishops; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GenerateBishopAttacks(blackKingSq, occupied);
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Bishop, Color::White);
        }

        // generate black bishops
        for (uint32_t i = 0; i < material.numBlackBishops; ++i)
        {
            const Bitboard mask = ~occupied;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Bishop, Color::Black);
        }

        // generate white knights, they cannot attack black king
        for (uint32_t i = 0; i < material.numWhiteKnights; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GetKnightAttacks(blackKingSq);
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Knight, Color::White);
        }

        // generate black knights
        for (uint32_t i = 0; i < material.numBlackKnights; ++i)
        {
            const Bitboard mask = ~occupied;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Knight, Color::Black);
        }

        break;
    }

    ASSERT(outPosition.IsValid());
    ASSERT(!outPosition.IsInCheck(Color::Black));

    return true;
}