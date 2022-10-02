#include "PositionUtils.hpp"

#include "Position.hpp"
#include "MoveList.hpp"
#include "Time.hpp"
#include "Material.hpp"

#include <random>

static_assert(sizeof(PackedPosition) == 28, "Invalid packed position size");

bool PackPosition(const Position& inPos, PackedPosition& outPos)
{
    outPos.occupied = inPos.Whites().Occupied() | inPos.Blacks().Occupied();
    outPos.moveCount = inPos.GetMoveCount();
    outPos.sideToMove = inPos.GetSideToMove() == Color::White ? 0 : 1;
    outPos.halfMoveCount = inPos.GetHalfMoveCount();
    outPos.enPassantFile = inPos.GetEnPassantSquare().IsValid() ? inPos.GetEnPassantSquare().File() : 0xF;

    outPos.castlingRights = 0;
    if (inPos.GetWhitesCastlingRights() & c_shortCastleMask)    outPos.castlingRights |= (1 << 0);
    if (inPos.GetWhitesCastlingRights() & c_longCastleMask)     outPos.castlingRights |= (1 << 1);
    if (inPos.GetBlacksCastlingRights() & c_shortCastleMask)    outPos.castlingRights |= (1 << 2);
    if (inPos.GetBlacksCastlingRights() & c_longCastleMask)     outPos.castlingRights |= (1 << 3);

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

    {
        uint8_t whiteCastlingRights = 0;
        uint8_t blackCastlingRights = 0;
        if (inPos.castlingRights & 0b0001)  whiteCastlingRights |= c_shortCastleMask;
        if (inPos.castlingRights & 0b0010)  whiteCastlingRights |= c_longCastleMask;
        if (inPos.castlingRights & 0b0100)  blackCastlingRights |= c_shortCastleMask;
        if (inPos.castlingRights & 0b1000)  blackCastlingRights |= c_longCastleMask;

        outPos.SetWhitesCastlingRights(whiteCastlingRights);
        outPos.SetBlacksCastlingRights(blackCastlingRights);
    }

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

bool Position::operator != (const Position& rhs) const
{
    return
        Whites() != rhs.Whites() ||
        Blacks() != rhs.Blacks() ||
        mSideToMove != rhs.mSideToMove ||
        mEnPassantSquare != rhs.mEnPassantSquare ||
        mWhitesCastlingRights != rhs.mWhitesCastlingRights ||
        mBlacksCastlingRights != rhs.mBlacksCastlingRights;
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

    if ((((uint64_t)Whites().rooks & mWhitesCastlingRights) != mWhitesCastlingRights) ||
        ((((uint64_t)Blacks().rooks >> (7 * 8)) & mBlacksCastlingRights) != mBlacksCastlingRights))
    {
        return false;
    }

    // TODO 960
    /*
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
    */

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

    size_t loc = 0;

    // board
    {
        uint8_t rank = 7;
        uint8_t file = 0;
        for (; loc < fenString.length() && !isspace(fenString[loc]); ++loc)
        {
            const char ch = fenString[loc];

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

                SetPiece(square, piece, color);

                file++;
            }
        }
    }

    // next to move
    if (++loc < fenString.length())
    {
        const char nextToMove = (char)tolower(fenString[loc]);
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
    else
    {
        fprintf(stderr, "Invalid FEN: missing side to move\n");
        return false;
    }

    // castling rights
    if (Whites().king && Blacks().king)
    {
        const Square whiteKingSq(FirstBitSet(Whites().king));
        const Square blackKingSq(FirstBitSet(Blacks().king));

        mWhitesCastlingRights = 0;
        mBlacksCastlingRights = 0;
        for (loc += 2; loc < fenString.length() && !isspace(fenString[loc]); ++loc)
        {
            constexpr uint8_t longCastleMask[] =    { 0b00000000, 0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111 };
            constexpr uint8_t shortCastleMask[] =   { 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000, 0b00000000 };

            const char c = fenString[loc];
            if (c >= 'A' && c <= 'H')
            {
                mWhitesCastlingRights = mWhitesCastlingRights | (1 << (c - 'A'));
            }
            else if (c >= 'a' && c <= 'h')
            {
                mBlacksCastlingRights = mBlacksCastlingRights | (1 << (c - 'a'));
            }
            else if (c == 'K')
            {
                uint8_t mask = shortCastleMask[whiteKingSq.File()] & (uint8_t)(uint64_t)Whites().rooks;
                if (PopCount(mask) > 1) mask = 0; // ambiguous short castle
                mWhitesCastlingRights = mWhitesCastlingRights | mask;
            }
            else if (c == 'Q')
            {
                uint8_t mask = longCastleMask[whiteKingSq.File()] & (uint8_t)(uint64_t)Whites().rooks;
                if (PopCount(mask) > 1) mask = 0; // ambiguous long castle
                mWhitesCastlingRights = mWhitesCastlingRights | mask;
            }
            else if (c == 'k')
            {
                uint8_t mask = shortCastleMask[blackKingSq.File()] & (uint8_t)(uint64_t)((uint64_t)Blacks().rooks >> (7 * 8));
                if (PopCount(mask) > 1) mask = 0; // ambiguous short castle
                mBlacksCastlingRights = mBlacksCastlingRights | mask;
            }
            else if (c == 'q')
            {
                uint8_t mask = longCastleMask[blackKingSq.File()] & (uint8_t)(uint64_t)((uint64_t)Blacks().rooks >> (7 * 8));
                if (PopCount(mask) > 1) mask = 0; // ambiguous long castle
                mBlacksCastlingRights = mBlacksCastlingRights | mask;
            }
            else if (c == '-')
            {
                continue;
            }
            else
            {
                fprintf(stderr, "Invalid FEN: invalid castling rights\n");
                return false;
            }
        }

        // clear up castling rights if rook is not present on the square
        mWhitesCastlingRights &= (uint8_t)(uint64_t)Whites().rooks;
        mBlacksCastlingRights &= (uint8_t)(uint64_t)((uint64_t)Blacks().rooks >> (7 * 8));

        // clear up castling rights if king is in wrong place
        if (whiteKingSq.Rank() > 0 || whiteKingSq.File() == 0 || whiteKingSq.File() == 7) mWhitesCastlingRights = 0;
        if (blackKingSq.Rank() < 7 || blackKingSq.File() == 0 || blackKingSq.File() == 7) mBlacksCastlingRights = 0;
    }

    std::string enPassantSquare;
    for (++loc; loc < fenString.length() && !isspace(fenString[loc]); ++loc)
    {
        enPassantSquare += fenString[loc];
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
        for (++loc; loc < fenString.length() && !isspace(fenString[loc]); ++loc)
        {
            halfMovesStr += fenString[loc];
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
        for (++loc; loc < fenString.length() && !isspace(fenString[loc]); ++loc)
        {
            moveNumberStr += fenString[loc];
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
        str += ' ';

        const Square whiteKingSq(FirstBitSet(Whites().king));
        const Square blackKingSq(FirstBitSet(Blacks().king));

        if (!s_enableChess960)
        {
            if (GetShortCastleRookSquare(Whites().GetKingSquare(), mWhitesCastlingRights).IsValid())  str += 'K';
            if (GetLongCastleRookSquare(Whites().GetKingSquare(), mWhitesCastlingRights).IsValid())   str += 'Q';
            if (GetShortCastleRookSquare(Blacks().GetKingSquare(), mBlacksCastlingRights).IsValid())  str += 'k';
            if (GetLongCastleRookSquare(Blacks().GetKingSquare(), mBlacksCastlingRights).IsValid())   str += 'q';
        }
        else
        {
            for (uint8_t i = 0; i < 8; ++i)
            {
                if (mWhitesCastlingRights & (1 << i))  str += ('A' + i);
            }
            for (uint8_t i = 0; i < 8; ++i)
            {
                if (mBlacksCastlingRights & (1 << i))  str += ('a' + i);
            }
        }

        if (mWhitesCastlingRights == 0 && mBlacksCastlingRights == 0) str += '-';
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
            if (move.IsShortCastle())
            {
                str = "O-O";
            }
            else if (move.IsLongCastle())
            {
                str = "O-O-O";
            }
            else
            {
                str = PieceToChar(move.GetPiece());

                {
                    bool ambiguousFile = false;
                    bool ambiguousRank = false;
                    bool ambiguousPiece = false;
                    {
                        MoveList moves;
                        GenerateMoveList(moves);

                        for (uint32_t i = 0; i < moves.numMoves; ++i)
                        {
                            const Move& refMove = moves[i].move;
                            
                            if (refMove.GetPiece() == move.GetPiece() &&
                                refMove.ToSquare() == move.ToSquare() &&
                                refMove.FromSquare() != move.FromSquare() &&
                                IsMoveLegal(refMove))
                            {
                                if (refMove.FromSquare().File() == move.FromSquare().File())  ambiguousFile = true;
                                if (refMove.FromSquare().Rank() == move.FromSquare().Rank())  ambiguousRank = true;
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

        ASSERT(move == MoveFromString(str, MoveNotation::SAN));
    }
    else
    {
        DEBUG_BREAK();
    }

    return str;
}

Move Position::MoveFromPacked(const PackedMove& packedMove) const
{
    if (!packedMove.FromSquare().IsValid())
    {
        return Move();
    }

    const Piece movedPiece = GetCurrentSide().GetPieceAtSquare(packedMove.FromSquare());

    const Bitboard occupiedByCurrent = GetCurrentSide().Occupied();
    const Bitboard occupiedByOpponent = GetOpponentSide().Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;
    const bool isCapture = packedMove.ToSquare().GetBitboard() & occupiedByOpponent;

    switch (movedPiece)
    {
        case Piece::Pawn:
        {
            // TODO generate pawn move directly
            MoveList moves;
            GeneratePawnMoveList(moves);
            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                if (moves[i].move == packedMove)
                {
                    return moves[i].move;
                }
            }
            break;
        }

        case Piece::Knight:
        {
            Bitboard attackBitboard = (~occupiedByCurrent) & Bitboard::GetKnightAttacks(packedMove.FromSquare());
            if (packedMove.ToSquare().GetBitboard() & attackBitboard)
            {
                return Move::Make(packedMove.FromSquare(), packedMove.ToSquare(), movedPiece, Piece::None, isCapture);
            }
            break;
        }

        case Piece::Bishop:
        {
            Bitboard attackBitboard = (~occupiedByCurrent) & Bitboard::GenerateBishopAttacks(packedMove.FromSquare(), occupiedSquares);
            if (packedMove.ToSquare().GetBitboard() & attackBitboard)
            {
                return Move::Make(packedMove.FromSquare(), packedMove.ToSquare(), movedPiece, Piece::None, isCapture);
            }
            break;
        }

        case Piece::Rook:
        {
            Bitboard attackBitboard = (~occupiedByCurrent) & Bitboard::GenerateRookAttacks(packedMove.FromSquare(), occupiedSquares);
            if (packedMove.ToSquare().GetBitboard() & attackBitboard)
            {
                return Move::Make(packedMove.FromSquare(), packedMove.ToSquare(), movedPiece, Piece::None, isCapture);
            }
            break;
        }

        case Piece::Queen:
        {
            Bitboard attackBitboard = (~occupiedByCurrent) &
                (Bitboard::GenerateRookAttacks(packedMove.FromSquare(), occupiedSquares) |
                 Bitboard::GenerateBishopAttacks(packedMove.FromSquare(), occupiedSquares));
            if (packedMove.ToSquare().GetBitboard() & attackBitboard)
            {
                return Move::Make(packedMove.FromSquare(), packedMove.ToSquare(), movedPiece, Piece::None, isCapture);
            }
            break;
        }

        case Piece::King:
        {
            // TODO generate king move directly
            MoveList moves;
            GenerateKingMoveList(moves);

            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                if (moves[i].move == packedMove)
                {
                    return moves[i].move;
                }
            }
            break;
        }
    }

    return Move();
}

Move Position::MoveFromString(const std::string& str, MoveNotation notation) const
{
    if (notation == MoveNotation::LAN)
    {
        if (str.length() < 4)
        {
            fprintf(stderr, "MoveFromString: Move string too short\n");
            return {};
        }

        Square fromSquare = Square::FromString(str.substr(0, 2));
        Square toSquare = Square::FromString(str.substr(2, 2));

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
        bool isLongCastle = false;
        bool isShortCastle = false;

        if (movedPiece == Piece::King)
        {
            const uint8_t currentSideCastlingRights = mSideToMove == Color::White ? mWhitesCastlingRights : mBlacksCastlingRights;
            const Square longCastleRookSquare = GetLongCastleRookSquare(fromSquare, currentSideCastlingRights);
            const Square shortCastleRookSquare = GetShortCastleRookSquare(fromSquare, currentSideCastlingRights);

            if ((toSquare == longCastleRookSquare) ||
                (fromSquare == Square_e1 && toSquare == Square_c1 && longCastleRookSquare == Square_a1) ||
                (fromSquare == Square_e8 && toSquare == Square_c8 && longCastleRookSquare == Square_a8))
            {
                isLongCastle = true;
                isCapture = false;
                toSquare = longCastleRookSquare;
            }
            else if ((toSquare == shortCastleRookSquare) ||
                (fromSquare == Square_e1 && toSquare == Square_g1 && shortCastleRookSquare == Square_h1) ||
                (fromSquare == Square_e8 && toSquare == Square_g8 && shortCastleRookSquare == Square_h8))
            {
                isShortCastle = true;
                isCapture = false;
                toSquare = shortCastleRookSquare;
            }

            MoveList moves;
            GenerateKingMoveList(moves);

            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                if (moves[i].move.FromSquare() == fromSquare && moves[i].move.ToSquare() == toSquare)
                {
                    return moves[i].move;
                }
            }

            fprintf(stderr, "MoveFromString: Failed to parse king move\n");
            return {};
        }

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

        return Move::Make(fromSquare, toSquare, movedPiece, promoteTo, isCapture, isEnPassant, isLongCastle, isShortCastle);
    }
    else if (notation == MoveNotation::SAN)
    {
        if (str.length() < 2)
        {
            fprintf(stderr, "MoveFromString: Move string too short\n");
            return {};
        }

        if (str == "O-O" || str == "0-0")
        {
            if (mSideToMove == Color::White)
            {
                const Square sourceSquare = Whites().GetKingSquare();
                const Square targetSquare = GetShortCastleRookSquare(sourceSquare, mWhitesCastlingRights);
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, false, true);
            }
            else
            {
                const Square sourceSquare = Blacks().GetKingSquare();
                const Square targetSquare = GetShortCastleRookSquare(sourceSquare, mBlacksCastlingRights);
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, false, true);
            }
        }
        else if (str == "O-O-O" || str == "0-0-0")
        {
            if (mSideToMove == Color::White)
            {
                const Square sourceSquare = Whites().GetKingSquare();
                const Square targetSquare = GetLongCastleRookSquare(sourceSquare, mWhitesCastlingRights);
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, true, false);
            }
            else
            {
                const Square sourceSquare = Blacks().GetKingSquare();
                const Square targetSquare = GetLongCastleRookSquare(sourceSquare, mBlacksCastlingRights);
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, true, false);
            }
        }

        uint32_t offset = 0;

        bool isCapture = false;
        Piece movedPiece = Piece::Pawn;
        Piece promoteTo = Piece::None;

        int32_t fromFile = -1;
        int32_t fromRank = -1;
        int32_t toFile = -1;
        int32_t toRank = -1;

        switch (str[0])
        {
        case 'P':   movedPiece = Piece::Pawn; offset++; break;
        case 'N':   movedPiece = Piece::Knight; offset++; break;
        case 'B':   movedPiece = Piece::Bishop; offset++; break;
        case 'R':   movedPiece = Piece::Rook; offset++; break;
        case 'Q':   movedPiece = Piece::Queen; offset++; break;
        case 'K':   movedPiece = Piece::King; offset++; break;
        }

        const auto isFile = [](const char c) { return c >= 'a' && c <= 'h'; };

        if (str.length() >= offset + 5 && isFile(str[offset]) && isdigit(str[offset + 1]) && str[offset + 2] == 'x' && isFile(str[offset + 3]) && isdigit(str[offset + 4]))
        {
            fromFile = str[offset + 0] - 'a';
            fromRank = str[offset + 1] - '1';
            toFile = str[offset + 3] - 'a';
            toRank = str[offset + 4] - '1';
            isCapture = true;
            offset += 5;
        }
        else if (str.length() >= offset + 4 && isFile(str[offset]) && isdigit(str[offset + 1]) && isFile(str[offset + 2]) && isdigit(str[offset + 3]))
        {
            fromFile = str[offset + 0] - 'a';
            fromRank = str[offset + 1] - '1';
            toFile = str[offset + 2] - 'a';
            toRank = str[offset + 3] - '1';
            offset += 4;
        }
        else if (str.length() >= offset + 4 && isFile(str[offset]) && str[offset + 1] == 'x' && isFile(str[offset + 2]) && isdigit(str[offset + 3]))
        {
            fromFile = str[offset + 0] - 'a';
            toFile = str[offset + 2] - 'a';
            toRank = str[offset + 3] - '1';
            isCapture = true;
            offset += 4;
        }
        else if (str.length() >= offset + 3 && isFile(str[offset]) && isFile(str[offset + 1]) && isdigit(str[offset + 2]))
        {
            fromFile = str[offset + 0] - 'a';
            toFile = str[offset + 1] - 'a';
            toRank = str[offset + 2] - '1';
            offset += 3;
        }
        else if (str.length() >= offset + 4 && isdigit(str[offset]) && str[offset + 1] == 'x' && isFile(str[offset + 2]) && isdigit(str[offset + 3]))
        {
            fromRank = str[offset + 0] - '1';
            toFile = str[offset + 2] - 'a';
            toRank = str[offset + 3] - '1';
            isCapture = true;
            offset += 4;
        }
        else if (str.length() >= offset + 3 && isdigit(str[offset]) && isFile(str[offset + 1]) && isdigit(str[offset + 2]))
        {
            fromRank = str[offset + 0] - '1';
            toFile = str[offset + 1] - 'a';
            toRank = str[offset + 2] - '1';
            offset += 3;
        }
        else if (str.length() >= offset + 3 && str[offset] == 'x' && isFile(str[offset + 1]) && isdigit(str[offset + 2]))
        {
            toFile = str[offset + 1] - 'a';
            toRank = str[offset + 2] - '1';
            isCapture = true;
            offset += 3;
        }
        else if (str.length() >= offset + 2 && isFile(str[offset]) && isdigit(str[offset + 1]))
        {
            toFile = str[offset + 0] - 'a';
            toRank = str[offset + 1] - '1';
            offset += 2;
        }
        else
        {
            fprintf(stderr, "MoveFromString: Failed to parse move\n");
            return {};
        }

        if (toFile < 0 || toFile >= 8)
        {
            fprintf(stderr, "MoveFromString: Invalid target square\n");
            return {};
        }
        if (toRank < 0 || toRank >= 8)
        {
            fprintf(stderr, "MoveFromString: Invalid target square\n");
            return {};
        }

        if (movedPiece == Piece::Pawn && ((mSideToMove == Color::White && toRank == 7) || (mSideToMove == Color::Black && toRank == 0)))
        {
            if (str.length() >= offset + 2 && str[offset] == '=' && CharToPiece(str[offset + 1], promoteTo))
            {
                offset += 2;
            }
            else
            {
                fprintf(stderr, "MoveFromString: Missing promotion\n");
                return {};
            }
        }

        const Square toSquare((uint8_t)toFile, (uint8_t)toRank);

        MoveList moves;
        GenerateMoveList(moves);

        for (uint32_t i = 0; i < moves.numMoves; ++i)
        {
            const Move& move = moves[i].move;
            if (movedPiece == move.GetPiece() &&
                toSquare == move.ToSquare() &&
                move.GetPromoteTo() == promoteTo &&
                (fromFile == -1 || fromFile == move.FromSquare().File()) &&
                (fromRank == -1 || fromRank == move.FromSquare().Rank()) &&
                IsMoveLegal(move))
            {
                return move;
            }
        }
    }
    else
    {
        DEBUG_BREAK();
    }

    return {};
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
    const Piece capturedOwnPiece = currentSide.GetPieceAtSquare(move.ToSquare());

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
    if (capturedOwnPiece != Piece::None && !(move.IsCastling() && capturedOwnPiece == Piece::Rook))
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
    }

    return MoveFromPacked(PackedMove(move)).IsValid();
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
    TimePoint startTime;

    if (print)
    {
        std::cout << "Running Perft... depth=" << depth << std::endl;
        startTime = TimePoint::GetCurrent();
    }

    MoveList moveList;
    GenerateMoveList(moveList);

    uint64_t nodes = 0;
    for (uint32_t i = 0; i < moveList.Size(); i++)
    {
        const Move& move = moveList.GetMove(i);

        ASSERT(move == MoveFromPacked(PackedMove(move)));

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
        const float t = (endTime - startTime).ToSeconds();

        std::cout << "Total nodes:      " << nodes << std::endl;
        std::cout << "Time:             " << t << " seconds" << std::endl;
        std::cout << "Nodes per second: " << 1.0e-6f * (nodes / t) << "M" << std::endl;
    }

    return nodes;
}

void GenerateRandomPosition(std::mt19937& randomGenerator, const MaterialKey& material, Position& outPosition)
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
}

void GenerateTranscendentalChessPosition(std::mt19937& randomGenerator, Position& outPosition)
{
    std::vector<Piece> pieces = { Piece::Rook, Piece::Knight, Piece::Bishop, Piece::Queen, Piece::King, Piece::Bishop, Piece::Knight, Piece::Rook };

    const auto randomizePieces = [&]()
    {
        for (;;)
        {
            std::shuffle(pieces.begin(), pieces.end(), randomGenerator);

            size_t bishopIndex = 0;
            size_t bishopPos[2];

            for (size_t i = 0; i < 8; ++i)
            {
                if (pieces[i] == Piece::Bishop)
                {
                    bishopPos[bishopIndex++] = i;
                }
            }
            ASSERT(bishopIndex == 2);

            if ((bishopPos[1] - bishopPos[0]) % 2 == 1)
            {
                return;
            }
        }
    };

    outPosition = Position();

    randomizePieces();

    for (uint8_t i = 0; i < 8; ++i)
    {
        outPosition.SetPiece(Square(i, 0), pieces[i], Color::White);
        outPosition.SetPiece(Square(i, 1), Piece::Pawn, Color::White);
    }

    randomizePieces();

    for (uint8_t i = 0; i < 8; ++i)
    {
        outPosition.SetPiece(Square(i, 7), pieces[i], Color::Black);
        outPosition.SetPiece(Square(i, 6), Piece::Pawn, Color::Black);
    }
}
