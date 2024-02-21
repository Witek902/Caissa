#include "PositionUtils.hpp"

#include "Position.hpp"
#include "MoveList.hpp"
#include "Time.hpp"
#include "Material.hpp"
#include "MoveGen.hpp"

#include <random>
#include <iostream>

static_assert(sizeof(PackedPosition) == 28, "Invalid packed position size");

bool PackPosition(const Position& inPos, PackedPosition& outPos)
{
    outPos.occupied = inPos.Whites().Occupied() | inPos.Blacks().Occupied();
    outPos.moveCount = inPos.GetMoveCount();
    outPos.sideToMove = inPos.GetSideToMove() == White ? 0 : 1;
    outPos.halfMoveCount = inPos.GetHalfMoveCount();
    outPos.enPassantFile = inPos.GetEnPassantSquare().IsValid() ? inPos.GetEnPassantSquare().File() : 0xF;

    ASSERT(outPos.occupied.Count() <= 32);

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

bool UnpackPosition(const PackedPosition& inPos, Position& outPos, bool computeHash)
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
            outPos.SetPiece(Square(index), piece, White);
        }
        else if (value >= 8 && value <= 8 + (uint8_t)Piece::King)
        {
            const Piece piece = (Piece)(value - 8 + (uint8_t)Piece::Pawn);
            outPos.SetPiece(Square(index), piece, Black);
        }
        else
        {
            success = false;
        }

        offset++;
    });

    outPos.SetSideToMove(inPos.sideToMove);
    outPos.SetMoveCount(inPos.moveCount);
    outPos.SetHalfMoveCount(inPos.halfMoveCount);

    {
        uint8_t whiteCastlingRights = 0;
        uint8_t blackCastlingRights = 0;
        if (inPos.castlingRights & 0b0001)  whiteCastlingRights |= c_shortCastleMask;
        if (inPos.castlingRights & 0b0010)  whiteCastlingRights |= c_longCastleMask;
        if (inPos.castlingRights & 0b0100)  blackCastlingRights |= c_shortCastleMask;
        if (inPos.castlingRights & 0b1000)  blackCastlingRights |= c_longCastleMask;

        outPos.SetCastlingRights(White, whiteCastlingRights);
        outPos.SetCastlingRights(Black, blackCastlingRights);
    }

    if (inPos.enPassantFile < 8)
    {
        outPos.SetEnPassantSquare(Square(inPos.enPassantFile, inPos.sideToMove == 0 ? 5 : 2));
    }

    if (computeHash)
    {
        outPos.ComputeHash();
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
        GetWhitesCastlingRights() == rhs.GetWhitesCastlingRights() &&
        GetBlacksCastlingRights() == rhs.GetBlacksCastlingRights();

    if (result && mHash != 0 && rhs.mHash != 0)
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
        GetWhitesCastlingRights() != rhs.GetWhitesCastlingRights() ||
        GetBlacksCastlingRights() != rhs.GetBlacksCastlingRights();
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

    if ((((uint64_t)Whites().rooks & GetWhitesCastlingRights()) != GetWhitesCastlingRights()) ||
        ((((uint64_t)Blacks().rooks >> (7 * 8)) & GetBlacksCastlingRights()) != GetBlacksCastlingRights()))
    {
        return false;
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

                    if (file > 8)
                    {
                        fprintf(stderr, "Invalid FEN: failed to parse board state\n");
                        return false;
                    }
                }
                else
                {
                    fprintf(stderr, "Invalid FEN: Too many pieces in rank %u\n", (uint32_t)(rank + 1));
                    return false;
                }
            }
            else if (ch == '/')
            {
                if (file != 8)
                {
                    fprintf(stderr, "Invalid FEN: Not enough pieces in rank %u\n", (uint32_t)(rank + 1));
                    return false;
                }

                file = 0;
                rank--;
            }
            else
            {
                const Square square(file, rank);
                const Color color = ch <= 90 ? White : Black;

                Piece piece;
                if (!CharToPiece(ch, piece))
                {
                    fprintf(stderr, "Invalid FEN: failed to parse board state\n");
                    return false;
                }

                SetPiece(square, piece, color);

                file++;

                if (file > 8)
                {
                    fprintf(stderr, "Invalid FEN: Too many pieces in rank %u\n", (uint32_t)(rank + 1));
                    return false;
                }
            }
        }
    }

    // next to move
    if (++loc < fenString.length())
    {
        const char nextToMove = (char)tolower(fenString[loc]);
        if (nextToMove == 'w')
        {
            mSideToMove = White;
        }
        else if (nextToMove == 'b')
        {
            mSideToMove = Black;
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

        mCastlingRights[0] = 0;
        mCastlingRights[1] = 0;
        for (loc += 2; loc < fenString.length() && !isspace(fenString[loc]); ++loc)
        {
            constexpr uint8_t longCastleMask[] =    { 0b00000000, 0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111 };
            constexpr uint8_t shortCastleMask[] =   { 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000, 0b00000000 };

            const char c = fenString[loc];
            if (c >= 'A' && c <= 'H')
            {
                mCastlingRights[0] = mCastlingRights[0] | (1 << (c - 'A'));
            }
            else if (c >= 'a' && c <= 'h')
            {
                mCastlingRights[1] = mCastlingRights[1] | (1 << (c - 'a'));
            }
            else if (c == 'K')
            {
                uint8_t mask = shortCastleMask[whiteKingSq.File()] & (uint8_t)(uint64_t)Whites().rooks;
                if (PopCount(mask) > 1) mask = 0; // ambiguous short castle
                mCastlingRights[0] = mCastlingRights[0] | mask;
            }
            else if (c == 'Q')
            {
                uint8_t mask = longCastleMask[whiteKingSq.File()] & (uint8_t)(uint64_t)Whites().rooks;
                if (PopCount(mask) > 1) mask = 0; // ambiguous long castle
                mCastlingRights[0] = mCastlingRights[0] | mask;
            }
            else if (c == 'k')
            {
                uint8_t mask = shortCastleMask[blackKingSq.File()] & (uint8_t)(uint64_t)((uint64_t)Blacks().rooks >> (7 * 8));
                if (PopCount(mask) > 1) mask = 0; // ambiguous short castle
                mCastlingRights[1] = mCastlingRights[1] | mask;
            }
            else if (c == 'q')
            {
                uint8_t mask = longCastleMask[blackKingSq.File()] & (uint8_t)(uint64_t)((uint64_t)Blacks().rooks >> (7 * 8));
                if (PopCount(mask) > 1) mask = 0; // ambiguous long castle
                mCastlingRights[1] = mCastlingRights[1] | mask;
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
        mCastlingRights[0] &= (uint8_t)(uint64_t)Whites().rooks;
        mCastlingRights[1] &= (uint8_t)(uint64_t)((uint64_t)Blacks().rooks >> (7 * 8));

        // clear up castling rights if king is in wrong place
        if (whiteKingSq.Rank() > 0 || whiteKingSq.File() == 0 || whiteKingSq.File() == 7) mCastlingRights[0] = 0;
        if (blackKingSq.Rank() < 7 || blackKingSq.File() == 0 || blackKingSq.File() == 7) mCastlingRights[1] = 0;
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

        if (mSideToMove == White)
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

    if (IsInCheck(mSideToMove ^ 1))
    {
        fprintf(stderr, "Invalid FEN: opponent cannot be in check\n");
        return false;
    }

    return true;
}

std::string Position::ToFEN(bool skipMoveCounts) const
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
        str += mSideToMove == White ? 'w' : 'b';
    }

    // castling rights
    {
        str += ' ';

        const Square whiteKingSq(FirstBitSet(Whites().king));
        const Square blackKingSq(FirstBitSet(Blacks().king));

        if (!s_enableChess960)
        {
            if (GetShortCastleRookSquare(Whites().GetKingSquare(), GetWhitesCastlingRights()).IsValid())  str += 'K';
            if (GetLongCastleRookSquare(Whites().GetKingSquare(), GetWhitesCastlingRights()).IsValid())   str += 'Q';
            if (GetShortCastleRookSquare(Blacks().GetKingSquare(), GetBlacksCastlingRights()).IsValid())  str += 'k';
            if (GetLongCastleRookSquare(Blacks().GetKingSquare(), GetBlacksCastlingRights()).IsValid())   str += 'q';
        }
        else
        {
            for (uint8_t i = 0; i < 8; ++i)
            {
                if (GetWhitesCastlingRights() & (1 << i))  str += ('A' + i);
            }
            for (uint8_t i = 0; i < 8; ++i)
            {
                if (GetBlacksCastlingRights() & (1 << i))  str += ('a' + i);
            }
        }

        if (GetWhitesCastlingRights() == 0 && GetBlacksCastlingRights() == 0) str += '-';
    }

    // en passant square
    {
        str += ' ';
        str += mEnPassantSquare.IsValid() ? mEnPassantSquare.ToString() : "-";
    }

    if (!skipMoveCounts)
    {
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
                        Threats threats;
                        ComputeThreats(threats);

                        MoveList moves;
                        GenerateMoveList(*this, threats.allThreats, moves);

                        for (uint32_t i = 0; i < moves.Size(); ++i)
                        {
                            const Move& refMove = moves.GetMove(i);
                            
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
            if (GetSideToMove() == White)
            {
                GeneratePawnMoveList<MoveGenerationMode::Captures, White>(*this, moves);
                GeneratePawnMoveList<MoveGenerationMode::Quiets, White>(*this, moves);
            }
            else
            {
                GeneratePawnMoveList<MoveGenerationMode::Captures, Black>(*this, moves);
                GeneratePawnMoveList<MoveGenerationMode::Quiets, Black>(*this, moves);
            }
            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                const Move move = moves.GetMove(i);
                if (move == packedMove)
                {
                    return move;
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
            Bitboard attackBitboard = Bitboard::GetKingAttacks(packedMove.FromSquare());
            attackBitboard &= ~occupiedByCurrent; // can't capture own piece
            attackBitboard &= ~Bitboard::GetKingAttacks(GetOpponentSide().GetKingSquare()); // can't move to square controlled by opponent's king

            if (packedMove.ToSquare().GetBitboard() & attackBitboard)
            {
                return Move::Make(packedMove.FromSquare(), packedMove.ToSquare(), movedPiece, Piece::None, isCapture);
            }

            // check castlings
            if (!isCapture)
            {
                TMoveList<2> moves;
                if (GetSideToMove() == White)
                    GenerateCastlingMoveList<White>(*this, moves);
                else
                    GenerateCastlingMoveList<Black>(*this, moves);

                for (uint32_t i = 0; i < moves.Size(); ++i)
                {
                    const Move move = moves.GetMove(i);
                    if (move == packedMove)
                    {
                        return move;
                    }
                }
            }

            break;
        }
    }

    return Move();
}

Move Position::MoveFromString(const std::string& moveString, MoveNotation notation) const
{
    if (notation == MoveNotation::LAN)
    {
        if (moveString.length() < 4)
        {
            fprintf(stderr, "MoveFromString: Move string too short\n");
            return {};
        }

        Square fromSquare = Square::FromString(moveString.substr(0, 2));
        Square toSquare = Square::FromString(moveString.substr(2, 2));

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
            const uint8_t currentSideCastlingRights = mCastlingRights[(uint32_t)mSideToMove];
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

            Threats threats;
            ComputeThreats(threats);

            MoveList moves;
            GenerateKingMoveList(*this, threats.allThreats, moves);

            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                const Move move = moves.GetMove(i);
                if (move.FromSquare() == fromSquare && move.ToSquare() == toSquare)
                {
                    return move;
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
        if (moveString.length() > 4)
        {
            if (!CharToPiece(moveString[4], promoteTo))
            {
                fprintf(stderr, "MoveFromString: Failed to parse promotion\n");
                return {};
            }
        }

        return Move::Make(fromSquare, toSquare, movedPiece, promoteTo, isCapture, isEnPassant, isLongCastle, isShortCastle);
    }
    else if (notation == MoveNotation::SAN)
    {
        // trim suffixes such as "!?", "+"
        const std::string str = moveString.substr(0, moveString.find_last_not_of("?!#+") + 1);

        if (str.length() < 2)
        {
            fprintf(stderr, "MoveFromString: Move string too short\n");
            return {};
        }

        if (str == "O-O" || str == "0-0")
        {
            if (mSideToMove == White)
            {
                const Square sourceSquare = Whites().GetKingSquare();
                const Square targetSquare = GetShortCastleRookSquare(sourceSquare, GetWhitesCastlingRights());
                ASSERT(sourceSquare.IsValid());
                ASSERT(targetSquare.IsValid());
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, false, true);
            }
            else
            {
                const Square sourceSquare = Blacks().GetKingSquare();
                const Square targetSquare = GetShortCastleRookSquare(sourceSquare, GetBlacksCastlingRights());
                ASSERT(sourceSquare.IsValid());
                ASSERT(targetSquare.IsValid());
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, false, true);
            }
        }
        else if (str == "O-O-O" || str == "0-0-0")
        {
            if (mSideToMove == White)
            {
                const Square sourceSquare = Whites().GetKingSquare();
                const Square targetSquare = GetLongCastleRookSquare(sourceSquare, GetWhitesCastlingRights());
                ASSERT(sourceSquare.IsValid());
                ASSERT(targetSquare.IsValid());
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, true, false);
            }
            else
            {
                const Square sourceSquare = Blacks().GetKingSquare();
                const Square targetSquare = GetLongCastleRookSquare(sourceSquare, GetBlacksCastlingRights());
                ASSERT(sourceSquare.IsValid());
                ASSERT(targetSquare.IsValid());
                return Move::Make(sourceSquare, targetSquare, Piece::King, Piece::None, false, false, true, false);
            }
        }

        uint32_t offset = 0;

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

        if (movedPiece == Piece::Pawn && ((mSideToMove == White && toRank == 7) || (mSideToMove == Black && toRank == 0)))
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

        Threats threats;
        ComputeThreats(threats);

        MoveList moves;
        GenerateMoveList(*this, threats.allThreats, moves);

        for (uint32_t i = 0; i < moves.Size(); ++i)
        {
            const Move move = moves.GetMove(i);

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
        if ((mSideToMove == White && move.ToSquare().Rank() == 7) ||
            (mSideToMove == Black && move.ToSquare().Rank() == 0))
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

bool Position::IsCapture(const PackedMove& move) const
{
    return
        (GetCurrentSide().Occupied() & move.FromSquare().GetBitboard()) &&
        (GetOpponentSide().Occupied() & move.ToSquare().GetBitboard());
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
    GenerateMoveList(*this, Bitboard::GetKingAttacks(GetOpponentSide().GetKingSquare()), moveList);

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

void GenerateRandomPosition(std::mt19937& randomGenerator, const RandomPosDesc& desc, Position& outPosition)
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
    
    constexpr const Bitboard pawnMask = ~Bitboard::RankBitboard<0>() & ~Bitboard::RankBitboard<7>();

    for (;;)
    {
        outPosition = Position();

        Bitboard occupied = 0;

        // generate white king square
        Square whiteKingSq;
        {
            const Bitboard mask = desc.allowedWhiteKing;
            ASSERT(mask);
            whiteKingSq = genLegalSquare(mask);
            ASSERT(whiteKingSq.IsValid());
            occupied |= whiteKingSq.GetBitboard();
            outPosition.SetPiece(whiteKingSq, Piece::King, White);
        }

        // generate black king square
        Square blackKingSq;
        {
            const Bitboard mask = ~whiteKingSq.GetBitboard() & ~Bitboard::GetKingAttacks(whiteKingSq) & desc.allowedBlackKing;
            ASSERT(mask);
            blackKingSq = genLegalSquare(mask);
            ASSERT(blackKingSq.IsValid());
            occupied |= blackKingSq.GetBitboard();
            outPosition.SetPiece(blackKingSq, Piece::King, Black);
        }

        // generate white pawns on ranks 1-7, they cannot attack black king
        for (uint32_t i = 0; i < desc.materialKey.numWhitePawns; ++i)
        {
            const Bitboard mask = ~occupied & pawnMask & ~Bitboard::GetPawnAttacks(blackKingSq, Black) & desc.allowedWhitePawns;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Pawn, White);
        }

        // TODO generate en-passant square if possible

        // generate black pawns on ranks 1-7
        for (uint32_t i = 0; i < desc.materialKey.numBlackPawns; ++i)
        {
            const Bitboard mask = ~occupied & pawnMask & desc.allowedBlackPawns;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Pawn, Black);
        }

        // generate white queens, they cannot attack black king
        for (uint32_t i = 0; i < desc.materialKey.numWhiteQueens; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GenerateQueenAttacks(blackKingSq, occupied) & desc.allowedWhiteQueens;
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Queen, White);
        }

        // generate black queens
        for (uint32_t i = 0; i < desc.materialKey.numBlackQueens; ++i)
        {
            const Bitboard mask = ~occupied & desc.allowedBlackQueens;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Queen, Black);
        }

        // generate white rooks, they cannot attack black king
        for (uint32_t i = 0; i < desc.materialKey.numWhiteRooks; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GenerateRookAttacks(blackKingSq, occupied) & desc.allowedWhiteRooks;
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Rook, White);
        }

        // generate black rooks
        for (uint32_t i = 0; i < desc.materialKey.numBlackRooks; ++i)
        {
            const Bitboard mask = ~occupied & desc.allowedBlackRooks;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Rook, Black);
        }

        // generate white bishops, they cannot attack black king
        for (uint32_t i = 0; i < desc.materialKey.numWhiteBishops; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GenerateBishopAttacks(blackKingSq, occupied) & desc.allowedWhiteBishops;
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Bishop, White);
        }

        // generate black bishops
        for (uint32_t i = 0; i < desc.materialKey.numBlackBishops; ++i)
        {
            const Bitboard mask = ~occupied & desc.allowedBlackBishops;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Bishop, Black);
        }

        // generate white knights, they cannot attack black king
        for (uint32_t i = 0; i < desc.materialKey.numWhiteKnights; ++i)
        {
            const Bitboard mask = ~occupied & ~Bitboard::GetKnightAttacks(blackKingSq) & desc.allowedWhiteKnights;
            if (!mask) continue;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Knight, White);
        }

        // generate black knights
        for (uint32_t i = 0; i < desc.materialKey.numBlackKnights; ++i)
        {
            const Bitboard mask = ~occupied & desc.allowedBlackKnights;
            const Square sq = genLegalSquare(mask);
            ASSERT(sq.IsValid());
            occupied |= sq.GetBitboard();
            outPosition.SetPiece(sq, Piece::Knight, Black);
        }

        break;
    }

    ASSERT(outPosition.IsValid());
    ASSERT(!outPosition.IsInCheck(Black));
}
