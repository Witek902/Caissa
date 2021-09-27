#include "Position.hpp"
#include "MoveList.hpp"

#include <chrono>
#include <random>

#include <immintrin.h>

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

bool Position::IsValid() const
{
    // validate piece counts
    if ((Whites().pawns.Count() + Whites().knights.Count() + Whites().bishops.Count() + Whites().rooks.Count() + Whites().queens.Count() > 15u)) return false;
    if ((Blacks().pawns.Count() + Blacks().knights.Count() + Blacks().bishops.Count() + Blacks().rooks.Count() + Blacks().queens.Count() > 15u)) return false;
    if (Whites().pawns.Count() > 8u || Blacks().pawns.Count() > 8u) return false;
    if (Whites().knights.Count() > 9u || Blacks().knights.Count() > 9u) return false;
    if (Whites().bishops.Count() > 9u || Blacks().bishops.Count() > 9u) return false;
    if (Whites().rooks.Count() > 9u || Blacks().rooks.Count() > 9u) return false;
    if (Whites().queens.Count() > 9u || Blacks().queens.Count() > 9u) return false;
    if (Whites().king.Count() != 1u || Blacks().king.Count() != 1u) return false;

    // validate pawn locations
    {
        bool pawnsValid = true;
        Whites().pawns.Iterate([&](uint32_t index)
        {
            uint8_t pawnRank = Square(index).Rank();
            pawnsValid &= pawnRank >= 1u;   // pawns can't go backward
            pawnsValid &= pawnRank < 7u;    // unpromoted pawn
        });
        Blacks().pawns.Iterate([&](uint32_t index)
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
                side.occupied |= square.GetBitboard();

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
            fprintf(stderr, "Invalid FEN: invalid en passant square\n");
            return false;
        }
    }
    else
    {
        mEnPassantSquare = Square::Invalid();
    }

    // TODO!
    // half-moves
    // move number

    //printf("\nhalf-moves since last pawn move/capture: "); for (++p; *p != ' '; ++p) putchar(*p);
    //printf("\n(full) move number: %s\n", ++p);

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

std::string Position::MoveToString(const Move& move) const
{
    ASSERT(move.GetPiece() != Piece::None);

    Position afterMove(*this);
    if (!afterMove.DoMove(move))
    {
        return "illegal move";
    }

    std::string str;

    if (move.GetPiece() == Piece::Pawn)
    {
        if (move.IsCapture())
        {
            str += move.FromSquare().ToString();
            str += 'x';
        }

        str += move.ToSquare().ToString();
        if (move.ToSquare().Rank() == 7u && move.GetPromoteTo() != Piece::None)
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

                bool ambiguous = false;
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
                            ambiguous = true;
                            break;
                        }
                    }
                }

                if (ambiguous)
                {
                    if ((movedPieceBitboard & Bitboard::RankBitboard(move.FromSquare().Rank())).Count() > 1)
                    {
                        str += 'a' + move.FromSquare().File();
                    }
                    else if ((movedPieceBitboard & Bitboard::FileBitboard(move.FromSquare().File())).Count() > 1)
                    {
                        str += '1' + move.FromSquare().Rank();
                    }
                    else
                    {
                        str += move.FromSquare().ToString();
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

    if (move.IsCapture() && move.IsEnPassant())
    {
        str += " e.p.";
    }

    if (afterMove.IsInCheck(afterMove.GetSideToMove()))
    {
        str += '+';
    }

    // TODO! checkmate

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
    MoveList moves;
    GenerateMoveList(moves);

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
    ASSERT(move.IsValid());

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

    auto startTime = std::chrono::high_resolution_clock::now();

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

    auto endTime = std::chrono::high_resolution_clock::now();

    if (print)
    {
        const float timeInSeconds = (1.0e-6f * (float)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count());

        std::cout << "Total nodes: " << nodes << std::endl;
        std::cout << "Time: " << timeInSeconds << " seconds" << std::endl;
    }

    return nodes;
}

static std::random_device g_randomDevice;
static thread_local std::mt19937_64 g_randomGenerator(g_randomDevice());

bool GenerateRandomPosition(const MaterialKey material, Position& outPosition)
{
    std::mt19937_64& randomGenerator = g_randomGenerator;

    const auto genLegalSquare = [&randomGenerator](const Bitboard mask) -> Square
    {
        std::uniform_int_distribution<uint32_t> distr;

        if (!mask) return Square::Invalid();

        const uint32_t numLegalSquares = mask.Count();
        const uint32_t maskedSquareIndex = distr(randomGenerator) % numLegalSquares;

        const uint64_t squareMask = _pdep_u64(1ull << maskedSquareIndex, mask);
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