#pragma once

#include "Position.hpp"
#include "MoveList.hpp"

enum class MoveGenerationMode
{
    Captures,   // captures and queen promotion
    Quiets,     // quiet moves and underpromotions
};

template<MoveGenerationMode mode, bool isCapture>
INLINE void GeneratePromotionsMoveList(const Square from, const Square to, MoveList::BatchWriter& writer)
{
    if constexpr (mode == MoveGenerationMode::Captures)
    {
        writer.Push(Move::Make(from, to, Piece::Pawn, Piece::Queen, isCapture));
    }
    else // generate underpromotions
    {
        writer.Push(Move::Make(from, to, Piece::Pawn, Piece::Knight, isCapture));
        writer.Push(Move::Make(from, to, Piece::Pawn, Piece::Bishop, isCapture));
        writer.Push(Move::Make(from, to, Piece::Pawn, Piece::Rook, isCapture));
    }
}

// squares on the line through the king and a piece pinned to it
INLINE Bitboard GetPinLine(const Square kingSquare, const Square pinnedSquare)
{
    const Bitboard kingRookRays = Bitboard::GetRookAttacks(kingSquare);

    if (kingRookRays & pinnedSquare.GetBitboard())
        return kingRookRays & Bitboard::GetRookAttacks(pinnedSquare);

    return Bitboard::GetBishopAttacks(kingSquare) & Bitboard::GetBishopAttacks(pinnedSquare);
}

// Check if a pseudo-legal move keeps our king safe, assuming the side to move is not in check.
// NOTE: castling and en passant are not verified
INLINE bool IsPseudoLegalMoveSafe(const Position& pos, const Threats& threats, const Move move)
{
    if (move.GetPiece() == Piece::King)
        return move.IsCastling() || !(threats.allThreats & move.ToSquare());

    return !(threats.pinned & move.FromSquare()) || (GetPinLine(pos.GetCurrentSideKingSquare(), move.FromSquare()) & move.ToSquare());
}

template<MoveGenerationMode mode, Color sideToMove>
INLINE void GeneratePawnMoveList(const Position& pos, const Bitboard pawns, const Bitboard targetMask, MoveList::BatchWriter& writer)
{
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(sideToMove ^ 1);

    const Bitboard occupiedByCurrent = currentSide.Occupied();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;
    const Bitboard emptySquares = ~occupiedSquares;

    constexpr Direction pawnDirection = sideToMove == White ? Direction::North : Direction::South;
    constexpr Direction pawnRevDirection = sideToMove == White ? Direction::South : Direction::North;
    constexpr Bitboard promotionRank = sideToMove == White ? Bitboard::RankBitboard<7>() : Bitboard::RankBitboard<0>();
    constexpr Bitboard beforePromotionRank = sideToMove == White ? Bitboard::RankBitboard<6>() : Bitboard::RankBitboard<1>();

    if constexpr (mode == MoveGenerationMode::Quiets)
    {
        constexpr Bitboard doublePushesRank = sideToMove == White ? Bitboard::RankBitboard<3>() : Bitboard::RankBitboard<4>();
        const Bitboard singlePushes = pawns.Shift<pawnDirection>() & emptySquares & ~promotionRank;
        const Bitboard doublePushes = singlePushes.Shift<pawnDirection>() & (emptySquares & doublePushesRank) & targetMask;

        (singlePushes & targetMask).Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            writer.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>(),
                targetIndex, Piece::Pawn, Piece::None));
        });

        doublePushes.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            writer.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>().template Shift_Unsafe<pawnRevDirection>(),
                targetIndex, Piece::Pawn, Piece::None));
        });
    }

    if constexpr (mode == MoveGenerationMode::Captures)
    {
        const Bitboard leftCaptures = pawns.Shift<pawnDirection>().West() & occupiedByOpponent & ~promotionRank & targetMask;
        const Bitboard rightCaptures = pawns.Shift<pawnDirection>().East() & occupiedByOpponent & ~promotionRank & targetMask;

        leftCaptures.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            writer.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>().East_Unsafe(),
                targetIndex, Piece::Pawn, Piece::None, true));
        });
        rightCaptures.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            writer.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>().West_Unsafe(),
                targetIndex, Piece::Pawn, Piece::None, true));
        });

        // en passant
        if (pos.GetEnPassantSquare().IsValid() && (pos.GetEnPassantSquare().GetBitboard() & targetMask))
        {
            if (pos.GetEnPassantSquare().File() < 7)
            {
                const Square leftEpSquare = pos.GetEnPassantSquare().Shift<pawnRevDirection>().East_Unsafe();
                if (leftEpSquare.GetBitboard() & pawns)
                {
                    writer.Push(Move::Make(leftEpSquare, pos.GetEnPassantSquare(), Piece::Pawn, Piece::None, true, true));
                }
            }

            if (pos.GetEnPassantSquare().File() > 0)
            {
                const Square rightEpSquare = pos.GetEnPassantSquare().Shift<pawnRevDirection>().West_Unsafe();
                if (rightEpSquare.GetBitboard() & pawns)
                {
                    writer.Push(Move::Make(rightEpSquare, pos.GetEnPassantSquare(), Piece::Pawn, Piece::None, true, true));
                }
            }
        }
    }

    // promotions
    if (beforePromotionRank & pawns)
    {
        const Bitboard promotions = pawns.Shift<pawnDirection>() & emptySquares & promotionRank & targetMask;
        const Bitboard leftCapturePromotions = pawns.Shift<pawnDirection>().West() & occupiedByOpponent & promotionRank & targetMask;
        const Bitboard rightCapturesPromotions = pawns.Shift<pawnDirection>().East() & occupiedByOpponent & promotionRank & targetMask;

        promotions.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            GeneratePromotionsMoveList<mode,false>(Square(targetIndex).Shift_Unsafe<pawnRevDirection>(), targetIndex, writer);
        });
        leftCapturePromotions.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            GeneratePromotionsMoveList<mode,true>(Square(targetIndex).Shift_Unsafe<pawnRevDirection>().East_Unsafe(), targetIndex, writer);
        });
        rightCapturesPromotions.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            GeneratePromotionsMoveList<mode,true>(Square(targetIndex).Shift_Unsafe<pawnRevDirection>().West_Unsafe(), targetIndex, writer);
        });
    }
}

template<MoveGenerationMode mode, Color sideToMove>
INLINE void GeneratePawnMoveList(const Position& pos, MoveList& outMoveList)
{
    auto writer = outMoveList.BeginBatch();
    GeneratePawnMoveList<mode, sideToMove>(pos, pos.GetSide(sideToMove).pawns, ~0ull, writer);
    outMoveList.EndBatch(writer);
}

template<Color sideToMove>
inline void GenerateCastlingMoveList(const Position& pos, MoveList::BatchWriter& writer)
{
    const uint8_t currentSideCastlingRights = sideToMove == White ? pos.GetWhitesCastlingRights() : pos.GetBlacksCastlingRights();

    if (currentSideCastlingRights == 0u)
    {
        return;
    }

    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(sideToMove ^ 1);

    ASSERT(currentSide.king);
    const Square kingSquare = currentSide.GetKingSquare();

    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard opponentAttacks = pos.GetAttackedSquares(sideToMove ^ 1);

    // king can't be in check
    if ((currentSide.king & opponentAttacks) == 0u)
    {
        const Square longCastleRookSquare = pos.GetLongCastleRookSquare(kingSquare, currentSideCastlingRights);
        const Square shortCastleRookSquare = pos.GetShortCastleRookSquare(kingSquare, currentSideCastlingRights);

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
            const Bitboard occupiedSquares = (currentSide.Occupied() | occupiedByOpponent) & ~longCastleRookSquare.GetBitboard() & ~kingSquare.GetBitboard();

            if (0u == (opponentAttacks & kingCrossedSquares) &&
                0u == (kingCrossedSquares & occupiedSquares) &&
                0u == (rookCrossedSquares & occupiedSquares))
            {
                writer.Push(Move::Make(kingSquare, longCastleRookSquare, Piece::King, Piece::None, false, false, true, false));
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
            const Bitboard occupiedSquares = (currentSide.Occupied() | occupiedByOpponent) & ~shortCastleRookSquare.GetBitboard() & ~kingSquare.GetBitboard();

            if (0u == (opponentAttacks & kingCrossedSquares) &&
                0u == (kingCrossedSquares & occupiedSquares) &&
                0u == (rookCrossedSquares & occupiedSquares))
            {
                writer.Push(Move::Make(kingSquare, shortCastleRookSquare, Piece::King, Piece::None, false, false, false, true));
            }
        }
    }
}

template<Color sideToMove, uint32_t MaxSize>
inline void GenerateCastlingMoveList(const Position& pos, TMoveList<MaxSize>& outMoveList)
{
    auto writer = outMoveList.BeginBatch();
    GenerateCastlingMoveList<sideToMove>(pos, writer);
    outMoveList.EndBatch(writer);
}

template<MoveGenerationMode mode, Color sideToMove>
INLINE void GenerateKingMoveList(const Position& pos, const Bitboard threats, MoveList::BatchWriter& writer)
{
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(sideToMove ^ 1);

    ASSERT(currentSide.king);
    const Square kingSquare = currentSide.GetKingSquare();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();

    Bitboard attackBitboard = Bitboard::GetKingAttacks(kingSquare);
    attackBitboard &= ~currentSide.OccupiedExcludingKing(); // can't capture own piece
    attackBitboard &= ~threats; // can't move to a square controlled by the opponent
    attackBitboard &= mode == MoveGenerationMode::Captures ? occupiedByOpponent : ~occupiedByOpponent;

    // regular king moves
    attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
    {
        writer.Push(Move::Make(kingSquare, Square(toIndex), Piece::King, Piece::None, mode == MoveGenerationMode::Captures));
    });

    if constexpr (mode == MoveGenerationMode::Quiets)
    {
        GenerateCastlingMoveList<sideToMove>(pos, writer);
    }
}

// pinned pieces can only move along the line through the king; kept out of line, because pins are rare
template<MoveGenerationMode mode, Color sideToMove>
NO_INLINE uint32_t GeneratePinnedPawnMoveList(const Position& pos, const Bitboard pinnedPawns, MoveList::BatchWriter writer)
{
    const Square kingSquare = pos.GetSide(sideToMove).GetKingSquare();
    pinnedPawns.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        GeneratePawnMoveList<mode, sideToMove>(pos, Square(fromIndex).GetBitboard(), GetPinLine(kingSquare, Square(fromIndex)), writer);
    });
    return writer.count;
}

template<MoveGenerationMode mode, Color sideToMove>
NO_INLINE uint32_t GeneratePinnedSliderMoveList(const Position& pos, const Bitboard pinnedSliders, const Bitboard filter, MoveList::BatchWriter writer)
{
    constexpr const bool isCapture = mode == MoveGenerationMode::Captures;
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const Square kingSquare = currentSide.GetKingSquare();
    const Bitboard occupied = pos.Occupied();

    pinnedSliders.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Square fromSquare(fromIndex);
        const Piece piece = currentSide.GetPieceAtSquare(fromSquare);

        Bitboard attacks;
        if (piece == Piece::Rook)           attacks = Bitboard::GenerateRookAttacks(fromSquare, occupied);
        else if (piece == Piece::Bishop)    attacks = Bitboard::GenerateBishopAttacks(fromSquare, occupied);
        else                                attacks = Bitboard::GenerateQueenAttacks(fromSquare, occupied);

        (attacks & filter & GetPinLine(kingSquare, fromSquare)).Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            writer.Push(Move::Make(fromSquare, Square(toIndex), piece, Piece::None, isCapture));
        });
    });
    return writer.count;
}

template<MoveGenerationMode mode, Color sideToMove>
inline void GenerateMoveList(const Position& pos, const Bitboard threats, const Bitboard pinned, MoveList& outMoveList)
{
    constexpr const bool isCapture = mode == MoveGenerationMode::Captures;
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(sideToMove ^ 1);

    const Bitboard occupiedByCurrent = currentSide.Occupied();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;

    Bitboard filter;
    if constexpr (isCapture)
        filter = occupiedByOpponent;
    else
        filter = ~occupiedSquares;

    auto writer = outMoveList.BeginBatch();

    GeneratePawnMoveList<mode, sideToMove>(pos, currentSide.pawns & ~pinned, ~0ull, writer);
    if (const Bitboard pinnedPawns = currentSide.pawns & pinned) [[unlikely]]
        writer.count = GeneratePinnedPawnMoveList<mode, sideToMove>(pos, pinnedPawns, writer);

    // a pinned knight can never stay on the pin line
    (currentSide.knights & ~pinned).Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = Bitboard::GetKnightAttacks(Square(fromIndex)) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            writer.Push(Move::MakeSimple<Piece::Knight, isCapture>(fromIndex, Square(toIndex)));
        });
    });

    (currentSide.rooks & ~pinned).Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = Bitboard::GenerateRookAttacks(Square(fromIndex), occupiedSquares) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            writer.Push(Move::MakeSimple<Piece::Rook, isCapture>(fromIndex, Square(toIndex)));
        });
    });

    (currentSide.bishops & ~pinned).Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = Bitboard::GenerateBishopAttacks(Square(fromIndex), occupiedSquares) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            writer.Push(Move::MakeSimple<Piece::Bishop, isCapture>(fromIndex, Square(toIndex)));
        });
    });

    (currentSide.queens & ~pinned).Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = filter & Bitboard::GenerateQueenAttacks(Square(fromIndex), occupiedSquares);
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            writer.Push(Move::MakeSimple<Piece::Queen, isCapture>(fromIndex, Square(toIndex)));
        });
    });

    if (const Bitboard pinnedSliders = pinned & (currentSide.rooks | currentSide.bishops | currentSide.queens)) [[unlikely]]
        writer.count = GeneratePinnedSliderMoveList<mode, sideToMove>(pos, pinnedSliders, filter, writer);

    GenerateKingMoveList<mode, sideToMove>(pos, threats, writer);

    outMoveList.EndBatch(writer);
}

template<MoveGenerationMode mode>
INLINE void GenerateMoveList(const Position& pos, const Bitboard threats, const Bitboard pinned, MoveList& outMoveList)
{
    if (pos.GetSideToMove() == White)
    {
        GenerateMoveList<mode, White>(pos, threats, pinned, outMoveList);
    }
    else
    {
        GenerateMoveList<mode, Black>(pos, threats, pinned, outMoveList);
    }
}

// generates no moves that leave the king in check, except while in check, en passant and castling
template<MoveGenerationMode mode>
INLINE void GenerateMoveList(const Position& pos, const Threats& threats, MoveList& outMoveList)
{
    GenerateMoveList<mode>(pos, threats.allThreats, threats.pinned, outMoveList);
}

template<MoveGenerationMode mode>
INLINE void GenerateMoveList(const Position& pos, const Bitboard threats, MoveList& outMoveList)
{
    GenerateMoveList<mode>(pos, threats, 0ull, outMoveList);
}

INLINE void GenerateMoveList(const Position& pos, const Threats& threats, MoveList& outMoveList)
{
    GenerateMoveList<MoveGenerationMode::Captures>(pos, threats, outMoveList);
    GenerateMoveList<MoveGenerationMode::Quiets>(pos, threats, outMoveList);
}

INLINE void GenerateMoveList(const Position& pos, const Bitboard threats, MoveList& outMoveList)
{
    GenerateMoveList<MoveGenerationMode::Captures>(pos, threats, outMoveList);
    GenerateMoveList<MoveGenerationMode::Quiets>(pos, threats, outMoveList);
}

INLINE void GenerateMoveList(const Position& pos, MoveList& outMoveList)
{
    GenerateMoveList(pos, Bitboard::GetKingAttacks(pos.GetOpponentSide().GetKingSquare()), outMoveList);
}

INLINE void GenerateKingMoveList(const Position& pos, const Bitboard threats, MoveList& outMoveList)
{
    auto writer = outMoveList.BeginBatch();
    if (pos.GetSideToMove() == White)
    {
        GenerateKingMoveList<MoveGenerationMode::Captures, White>(pos, threats, writer);
        GenerateKingMoveList<MoveGenerationMode::Quiets, White>(pos, threats, writer);
    }
    else
    {
        GenerateKingMoveList<MoveGenerationMode::Captures, Black>(pos, threats, writer);
        GenerateKingMoveList<MoveGenerationMode::Quiets, Black>(pos, threats, writer);
    }
    outMoveList.EndBatch(writer);
}
