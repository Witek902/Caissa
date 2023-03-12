#pragma once

#include "Position.hpp"
#include "MoveList.hpp"

enum class MoveGenerationMode
{
    Captures,   // captures and queen promotion
    Quiets,     // quiet moves and underpromotions
};

template<MoveGenerationMode mode, bool isCapture>
INLINE inline void GeneratePromotionsMoveList(const Square from, const Square to, MoveList& outMoveList)
{
    if constexpr (mode == MoveGenerationMode::Captures)
    {
        outMoveList.Push(Move::Make(from, to, Piece::Pawn, Piece::Queen, isCapture));
    }
    else // generate underpromotions
    {
        outMoveList.Push(Move::Make(from, to, Piece::Pawn, Piece::Knight, isCapture));
        outMoveList.Push(Move::Make(from, to, Piece::Pawn, Piece::Bishop, isCapture));
        outMoveList.Push(Move::Make(from, to, Piece::Pawn, Piece::Rook, isCapture));
    }
}

template<MoveGenerationMode mode, Color sideToMove>
inline void GeneratePawnMoveList(const Position& pos, MoveList& outMoveList)
{
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(GetOppositeColor(sideToMove));

    const Bitboard occupiedByCurrent = currentSide.Occupied();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;
    const Bitboard emptySquares = ~occupiedSquares;

    constexpr Direction pawnDirection = sideToMove == Color::White ? Direction::North : Direction::South;
    constexpr Direction pawnRevDirection = sideToMove == Color::White ? Direction::South : Direction::North;
    constexpr Bitboard doublePushesRank = sideToMove == Color::White ? Bitboard::RankBitboard<3>() : Bitboard::RankBitboard<4>();
    constexpr Bitboard promotionRank = sideToMove == Color::White ? Bitboard::RankBitboard<7>() : Bitboard::RankBitboard<0>();
    constexpr Bitboard beforePromotionRank = sideToMove == Color::White ? Bitboard::RankBitboard<6>() : Bitboard::RankBitboard<1>();

    if constexpr (mode == MoveGenerationMode::Quiets)
    {
        const Bitboard singlePushes = currentSide.pawns.Shift<pawnDirection>() & emptySquares & ~promotionRank;
        const Bitboard doublePushes = singlePushes.Shift<pawnDirection>() & (emptySquares & doublePushesRank);

        singlePushes.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>(),
                targetIndex, Piece::Pawn, Piece::None));
        });

        doublePushes.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>().Shift_Unsafe<pawnRevDirection>(),
                targetIndex, Piece::Pawn, Piece::None));
        });
    }

    if constexpr (mode == MoveGenerationMode::Captures)
    {
        const Bitboard leftCaptures = currentSide.pawns.Shift<pawnDirection>().West() & occupiedByOpponent & ~promotionRank;
        const Bitboard rightCaptures = currentSide.pawns.Shift<pawnDirection>().East() & occupiedByOpponent & ~promotionRank;

        leftCaptures.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>().East_Unsafe(),
                targetIndex, Piece::Pawn, Piece::None, true));
        });
        rightCaptures.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(
                Square(targetIndex).Shift_Unsafe<pawnRevDirection>().West_Unsafe(),
                targetIndex, Piece::Pawn, Piece::None, true));
        });

        // en passant
        if (pos.GetEnPassantSquare().IsValid())
        {
            if (pos.GetEnPassantSquare().File() < 7)
            {
                const Square leftEpSquare = pos.GetEnPassantSquare().Shift<pawnRevDirection>().East_Unsafe();
                if (leftEpSquare.GetBitboard() & currentSide.pawns)
                {
                    outMoveList.Push(Move::Make(leftEpSquare, pos.GetEnPassantSquare(), Piece::Pawn, Piece::None, true, true));
                }
            }

            if (pos.GetEnPassantSquare().File() > 0)
            {
                const Square rightEpSquare = pos.GetEnPassantSquare().Shift<pawnRevDirection>().West_Unsafe();
                if (rightEpSquare.GetBitboard() & currentSide.pawns)
                {
                    outMoveList.Push(Move::Make(rightEpSquare, pos.GetEnPassantSquare(), Piece::Pawn, Piece::None, true, true));
                }
            }
        }
    }

    // promotions
    if (beforePromotionRank & currentSide.pawns)
    {
        const Bitboard promotions = currentSide.pawns.Shift<pawnDirection>() & emptySquares & promotionRank;
        const Bitboard leftCapturePromotions = currentSide.pawns.Shift<pawnDirection>().West() & occupiedByOpponent & promotionRank;
        const Bitboard rightCapturesPromotions = currentSide.pawns.Shift<pawnDirection>().East() & occupiedByOpponent & promotionRank;

        promotions.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            GeneratePromotionsMoveList<mode,false>(Square(targetIndex).Shift_Unsafe<pawnRevDirection>(), targetIndex, outMoveList);
        });
        leftCapturePromotions.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            GeneratePromotionsMoveList<mode,true>(Square(targetIndex).Shift_Unsafe<pawnRevDirection>().East_Unsafe(), targetIndex, outMoveList);
        });
        rightCapturesPromotions.Iterate([&](uint32_t targetIndex) INLINE_LAMBDA
        {
            GeneratePromotionsMoveList<mode,true>(Square(targetIndex).Shift_Unsafe<pawnRevDirection>().West_Unsafe(), targetIndex, outMoveList);
        });
    }
}

template<MoveGenerationMode mode, Color sideToMove>
inline void GenerateKingMoveList(const Position& pos, MoveList& outMoveList)
{
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(GetOppositeColor(sideToMove));

    ASSERT(currentSide.king);
    const Square kingSquare = currentSide.GetKingSquare();
    const Square opponentKingSquare = opponentSide.GetKingSquare();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();

    Bitboard attackBitboard = Bitboard::GetKingAttacks(kingSquare);
    attackBitboard &= ~currentSide.OccupiedExcludingKing(); // can't capture own piece
    attackBitboard &= ~Bitboard::GetKingAttacks(opponentKingSquare); // can't move to square controlled by opponent's king
    attackBitboard &= mode == MoveGenerationMode::Captures ? occupiedByOpponent : ~occupiedByOpponent;

    // regular king moves
    attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
    {
        outMoveList.Push(Move::Make(kingSquare, Square(toIndex), Piece::King, Piece::None, mode == MoveGenerationMode::Captures));
    });

    // castling
    if constexpr (mode == MoveGenerationMode::Quiets)
    {
        const uint8_t currentSideCastlingRights = sideToMove == Color::White ? pos.GetWhitesCastlingRights() : pos.GetBlacksCastlingRights();

        if (0 != currentSideCastlingRights)
        {
            const Bitboard opponentAttacks = pos.GetAttackedSquares(GetOppositeColor(sideToMove));

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
                        outMoveList.Push(Move::Make(kingSquare, longCastleRookSquare, Piece::King, Piece::None, false, false, true, false));
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
                        outMoveList.Push(Move::Make(kingSquare, shortCastleRookSquare, Piece::King, Piece::None, false, false, false, true));
                    }
                }
            }
        }
    }
}

template<MoveGenerationMode mode, Color sideToMove>
inline void GenerateMoveList(const Position& pos, MoveList& outMoveList)
{
    const SidePosition& currentSide = pos.GetSide(sideToMove);
    const SidePosition& opponentSide = pos.GetSide(GetOppositeColor(sideToMove));

    const Bitboard occupiedByCurrent = currentSide.Occupied();
    const Bitboard occupiedByOpponent = opponentSide.Occupied();
    const Bitboard occupiedSquares = occupiedByCurrent | occupiedByOpponent;

    Bitboard filter = ~currentSide.Occupied(); // can't capture own piece
    filter &= mode == MoveGenerationMode::Captures ? occupiedByOpponent : ~occupiedByOpponent;

    GeneratePawnMoveList<mode, sideToMove>(pos, outMoveList);

    currentSide.knights.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = Bitboard::GetKnightAttacks(Square(fromIndex)) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(fromIndex, Square(toIndex), Piece::Knight, Piece::None, mode == MoveGenerationMode::Captures));
        });
    });

    currentSide.rooks.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = Bitboard::GenerateRookAttacks(Square(fromIndex), occupiedSquares) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(fromIndex, Square(toIndex), Piece::Rook, Piece::None, mode == MoveGenerationMode::Captures));
        });
    });

    currentSide.bishops.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = Bitboard::GenerateBishopAttacks(Square(fromIndex), occupiedSquares) & filter;
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(fromIndex, toIndex, Piece::Bishop, Piece::None, mode == MoveGenerationMode::Captures));
        });
    });

    currentSide.queens.Iterate([&](uint32_t fromIndex) INLINE_LAMBDA
    {
        const Bitboard attackBitboard = filter & Bitboard::GenerateQueenAttacks(Square(fromIndex), occupiedSquares);
        attackBitboard.Iterate([&](uint32_t toIndex) INLINE_LAMBDA
        {
            outMoveList.Push(Move::Make(fromIndex, Square(toIndex), Piece::Queen, Piece::None, mode == MoveGenerationMode::Captures));
        });
    });

    GenerateKingMoveList<mode, sideToMove>(pos, outMoveList);
}

template<MoveGenerationMode mode>
inline void GenerateMoveList(const Position& pos, MoveList& outMoveList)
{
    if (pos.GetSideToMove() == Color::White)
    {
        GenerateMoveList<mode, Color::White>(pos, outMoveList);
    }
    else
    {
        GenerateMoveList<mode, Color::Black>(pos, outMoveList);
    }
}

inline void GenerateMoveList(const Position& pos, MoveList& outMoveList)
{
    GenerateMoveList<MoveGenerationMode::Captures>(pos, outMoveList);
    GenerateMoveList<MoveGenerationMode::Quiets>(pos, outMoveList);
}

inline void GenerateKingMoveList(const Position& pos, MoveList& outMoveList)
{
    if (pos.GetSideToMove() == Color::White)
    {
        GenerateKingMoveList<MoveGenerationMode::Captures, Color::White>(pos, outMoveList);
        GenerateKingMoveList<MoveGenerationMode::Quiets, Color::White>(pos, outMoveList);
    }
    else
    {
        GenerateKingMoveList<MoveGenerationMode::Captures, Color::Black>(pos, outMoveList);
        GenerateKingMoveList<MoveGenerationMode::Quiets, Color::Black>(pos, outMoveList);
    }
}