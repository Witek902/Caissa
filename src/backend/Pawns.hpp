#pragma once

#include "Common.hpp"


inline bool IsPassedPawn(Square pawnSquare, const Color sideToMove, Bitboard whitePawns, Bitboard blackPawns)
{
    if (sideToMove != Color::White)
    {
        pawnSquare = pawnSquare.FlippedRank();
        std::swap(whitePawns, blackPawns);
        whitePawns = whitePawns.MirroredVertically();
        blackPawns = blackPawns.MirroredVertically();
    }

    if (pawnSquare.Rank() >= 6)
    {
        // pawn is ready to promotion - consider is as passed
        return true;
    }
    else
    {
        constexpr const Bitboard fileMask = Bitboard::FileBitboard<0>();

        Bitboard passedPawnMask = fileMask << (pawnSquare.Index() + 8u);

        // blocked pawn
        if ((whitePawns & passedPawnMask) != 0)
        {
            return false;
        }

        if (pawnSquare.File() > 0u) passedPawnMask |= fileMask << (pawnSquare.Index() + 7u);
        if (pawnSquare.File() < 7u) passedPawnMask |= fileMask << (pawnSquare.Index() + 9u);

        if ((blackPawns & passedPawnMask) == 0)
        {
            return true;
        }
    }

    return false;
}

inline int32_t CountPassedPawns(const Bitboard ourPawns, const Bitboard theirPawns)
{
    int32_t count = 0;

    ourPawns.Iterate([&](uint32_t square) INLINE_LAMBDA
    {
        const uint32_t rank = square / 8;
        const uint32_t file = square % 8;
        
        if (rank >= 6)
        {
            // pawn is ready to promotion - consider is as passed
            count++;
        }
        else
        {
            constexpr const Bitboard fileMask = Bitboard::FileBitboard<0>();

            Bitboard passedPawnMask = fileMask << (square + 8);
            if (file > 0) passedPawnMask |= fileMask << (square + 7);
            if (file < 7) passedPawnMask |= fileMask << (square + 9);

            if ((theirPawns & passedPawnMask) == 0)
            {
                count++;
            }
        }
    });

    return count;
}
