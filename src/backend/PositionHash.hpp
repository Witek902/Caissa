#pragma once

#include "Common.hpp"
#include "Piece.hpp"
#include "Square.hpp"

static constexpr uint32_t c_ZobristHashSize = 792;

static constexpr uint64_t c_SideToMoveZobristHash = 1u;

extern const uint64_t s_ZobristHash[c_ZobristHashSize];

INLINE static uint64_t GetPieceZobristHash(const Color color, const Piece piece, const uint32_t squareIndex)
{
    const uint32_t pieceIndex = (uint32_t)piece - (uint32_t)Piece::Pawn;
    const uint32_t offset = (uint32_t)color + 2 * (squareIndex + 64 * pieceIndex);
    ASSERT(offset < 2 * 6 * 64);
    return s_ZobristHash[offset];
}

INLINE static uint64_t GetEnPassantFileZobristHash(uint32_t fileIndex)
{
    ASSERT(fileIndex < 8);

    // skip position hashes
    const uint32_t offset = (2 * 6 * 64) + fileIndex;
    ASSERT(offset < 2 * 6 * 64 + 8);
    return s_ZobristHash[offset];
}

INLINE static uint64_t GetCastlingRightsZobristHash(const Color color, uint32_t rookIndex)
{
    ASSERT(rookIndex < 8);

    // skip position hashes and en passant hashes
    const uint32_t offset = (2 * 6 * 64 + 8) + 2 * rookIndex + (uint32_t)color;
    ASSERT(offset < 2 * 6 * 64 + 8 + 16);
    return s_ZobristHash[offset];
}
