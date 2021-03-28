#pragma once

#include "Common.hpp"

#include <inttypes.h>
#include <assert.h>
#include <string>

class Square;

struct Bitboard
{
    uint64_t value;

    Bitboard() = default;
    __forceinline Bitboard(uint64_t value) : value(value) {}
    __forceinline Bitboard(const Bitboard& other) = default;
    __forceinline Bitboard& operator = (const Bitboard& other) = default;
    __forceinline Bitboard operator & (const Bitboard& rhs) const { return value & rhs.value; }
    __forceinline Bitboard operator | (const Bitboard& rhs) const { return value | rhs.value; }
    __forceinline Bitboard operator ^ (const Bitboard& rhs) const { return value ^ rhs.value; }
    __forceinline Bitboard& operator &= (const Bitboard& rhs) { value &= rhs.value; return *this; }
    __forceinline Bitboard& operator |= (const Bitboard& rhs) { value |= rhs.value; return *this; }
    __forceinline Bitboard& operator ^= (const Bitboard& rhs) { value ^= rhs.value; return *this; }
    __forceinline operator uint64_t() const { return value; }

    // debug print
    std::string Print() const;

    template<typename Func>
    __forceinline void Iterate(const Func func) const
    {
        uint64_t mask = value;
        unsigned long index;
        while (_BitScanForward64(&index, mask))
        {
            mask &= ~(1ull << index);
            func(index);
        };
    }

    __forceinline static Bitboard RankBitboard(uint32_t rank)
    {
        ASSERT(rank < 8u);
        return 0xFFull << (8u * rank);
    }

    __forceinline static Bitboard FileBitboard(uint32_t file)
    {
        ASSERT(file < 8u);
        return 0x0101010101010101ull << file;
    }

    __forceinline uint32_t Count() const
    {
        return static_cast<uint32_t>(__popcnt64(value));
    }

    static Bitboard GetKingAttacks(const Square& knightSquare);
    static Bitboard GetKnightAttacks(const Square& knightSquare);

    static Bitboard GenerateRookAttacks(const Square& rookSquare, const Bitboard occupiedBitboard);
    static Bitboard GenerateBishopAttacks(const Square& bishopSquare, const Bitboard occupiedBitboard);
};

void InitBitboards();
