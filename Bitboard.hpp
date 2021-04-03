#pragma once

#include "Common.hpp"

#include <inttypes.h>
#include <assert.h>
#include <string>

class Square;

enum class RayDir
{
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
};

struct Bitboard
{
    uint64_t value;

    Bitboard() = default;
    INLINE constexpr Bitboard(uint64_t value) : value(value) {}
    INLINE constexpr Bitboard(const Bitboard& other) = default;
    INLINE constexpr Bitboard& operator = (const Bitboard& other) = default;
    INLINE constexpr Bitboard operator & (const Bitboard& rhs) const { return value & rhs.value; }
    INLINE constexpr Bitboard operator | (const Bitboard& rhs) const { return value | rhs.value; }
    INLINE constexpr Bitboard operator ^ (const Bitboard& rhs) const { return value ^ rhs.value; }
    INLINE constexpr Bitboard& operator &= (const Bitboard& rhs) { value &= rhs.value; return *this; }
    INLINE constexpr Bitboard& operator |= (const Bitboard& rhs) { value |= rhs.value; return *this; }
    INLINE constexpr Bitboard& operator ^= (const Bitboard& rhs) { value ^= rhs.value; return *this; }
    INLINE constexpr operator uint64_t() const { return value; }
    INLINE constexpr Bitboard operator ~() const { return ~value; }

    // debug print
    std::string Print() const;

    template<typename Func>
    INLINE void Iterate(const Func func) const
    {
        uint64_t mask = value;
        unsigned long index;
        while (_BitScanForward64(&index, mask))
        {
            mask &= ~(1ull << index);
            func(index);
        };
    }

    template<uint32_t rank>
    INLINE static constexpr Bitboard RankBitboard()
    {
        static_assert(rank < 8u, "Invalid rank");
        return 0xFFull << (8u * rank);
    }

    template<uint32_t file>
    INLINE static constexpr Bitboard FileBitboard()
    {
        static_assert(file < 8u, "Invalid file");
        return 0x0101010101010101ull << file;
    }

    INLINE static Bitboard RankBitboard(uint32_t rank)
    {
        ASSERT(rank < 8u);
        return 0xFFull << (8u * rank);
    }

    INLINE static Bitboard FileBitboard(uint32_t file)
    {
        ASSERT(file < 8u);
        return 0x0101010101010101ull << file;
    }

    INLINE static Bitboard ShiftRight(Bitboard board, uint32_t num)
    {
        for (uint32_t i = 0; i < num; i++)
        {
            board = (board << 1u) & (~FileBitboard<0>());
        }
        return board;
    }


    INLINE static Bitboard ShiftLeft(Bitboard board, uint32_t num)
    {
        for (uint32_t i = 0; i < num; i++)
        {
            board = (board >> 1u) & (~FileBitboard<7>());
        }
        return board;
    }

    INLINE uint32_t Count() const
    {
        return static_cast<uint32_t>(__popcnt64(value));
    }

    INLINE bool BitScanForward(uint32_t& outIndex) const
    {
        unsigned long index;
        if (_BitScanForward64(&index, value))
        {
            outIndex = index;
            return true;
        }
        else
        {
            return false;
        }
    }

    INLINE bool BitScanReverse(uint32_t& outIndex) const
    {
        unsigned long index;
        if (_BitScanReverse64(&index, value))
        {
            outIndex = index;
            return true;
        }
        else
        {
            return false;
        }
    }

    static Bitboard GetRay(const Square square, const RayDir dir);

    static Bitboard GetKingAttacks(const Square square);
    static Bitboard GetKnightAttacks(const Square square);
    static Bitboard GetRookAttacks(const Square square);

    static Bitboard GenerateRookAttacks(const Square square, const Bitboard blockers);
    static Bitboard GenerateBishopAttacks(const Square square, const Bitboard blockers);
};

void InitBitboards();
