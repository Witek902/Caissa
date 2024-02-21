#pragma once

#include "Bitboard.hpp"

#include <string>

enum SquareName : uint32_t
{
    Square_a1, Square_b1, Square_c1, Square_d1, Square_e1, Square_f1, Square_g1, Square_h1,
    Square_a2, Square_b2, Square_c2, Square_d2, Square_e2, Square_f2, Square_g2, Square_h2,
    Square_a3, Square_b3, Square_c3, Square_d3, Square_e3, Square_f3, Square_g3, Square_h3,
    Square_a4, Square_b4, Square_c4, Square_d4, Square_e4, Square_f4, Square_g4, Square_h4,
    Square_a5, Square_b5, Square_c5, Square_d5, Square_e5, Square_f5, Square_g5, Square_h5,
    Square_a6, Square_b6, Square_c6, Square_d6, Square_e6, Square_f6, Square_g6, Square_h6,
    Square_a7, Square_b7, Square_c7, Square_d7, Square_e7, Square_f7, Square_g7, Square_h7,
    Square_a8, Square_b8, Square_c8, Square_d8, Square_e8, Square_f8, Square_g8, Square_h8,
};

class Square
{
public:
    static constexpr uint32_t NumSquares = 64;

    static void Init();

    INLINE static const Square Invalid()
    {
        Square square;
        square.mIndex = 0xFF;
        return square;
    }

    INLINE Square() = default;

    INLINE Square(uint32_t value)
        : mIndex(static_cast<uint8_t>(value))
    {
        ASSERT(value < 64u);
    }

    INLINE Square(SquareName name)
        : mIndex(static_cast<uint8_t>(name))
    {
        ASSERT(mIndex < 64u);
    }

    INLINE Square(uint8_t file, uint8_t rank)
    {
        ASSERT(file < 8u);
        ASSERT(rank < 8u);
        mIndex = file + (rank * 8u);
    }

    INLINE Square(const Square&) = default;
    INLINE Square& operator = (const Square&) = default;

    INLINE bool operator == (const Square& rhs) const { return mIndex == rhs.mIndex; }
    INLINE bool operator != (const Square& rhs) const { return mIndex != rhs.mIndex; }

    INLINE uint8_t Index() const
    {
        return mIndex;
    }

    INLINE Bitboard GetBitboard() const
    {
        return 1ull << mIndex;
    }

    // aka. row
    INLINE uint8_t Rank() const
    {
        return mIndex / 8u;
    }

    // aka. column
    INLINE uint8_t File() const
    {
        return mIndex % 8u;
    }

    INLINE uint8_t Diagonal() const
    {
        return Rank() - File() + 7;
    }

    INLINE uint8_t AntiDiagonal() const
    {
        return Rank() + File();
    }

    INLINE uint8_t RelativeRank(const Color color) const
    {
        const uint8_t rank = mIndex / 8u;
        return color == White ? rank : (7 - rank);
    }

    INLINE Square North() const
    {
        return Rank() < 7 ? mIndex + 8 : Invalid();
    }

    INLINE Square South() const
    {
        return Rank() > 0 ? mIndex - 8 : Invalid();
    }

    INLINE Square East() const
    {
        return File() < 7 ? (mIndex + 1) : Invalid();
    }

    INLINE Square West() const
    {
        return File() > 0 ? (mIndex - 1) : Invalid();
    }

    INLINE Square North_Unsafe() const { ASSERT(Rank() < 7); return mIndex + 8; }
    INLINE Square South_Unsafe() const { ASSERT(Rank() > 0); return mIndex - 8; }
    INLINE Square East_Unsafe() const { ASSERT(File() < 7); return mIndex + 1; }
    INLINE Square West_Unsafe() const { ASSERT(File() > 0); return mIndex - 1; }

    template<Direction dir>
    INLINE constexpr Square Shift() const
    {
        if constexpr (dir == Direction::North) return North();
        if constexpr (dir == Direction::South) return South();
        if constexpr (dir == Direction::East) return East();
        if constexpr (dir == Direction::West) return West();
        if constexpr (dir == Direction::NorthEast) return North().East();
        if constexpr (dir == Direction::NorthWest) return North().West();
        if constexpr (dir == Direction::SouthEast) return South().East();
        if constexpr (dir == Direction::SouthWest) return South().East();
    }

    template<Direction dir>
    INLINE constexpr Square Shift_Unsafe() const
    {
        if constexpr (dir == Direction::North) return North_Unsafe();
        if constexpr (dir == Direction::South) return South_Unsafe();
        if constexpr (dir == Direction::East) return East_Unsafe();
        if constexpr (dir == Direction::West) return West_Unsafe();
        if constexpr (dir == Direction::NorthEast) return North_Unsafe().East_Unsafe();
        if constexpr (dir == Direction::NorthWest) return North_Unsafe().West_Unsafe();
        if constexpr (dir == Direction::SouthEast) return South_Unsafe().East_Unsafe();
        if constexpr (dir == Direction::SouthWest) return South_Unsafe().East_Unsafe();
    }

    INLINE Square FlippedFile() const
    {
        return mIndex ^ 0b000111;
    }

    INLINE Square FlippedRank() const
    {
        return mIndex ^ (uint8_t)0b111000;
    }

    INLINE Square FlippedFileAndRank() const
    {
        return mIndex ^ (uint8_t)0b111111;
    }

    bool IsCorner() const
    {
        return mIndex == Square_a1 || mIndex == Square_a8 || mIndex == Square_h1 || mIndex == Square_h8;
    }

    int32_t EdgeDistance() const
    {
        const int32_t r = Rank();
        const int32_t f = File();
        const int32_t rd = std::min(r, 7 - r);
        const int32_t fd = std::min(f, 7 - f);
        return std::min(rd, fd);
    }

    int32_t DarkCornerDistance() const
    {
        return 7 - std::abs(7 - (int32_t)Rank() - (int32_t)File());
    }

    int32_t AnyCornerDistance() const
    {
        const int32_t r = Rank();
        const int32_t f = File();
        const int32_t a1 = std::max(r, f);
        const int32_t a8 = std::max(7 - r, f);
        const int32_t h1 = std::max(r, 7 - f);
        const int32_t h8 = std::max(7 - r, 7 - f);
        return std::min(std::min(a1, a8), std::min(h1, h8));
    }

    INLINE static int32_t Distance(const Square a, const Square b)
    {
        return sDistances[64u * a.mIndex + b.mIndex];
    }

    static int32_t ComputeDistance(const Square a, const Square b)
    {
        ASSERT(a.IsValid());
        ASSERT(b.IsValid());
        const int32_t r = std::abs((int32_t)a.Rank() - (int32_t)b.Rank());
        const int32_t f = std::abs((int32_t)a.File() - (int32_t)b.File());
        return std::max(r, f);
    }

    static Square FromString(const std::string& str);

    std::string ToString() const;

    INLINE bool IsValid() const
    {
        return mIndex < 64;
    }

    uint8_t mIndex;

    alignas(CACHELINE_SIZE) static uint8_t sDistances[NumSquares * NumSquares];
};

INLINE const Bitboard operator & (const Bitboard& lhs, const Square& rhs)
{
    return lhs & rhs.GetBitboard();
}