#pragma once

#include <inttypes.h>
#include <assert.h>
#include <string>

#include "Bitboard.hpp"

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
    Square() : mIndex(0xFF) { }

    Square(uint32_t value)
        : mIndex(static_cast<uint8_t>(value))
    {
        assert(value < 64u);
    }

    Square(SquareName name)
        : mIndex(static_cast<uint8_t>(name))
    {
        assert(mIndex < 64u);
    }

    Square(uint8_t file, uint8_t rank)
    {
        assert(file < 8u);
        assert(rank < 8u);
        mIndex = file + (rank * 8u);
    }

    Square(const Square&) = default;
    Square& operator = (const Square&) = default;

    bool operator == (const Square& rhs) const { return mIndex == rhs.mIndex; }
    bool operator != (const Square& rhs) const { return mIndex != rhs.mIndex; }

    uint8_t Index() const
    {
        return mIndex;
    }

    Bitboard Bitboard() const
    {
        return 1ull << mIndex;
    }

    // aka. row
    uint8_t Rank() const
    {
        return mIndex / 8u;
    }

    // aka. column
    uint8_t File() const
    {
        return mIndex % 8u;
    }

    static Square FromString(const std::string& str);

    std::string ToString() const;

    bool IsValid() const
    {
        return mIndex < 64;
    }

private:
    uint8_t mIndex;
};
