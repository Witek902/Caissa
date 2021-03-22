#pragma once

#include <inttypes.h>
#include <assert.h>
#include <string>

#include "Bitboard.hpp"

class Square
{
public:
    Square() : mIndex(0xFF) { }

    Square(uint32_t value)
        : mIndex(static_cast<uint8_t>(value))
    {
        assert(value < 64u);
    }

    Square(uint8_t file, uint8_t rank)
    {
        assert(file < 8u);
        assert(rank < 8u);
        mIndex = file + (rank * 8u);
    }

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

    std::string ToString() const
    {
        std::string str;
        str += 'a' + File();
        str += '1' + Rank();
        return str;
    }

    bool IsValid() const
    {
        return mIndex < 64;
    }

private:
    uint8_t mIndex;
};
