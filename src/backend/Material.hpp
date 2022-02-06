#pragma once

#include "Common.hpp"

#include <string>

union MaterialKey
{
    UNNAMED_STRUCT struct
    {
        uint64_t numWhitePawns : 6;
        uint64_t numWhiteKnights : 6;
        uint64_t numWhiteBishops : 6;
        uint64_t numWhiteRooks : 6;
        uint64_t numWhiteQueens : 6;
        uint64_t numBlackPawns : 6;
        uint64_t numBlackKnights : 6;
        uint64_t numBlackBishops : 6;
        uint64_t numBlackRooks : 6;
        uint64_t numBlackQueens : 6;
    };

    uint64_t value;

    INLINE constexpr MaterialKey() : value(0) { }

    INLINE constexpr MaterialKey(const MaterialKey& rhs) : value(rhs.value) { }

    INLINE constexpr MaterialKey(
        uint32_t wp, uint32_t wk, uint32_t wb, uint32_t wr, uint32_t wq,
        uint32_t bp, uint32_t bk, uint32_t bb, uint32_t br, uint32_t bq)
    {
        numWhitePawns   = wp;
        numWhiteKnights = wk;
        numWhiteBishops = wb;
        numWhiteRooks   = wr;
        numWhiteQueens  = wq;
        numBlackPawns   = bp;
        numBlackKnights = bk;
        numBlackBishops = bb;
        numBlackRooks   = br;
        numBlackQueens  = bq;
    }

    INLINE MaterialKey& operator = (const MaterialKey& rhs)
    {
        value = rhs.value;
        return *this;
    }

    INLINE constexpr bool operator == (const MaterialKey& rhs) const
    {
        return value == rhs.value;
    }

    INLINE constexpr uint32_t CountAll() const
    {
        return
            (uint32_t)numWhitePawns + (uint32_t)numWhiteKnights + (uint32_t)numWhiteBishops + (uint32_t)numWhiteRooks + (uint32_t)numWhiteQueens +
            (uint32_t)numBlackPawns + (uint32_t)numBlackKnights + (uint32_t)numBlackBishops + (uint32_t)numBlackRooks + (uint32_t)numBlackQueens;
    }

    uint32_t GetNeuralNetworkInputsNumber() const;

    std::string ToString() const;
};

namespace std
{

template <>
struct hash<MaterialKey>
{
    INLINE std::size_t operator()(const MaterialKey& k) const
    {
        return k.value;
    }
};

} // namespace std
