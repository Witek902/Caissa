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

    INLINE MaterialKey() : value(0) { }

    INLINE MaterialKey(const MaterialKey& rhs) : value(rhs.value) { }

    INLINE MaterialKey(
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

    INLINE bool operator == (const MaterialKey& rhs) const
    {
        return value == rhs.value;
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
