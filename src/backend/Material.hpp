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
        uint64_t __padding : 4;
    };

    uint64_t value;

    INLINE constexpr MaterialKey() : value(0) { }

    INLINE constexpr MaterialKey(const MaterialKey& rhs) : value(rhs.value) { }
    INLINE constexpr explicit MaterialKey(uint64_t v) : value(v) { }

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
        __padding       = 0;
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

    INLINE constexpr bool IsSymetric() const
    {
        return (value & 0x3FFFFFFF) == (value >> 30);
    }

    INLINE constexpr MaterialKey SwappedColors() const
    {
        const uint64_t whitesValue = (value >> 30) & 0x3FFFFFFFull;
        const uint64_t blacksValue = (value & 0x3FFFFFFFull) << 30;
        return MaterialKey(whitesValue | blacksValue);
    }

    INLINE uint32_t GetActivePiecesCount() const
    {
        uint32_t count = 0;
        count += numWhitePawns > 0 ? 1 : 0;
        count += numWhiteKnights > 0 ? 1 : 0;
        count += numWhiteBishops > 0 ? 1 : 0;
        count += numWhiteRooks > 0 ? 1 : 0;
        count += numWhiteQueens > 0 ? 1 : 0;
        count += numBlackPawns > 0 ? 1 : 0;
        count += numBlackKnights > 0 ? 1 : 0;
        count += numBlackBishops > 0 ? 1 : 0;
        count += numBlackRooks > 0 ? 1 : 0;
        count += numBlackQueens > 0 ? 1 : 0;
        return count;
    }

    uint32_t GetNeuralNetworkInputsNumber() const;

    void FromString(const char* str);
    std::string ToString() const;
};

///

enum MaterialMask : uint16_t
{
    MaterialMask_NONE           = 0,

    MaterialMask_WhitePawn      = 1 << 0,
    MaterialMask_WhiteKnight    = 1 << 1,
    MaterialMask_WhiteBishop    = 1 << 2,
    MaterialMask_WhiteRook      = 1 << 3,
    MaterialMask_WhiteQueen     = 1 << 4,

    MaterialMask_BlackPawn      = 1 << 5,
    MaterialMask_BlackKnight    = 1 << 6,
    MaterialMask_BlackBishop    = 1 << 7,
    MaterialMask_BlackRook      = 1 << 8,
    MaterialMask_BlackQueen     = 1 << 9,

    MaterialMask_MAX            = 1 << 10,
    MaterialMask_ALL            = MaterialMask_MAX - 1,
    MaterialMask_WhitesMAX = MaterialMask_BlackPawn,
};

INLINE constexpr MaterialMask operator | (MaterialMask a, MaterialMask b)
{
    return static_cast<MaterialMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

MaterialMask BuildMaterialMask(const Position& pos);

INLINE MaterialMask FlipColor(const MaterialMask mask)
{
    return MaterialMask((mask >> 5) | ((mask & 0x1F) << 5));
}

///

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
