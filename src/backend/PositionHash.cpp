#include "PositionHash.hpp"

// 2*6*64 for pieces piece
// 8 for en passant square
// 16 for castling rights
// 
// This gives 792 64-bit hashes required.
// Note: side-to-move hash is stored separately
alignas(64) uint64_t s_ZobristHash[c_ZobristHashSize];


static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoroshiro128(uint64_t s[2])
{
    // https://prng.di.unimi.it/xoroshiro128plus.c
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37); // c
    return result;
}

void InitZobristHash()
{
    uint64_t s[2] = { 0x2b2fa1f53b24b9f2, 0x0203c66609c7f249 };

    for (uint32_t i = 0; i < c_ZobristHashSize; ++i)
    {
        s_ZobristHash[i] = xoroshiro128(s);
    }
}
