#include "PositionHash.hpp"

// 2*6*64 for pieces piece
// 8 for en passant square
// 16 for castling rights
// 
// This gives 792 64-bit hashes required.
// Note: side-to-move hash is stored separately
alignas(64) uint64_t s_ZobristHash[c_ZobristHashSize];


void InitZobristHash()
{
    uint64_t s = 0xa7a57e2fba74af2cULL;

    for (uint32_t i = 0; i < c_ZobristHashSize; ++i)
    {
        // SplitMix64 hash function
        // https://prng.di.unimi.it/splitmix64.c
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;

        s_ZobristHash[i] = z ^ (z >> 31);
    }
}
