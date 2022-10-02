#include "PositionHash.hpp"

#include <random>

// 800 random bytes to store Zobrist hash
// 
// 2*6*64 for pieces piece
// 8 for en passant square
// 16 for castlight rights
// 
// This gives 792 64-bit hashes required. We overlap all the hashes (1 byte offsets),
// so required storage is 8x smaller.
// Note: side-to-move hash is stored separately
alignas(64) uint64_t s_ZobristHash[128];

void InitZobristHash()
{
    std::mt19937_64 mt(0x06db3aa64a37b526LLU);
    std::uniform_int_distribution<uint64_t> distr;

    for (uint32_t i = 0; i < 100; ++i)
    {
        s_ZobristHash[i] = distr(mt);
    }
}
