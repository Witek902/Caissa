#include "PositionHash.hpp"

#include <random>

alignas(64) uint64_t s_ZobristHash[c_ZobristHashSize];

void InitZobristHash()
{
    std::mt19937_64 mt;

    for (uint32_t i = 0; i < c_ZobristHashSize; ++i)
    {
        s_ZobristHash[i] = mt();
    }
}
