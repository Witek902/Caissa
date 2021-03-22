#include "Bitboard.hpp"
#include "Square.hpp"

std::string Bitboard::Print() const
{
    std::string str;

    // reverse, because first are printed higher ranks
    for (uint8_t rank = 8u; rank-- > 0u; )
    {
        str += '1' + rank;
        str += " ";

        for (uint8_t file = 0; file < 8u; ++file)
        {
            const uint32_t offset = rank * 8u + file;

            if ((value >> offset) & 1ull)
            {
                str += "X";
            }
            else
            {
                str += ".";
            }

            if (file < 7u)
            {
                str += " ";
            }
        }

        str += "\n";
    }

    str += "  a b c d e f g h\n";

    return str;
}

Bitboard Bitboard::GenerateRookAttacks(const Square& rookSquare, Bitboard occupiedBitboard)
{
    uint64_t bitboard = 0u;

    const uint8_t f = rookSquare.File();
    const uint8_t r = rookSquare.Rank();

    uint32_t file, rank;

    for (file = f + 1; file < 8u; ++file) // go right
    {
        const uint64_t mask = 1ull << (r * 8 + file);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    for (file = f; file-- > 0u; ) // go left
    {
        const uint64_t mask = 1ull << (r * 8 + file);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    for (rank = r + 1; rank < 8u; ++rank) // go up
    {
        const uint64_t mask = 1ull << (rank * 8 + f);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    for (rank = r; rank-- > 0u; ) // go down
    {
        const uint64_t mask = 1ull << (rank * 8 + f);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    return bitboard;
}

Bitboard Bitboard::GenerateBishopAttacks(const Square& bishopSquare, Bitboard occupiedBitboard)
{
    uint64_t bitboard = 0u;

    const uint8_t f = bishopSquare.File();
    const uint8_t r = bishopSquare.Rank();

    uint32_t file, rank;

    for (file = f + 1, rank = r + 1; (file < 8u) && (rank < 8u); ++file, ++rank) // go up-right
    {
        const uint64_t mask = 1ull << (rank * 8 + file);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    for (file = f, rank = r + 1; (file-- > 0u) && (rank < 8u); ++rank) // go up-left
    {
        const uint64_t mask = 1ull << (rank * 8 + file);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    for (file = f + 1, rank = r; (file < 8u) && (rank-- > 0u); ++file) // go down-right
    {
        const uint64_t mask = 1ull << (rank * 8 + file);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    for (file = f, rank = r; (file-- > 0u) && (rank-- > 0u); ) // go down-left
    {
        const uint64_t mask = 1ull << (rank * 8 + file);
        bitboard |= mask;
        if (occupiedBitboard.value & mask) break;
    }

    return bitboard;
}