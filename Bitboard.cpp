#include "Bitboard.hpp"
#include "Square.hpp"
#include "Common.hpp"

static Bitboard gKingAttacksBitboard[Square::NumSquares];
static Bitboard gKnightAttacksBitboard[Square::NumSquares];

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

static void InitKingAttacks()
{
    const uint32_t numKingOffsets = 8u;
    const int32_t kingFileOffsets[numKingOffsets] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const int32_t kingRankOffsets[numKingOffsets] = { 1, 1, 0, -1, -1, -1, 0, 1 };

    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        Bitboard bitboard{ 0 };

        for (uint32_t i = 0; i < numKingOffsets; ++i)
        {
            const int32_t targetFile = (int32_t)square.File() + kingFileOffsets[i];
            const int32_t targetRank = (int32_t)square.Rank() + kingRankOffsets[i];

            // out of board
            if (targetFile < 0 || targetRank < 0 || targetFile >= 8 || targetRank >= 8) continue;

            const Square targetSquare((uint8_t)targetFile, (uint8_t)targetRank);

            bitboard |= targetSquare.Bitboard();
        }

        gKingAttacksBitboard[squareIndex] = bitboard;
    }
}

static void InitKnightAttacks()
{
    const uint32_t numKnightOffsets = 8u;
    const int32_t knightFileOffsets[numKnightOffsets] = { 1, 2, 2, 1, -1, -2, -2, -1 };
    const int32_t knightRankOffsets[numKnightOffsets] = { 2, 1, -1, -2, -2, -1, 1, 2 };

    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        Bitboard bitboard{ 0 };

        for (uint32_t i = 0; i < numKnightOffsets; ++i)
        {
            const int32_t targetFile = (int32_t)square.File() + knightFileOffsets[i];
            const int32_t targetRank = (int32_t)square.Rank() + knightRankOffsets[i];

            // out of board
            if (targetFile < 0 || targetRank < 0 || targetFile >= 8 || targetRank >= 8) continue;

            const Square targetSquare((uint8_t)targetFile, (uint8_t)targetRank);

            bitboard |= targetSquare.Bitboard();
        }

        gKnightAttacksBitboard[squareIndex] = bitboard;
    }
}

void InitBitboards()
{
    InitKingAttacks();
    InitKnightAttacks();
}

Bitboard Bitboard::GetKingAttacks(const Square& kingSquare)
{
    ASSERT(kingSquare.IsValid());
    return gKingAttacksBitboard[kingSquare.Index()];
}

Bitboard Bitboard::GetKnightAttacks(const Square& knightSquare)
{
    ASSERT(knightSquare.IsValid());
    return gKnightAttacksBitboard[knightSquare.Index()];
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