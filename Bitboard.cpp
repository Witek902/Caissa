#include "Bitboard.hpp"
#include "Square.hpp"
#include "Common.hpp"

static Bitboard gKingAttacksBitboard[Square::NumSquares];
static Bitboard gKnightAttacksBitboard[Square::NumSquares];
static Bitboard gRaysBitboard[Square::NumSquares][8];

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

static void InitRays()
{
    for (uint32_t squareIndex = 0; squareIndex < 64; squareIndex++)
    {
        const Square square(squareIndex);

        gRaysBitboard[squareIndex][(uint32_t)RayDir::North] = 0x0101010101010100ull << squareIndex;
        gRaysBitboard[squareIndex][(uint32_t)RayDir::South] = 0x0080808080808080ull >> (63 - squareIndex);
        gRaysBitboard[squareIndex][(uint32_t)RayDir::East] = 2 * ((1ull << (squareIndex | 7)) - (1ull << squareIndex));
        gRaysBitboard[squareIndex][(uint32_t)RayDir::West] = (1ull << squareIndex) - (1ull << (squareIndex & 56u));
        gRaysBitboard[squareIndex][(uint32_t)RayDir::NorthEast] = Bitboard::ShiftRight(0x8040201008040200ull, square.File()) << (square.Rank() * 8u);
        gRaysBitboard[squareIndex][(uint32_t)RayDir::NorthWest] = Bitboard::ShiftLeft(0x102040810204000ull, 7u - square.File()) << (square.Rank() * 8u);
        gRaysBitboard[squareIndex][(uint32_t)RayDir::SouthEast] = Bitboard::ShiftRight(0x2040810204080ull, square.File()) >> ((7 - square.Rank()) * 8u);
        gRaysBitboard[squareIndex][(uint32_t)RayDir::SouthWest] = Bitboard::ShiftLeft(0x40201008040201ull, 7u - square.File()) >> ((7 - square.Rank()) * 8u);
    }
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
    InitRays();
    InitKingAttacks();
    InitKnightAttacks();
}

Bitboard Bitboard::GetRay(const Square square, const RayDir dir)
{
    ASSERT(square.IsValid());
    ASSERT(static_cast<uint32_t>(dir) < 8u);
    return gRaysBitboard[square.Index()][static_cast<uint32_t>(dir)];
}

Bitboard Bitboard::GetKingAttacks(const Square kingSquare)
{
    ASSERT(kingSquare.IsValid());
    return gKingAttacksBitboard[kingSquare.Index()];
}

Bitboard Bitboard::GetKnightAttacks(const Square knightSquare)
{
    ASSERT(knightSquare.IsValid());
    return gKnightAttacksBitboard[knightSquare.Index()];
}

Bitboard Bitboard::GenerateRookAttacks(const Square square, const Bitboard blockers)
{
    uint32_t blockerIndexN;
    uint64_t bitboardN = GetRay(square, RayDir::North);
    if (Bitboard(bitboardN & blockers).BitScanForward(blockerIndexN))
    {
        bitboardN &= ~GetRay(blockerIndexN, RayDir::North);
    }

    uint32_t blockerIndexE;
    uint64_t bitboardE = GetRay(square, RayDir::East);
    if (Bitboard(bitboardE & blockers).BitScanForward(blockerIndexE))
    {
        bitboardE &= ~GetRay(blockerIndexE, RayDir::East);
    }

    uint32_t blockerIndexS;
    uint64_t bitboardS = GetRay(square, RayDir::South);
    if (Bitboard(bitboardS & blockers).BitScanReverse(blockerIndexS))
    {
        bitboardS &= ~GetRay(blockerIndexS, RayDir::South);
    }

    uint32_t blockerIndexW;
    uint64_t bitboardW = GetRay(square, RayDir::West);
    if (Bitboard(bitboardW & blockers).BitScanReverse(blockerIndexW))
    {
        bitboardW &= ~GetRay(blockerIndexW, RayDir::West);
    }

    return bitboardN | bitboardS | bitboardE | bitboardW;
}

Bitboard Bitboard::GenerateBishopAttacks(const Square square, const Bitboard blockers)
{
    uint32_t blockerIndexNW;
    uint64_t bitboardNW = GetRay(square, RayDir::NorthWest);
    if (Bitboard(bitboardNW & blockers).BitScanForward(blockerIndexNW))
    {
        bitboardNW &= ~GetRay(blockerIndexNW, RayDir::NorthWest);
    }

    uint32_t blockerIndexNE;
    uint64_t bitboardNE = GetRay(square, RayDir::NorthEast);
    if (Bitboard(bitboardNE & blockers).BitScanForward(blockerIndexNE))
    {
        bitboardNE &= ~GetRay(blockerIndexNE, RayDir::NorthEast);
    }

    uint32_t blockerIndexSE;
    uint64_t bitboardSE = GetRay(square, RayDir::SouthEast);
    if (Bitboard(bitboardSE & blockers).BitScanReverse(blockerIndexSE))
    {
        bitboardSE &= ~GetRay(blockerIndexSE, RayDir::SouthEast);
    }

    uint32_t blockerIndexSW;
    uint64_t bitboardSW = GetRay(square, RayDir::SouthWest);
    if (Bitboard(bitboardSW & blockers).BitScanReverse(blockerIndexSW))
    {
        bitboardSW &= ~GetRay(blockerIndexSW, RayDir::SouthWest);
    }

    return bitboardNW | bitboardNE | bitboardSE | bitboardSW;
}