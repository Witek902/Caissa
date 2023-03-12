#include "Bitboard.hpp"
#include "Square.hpp"
#include "Common.hpp"

#include <iostream>

static Bitboard gPawnAttacksBitboard[Square::NumSquares][2];
static Bitboard gKingAttacksBitboard[Square::NumSquares];
static Bitboard gKnightAttacksBitboard[Square::NumSquares];
static Bitboard gRookAttacksMasks[Square::NumSquares];
static Bitboard gBishopAttacksMasks[Square::NumSquares];
static Bitboard gRookAttacksBitboard[Square::NumSquares];
static Bitboard gBishopAttacksBitboard[Square::NumSquares];
static Bitboard gRaysBitboard[Square::NumSquares][8];
static Bitboard gBetweenBitboards[Square::NumSquares][Square::NumSquares];

static const uint64_t cRookMagics[Square::NumSquares] =
{
    0xa8002c000108020ULL, 0x6c00049b0002001ULL, 0x100200010090040ULL, 0x2480041000800801ULL, 0x280028004000800ULL,
    0x900410008040022ULL, 0x280020001001080ULL, 0x2880002041000080ULL, 0xa000800080400034ULL, 0x4808020004000ULL,
    0x2290802004801000ULL, 0x411000d00100020ULL, 0x402800800040080ULL, 0xb000401004208ULL, 0x2409000100040200ULL,
    0x1002100004082ULL, 0x22878001e24000ULL, 0x1090810021004010ULL, 0x801030040200012ULL, 0x500808008001000ULL,
    0xa08018014000880ULL, 0x8000808004000200ULL, 0x201008080010200ULL, 0x801020000441091ULL, 0x800080204005ULL,
    0x1040200040100048ULL, 0x120200402082ULL, 0xd14880480100080ULL, 0x12040280080080ULL, 0x100040080020080ULL,
    0x9020010080800200ULL, 0x813241200148449ULL, 0x491604001800080ULL, 0x100401000402001ULL, 0x4820010021001040ULL,
    0x400402202000812ULL, 0x209009005000802ULL, 0x810800601800400ULL, 0x4301083214000150ULL, 0x204026458e001401ULL,
    0x40204000808000ULL, 0x8001008040010020ULL, 0x8410820820420010ULL, 0x1003001000090020ULL, 0x804040008008080ULL,
    0x12000810020004ULL, 0x1000100200040208ULL, 0x430000a044020001ULL, 0x280009023410300ULL, 0xe0100040002240ULL,
    0x200100401700ULL, 0x2244100408008080ULL, 0x8000400801980ULL, 0x2000810040200ULL, 0x8010100228810400ULL,
    0x2000009044210200ULL, 0x4080008040102101ULL, 0x40002080411d01ULL, 0x2005524060000901ULL, 0x502001008400422ULL,
    0x489a000810200402ULL, 0x1004400080a13ULL, 0x4000011008020084ULL, 0x26002114058042ULL
};

static const uint8_t cRookMagicOffsets[Square::NumSquares] =
{
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52,
};

const uint64_t cBishopMagics[Square::NumSquares] =
{
    0x89a1121896040240ULL, 0x2004844802002010ULL, 0x2068080051921000ULL, 0x62880a0220200808ULL, 0x4042004000000ULL,
    0x100822020200011ULL, 0xc00444222012000aULL, 0x28808801216001ULL, 0x400492088408100ULL, 0x201c401040c0084ULL,
    0x840800910a0010ULL, 0x82080240060ULL, 0x2000840504006000ULL, 0x30010c4108405004ULL, 0x1008005410080802ULL,
    0x8144042209100900ULL, 0x208081020014400ULL, 0x4800201208ca00ULL, 0xf18140408012008ULL, 0x1004002802102001ULL,
    0x841000820080811ULL, 0x40200200a42008ULL, 0x800054042000ULL, 0x88010400410c9000ULL, 0x520040470104290ULL,
    0x1004040051500081ULL, 0x2002081833080021ULL, 0x400c00c010142ULL, 0x941408200c002000ULL, 0x658810000806011ULL,
    0x188071040440a00ULL, 0x4800404002011c00ULL, 0x104442040404200ULL, 0x511080202091021ULL, 0x4022401120400ULL,
    0x80c0040400080120ULL, 0x8040010040820802ULL, 0x480810700020090ULL, 0x102008e00040242ULL, 0x809005202050100ULL,
    0x8002024220104080ULL, 0x431008804142000ULL, 0x19001802081400ULL, 0x200014208040080ULL, 0x3308082008200100ULL,
    0x41010500040c020ULL, 0x4012020c04210308ULL, 0x208220a202004080ULL, 0x111040120082000ULL, 0x6803040141280a00ULL,
    0x2101004202410000ULL, 0x8200000041108022ULL, 0x21082088000ULL, 0x2410204010040ULL, 0x40100400809000ULL,
    0x822088220820214ULL, 0x40808090012004ULL, 0x910224040218c9ULL, 0x402814422015008ULL, 0x90014004842410ULL,
    0x1000042304105ULL, 0x10008830412a00ULL, 0x2520081090008908ULL, 0x40102000a0a60140ULL,
};

const uint8_t cBishopMagicOffsets[Square::NumSquares] =
{
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58,
};

static const uint32_t RookAttackTableSize = 4096;
static uint64_t gRookAttackTable[Square::NumSquares][RookAttackTableSize];

static const uint32_t BishopAttackTableSize = 512;
static uint64_t gBishopAttackTable[Square::NumSquares][BishopAttackTableSize];

///////////////////////////////////////////////////////////////////////////////////////////////////

Bitboard Bitboard::GetRay(const Square square, const Direction dir)
{
    ASSERT(square.IsValid());
    ASSERT(static_cast<uint32_t>(dir) < 8u);
    return gRaysBitboard[square.Index()][static_cast<uint32_t>(dir)];
}

Bitboard Bitboard::GetBetween(const Square squareA, const Square squareB)
{
    ASSERT(squareA.IsValid());
    ASSERT(squareB.IsValid());
    return gBetweenBitboards[squareA.Index()][squareB.Index()];
}

template<>
Bitboard Bitboard::GetPawnAttacks<Color::White>(const Square square)
{
    Bitboard bitboard;
    bitboard = (square.GetBitboard() & ~Bitboard::FileBitboard<0u>()) << 7u;
    bitboard |= (square.GetBitboard() & ~Bitboard::FileBitboard<7u>()) << 9u;
    return bitboard;
}

template<>
Bitboard Bitboard::GetPawnAttacks<Color::Black>(const Square square)
{
    Bitboard bitboard;
    bitboard = (square.GetBitboard() & ~Bitboard::FileBitboard<0u>()) >> 9u;
    bitboard |= (square.GetBitboard() & ~Bitboard::FileBitboard<7u>()) >> 7u;
    return bitboard;
}

template<>
Bitboard Bitboard::GetPawnAttacks<Color::White>(const Bitboard pawns)
{
    Bitboard bitboard;
    bitboard = (pawns & ~Bitboard::FileBitboard<0u>()) << 7u;
    bitboard |= (pawns & ~Bitboard::FileBitboard<7u>()) << 9u;
    return bitboard;
}

template<>
Bitboard Bitboard::GetPawnAttacks<Color::Black>(const Bitboard pawns)
{
    Bitboard bitboard;
    bitboard = (pawns & ~Bitboard::FileBitboard<0u>()) >> 9u;
    bitboard |= (pawns & ~Bitboard::FileBitboard<7u>()) >> 7u;
    return bitboard;
}

Bitboard Bitboard::GetPawnAttacks(const Square square, const Color color)
{
    ASSERT(square.IsValid());
    return gPawnAttacksBitboard[square.Index()][(uint32_t)color];
}

Bitboard Bitboard::GetKingAttacks(const Square square)
{
    ASSERT(square.IsValid());
    return gKingAttacksBitboard[square.Index()];
}

Bitboard Bitboard::GetKnightAttacks(const Square square)
{
    ASSERT(square.IsValid());
    return gKnightAttacksBitboard[square.Index()];
}

Bitboard Bitboard::GetKnightAttacks(const Bitboard squares)
{
    Bitboard result = 0;
    if (squares)
    {
        // based on: https://www.chessprogramming.org/Knight_Pattern
        const Bitboard l1 = (squares >> 1) & 0x7f7f7f7f7f7f7f7full;
        const Bitboard l2 = (squares >> 2) & 0x3f3f3f3f3f3f3f3full;
        const Bitboard r1 = (squares << 1) & 0xfefefefefefefefeull;
        const Bitboard r2 = (squares << 2) & 0xfcfcfcfcfcfcfcfcull;
        const Bitboard h1 = l1 | r1;
        const Bitboard h2 = l2 | r2;
        result = (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
    }
    return result;
}

Bitboard Bitboard::GetRookAttacks(const Square square)
{
    ASSERT(square.IsValid());
    return gRookAttacksBitboard[square.Index()];
}

Bitboard Bitboard::GetBishopAttacks(const Square square)
{
    ASSERT(square.IsValid());
    return gBishopAttacksBitboard[square.Index()];
}

Bitboard Bitboard::GetQueenAttacks(const Square square)
{
    return GetRookAttacks(square) | GetBishopAttacks(square);
}

Bitboard Bitboard::GenerateRookAttacks(const Square square, const Bitboard blockers)
{
    uint64_t b = blockers;
    b &= gRookAttacksMasks[square.Index()];
    b *= cRookMagics[square.Index()];
    b >>= cRookMagicOffsets[square.Index()];
    return gRookAttackTable[square.Index()][b];
}

Bitboard Bitboard::GenerateBishopAttacks(const Square square, const Bitboard blockers)
{
    uint64_t b = blockers;
    b &= gBishopAttacksMasks[square.Index()];
    b *= cBishopMagics[square.Index()];
    b >>= cBishopMagicOffsets[square.Index()];
    return gBishopAttackTable[square.Index()][b];
}

Bitboard Bitboard::GenerateQueenAttacks(const Square square, const Bitboard blockers)
{
    return GenerateRookAttacks(square, blockers) | GenerateBishopAttacks(square, blockers);
}

Bitboard Bitboard::GenerateRookAttacks_Slow(const Square square, const Bitboard blockers)
{
    uint32_t blockerIndexN;
    uint64_t bitboardN = GetRay(square, Direction::North);
    if (Bitboard(bitboardN & blockers).BitScanForward(blockerIndexN))
    {
        bitboardN &= ~GetRay(blockerIndexN, Direction::North);
    }

    uint32_t blockerIndexE;
    uint64_t bitboardE = GetRay(square, Direction::East);
    if (Bitboard(bitboardE & blockers).BitScanForward(blockerIndexE))
    {
        bitboardE &= ~GetRay(blockerIndexE, Direction::East);
    }

    uint32_t blockerIndexS;
    uint64_t bitboardS = GetRay(square, Direction::South);
    if (Bitboard(bitboardS & blockers).BitScanReverse(blockerIndexS))
    {
        bitboardS &= ~GetRay(blockerIndexS, Direction::South);
    }

    uint32_t blockerIndexW;
    uint64_t bitboardW = GetRay(square, Direction::West);
    if (Bitboard(bitboardW & blockers).BitScanReverse(blockerIndexW))
    {
        bitboardW &= ~GetRay(blockerIndexW, Direction::West);
    }

    return bitboardN | bitboardS | bitboardE | bitboardW;
}

Bitboard Bitboard::GenerateBishopAttacks_Slow(const Square square, const Bitboard blockers)
{
    uint32_t blockerIndexNW;
    uint64_t bitboardNW = GetRay(square, Direction::NorthWest);
    if (Bitboard(bitboardNW & blockers).BitScanForward(blockerIndexNW))
    {
        bitboardNW &= ~GetRay(blockerIndexNW, Direction::NorthWest);
    }

    uint32_t blockerIndexNE;
    uint64_t bitboardNE = GetRay(square, Direction::NorthEast);
    if (Bitboard(bitboardNE & blockers).BitScanForward(blockerIndexNE))
    {
        bitboardNE &= ~GetRay(blockerIndexNE, Direction::NorthEast);
    }

    uint32_t blockerIndexSE;
    uint64_t bitboardSE = GetRay(square, Direction::SouthEast);
    if (Bitboard(bitboardSE & blockers).BitScanReverse(blockerIndexSE))
    {
        bitboardSE &= ~GetRay(blockerIndexSE, Direction::SouthEast);
    }

    uint32_t blockerIndexSW;
    uint64_t bitboardSW = GetRay(square, Direction::SouthWest);
    if (Bitboard(bitboardSW & blockers).BitScanReverse(blockerIndexSW))
    {
        bitboardSW &= ~GetRay(blockerIndexSW, Direction::SouthWest);
    }

    return bitboardNW | bitboardNE | bitboardSE | bitboardSW;
}

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

        gRaysBitboard[squareIndex][(uint32_t)Direction::North] = 0x0101010101010100ull << squareIndex;
        gRaysBitboard[squareIndex][(uint32_t)Direction::South] = 0x0080808080808080ull >> (63 - squareIndex);
        gRaysBitboard[squareIndex][(uint32_t)Direction::East] = 2 * ((1ull << (squareIndex | 7)) - (1ull << squareIndex));
        gRaysBitboard[squareIndex][(uint32_t)Direction::West] = (1ull << squareIndex) - (1ull << (squareIndex & 56u));
        gRaysBitboard[squareIndex][(uint32_t)Direction::NorthEast] = Bitboard::ShiftRight(0x8040201008040200ull, square.File()) << (square.Rank() * 8u);
        gRaysBitboard[squareIndex][(uint32_t)Direction::NorthWest] = Bitboard::ShiftLeft(0x102040810204000ull, 7u - square.File()) << (square.Rank() * 8u);
        gRaysBitboard[squareIndex][(uint32_t)Direction::SouthEast] = Bitboard::ShiftRight(0x2040810204080ull, square.File()) >> ((7 - square.Rank()) * 8u);
        gRaysBitboard[squareIndex][(uint32_t)Direction::SouthWest] = Bitboard::ShiftLeft(0x40201008040201ull, 7u - square.File()) >> ((7 - square.Rank()) * 8u);
    }
}

void InitPawnAttacks()
{
    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        gPawnAttacksBitboard[squareIndex][0] = Bitboard::GetPawnAttacks<Color::White>(square);
        gPawnAttacksBitboard[squareIndex][1] = Bitboard::GetPawnAttacks<Color::Black>(square);
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

            bitboard |= targetSquare.GetBitboard();
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

            bitboard |= targetSquare.GetBitboard();
        }

        gKnightAttacksBitboard[squareIndex] = bitboard;
    }
}

static void InitRookAttacks()
{
    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        gRookAttacksBitboard[squareIndex] =
            Bitboard::RankBitboard(square.Rank()) |
            Bitboard::FileBitboard(square.File());
    }
}

static void InitBishopAttacks()
{
    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        gBishopAttacksBitboard[squareIndex] =
            Bitboard::GetRay(square, Direction::NorthEast) |
            Bitboard::GetRay(square, Direction::NorthWest) |
            Bitboard::GetRay(square, Direction::SouthEast) |
            Bitboard::GetRay(square, Direction::SouthWest);
    }
}

static Bitboard GetRookAttackMask(const Square square)
{
    Bitboard b = 0;

    b |= Bitboard::FileBitboard(square.File()) & (~Bitboard::RankBitboard<0>() & ~Bitboard::RankBitboard<7>());
    b |= Bitboard::RankBitboard(square.Rank()) & (~Bitboard::FileBitboard<0>() & ~Bitboard::FileBitboard<7>());

    // exclude self
    b &= ~square.GetBitboard();

    return b;
}

static Bitboard GetBishopAttackMask(const Square square)
{
    Bitboard b = 0;

    b |= Bitboard::GetRay(square, Direction::NorthEast);
    b |= Bitboard::GetRay(square, Direction::NorthWest);
    b |= Bitboard::GetRay(square, Direction::SouthEast);
    b |= Bitboard::GetRay(square, Direction::SouthWest);

    // exclude self and borders
    b &= ~square.GetBitboard();
    b &= ~Bitboard::FileBitboard<0>();
    b &= ~Bitboard::RankBitboard<0>();
    b &= ~Bitboard::FileBitboard<7>();
    b &= ~Bitboard::RankBitboard<7>();

    return b;
}

static void InitRookMagicBitboards()
{
    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const uint64_t magic = cRookMagics[squareIndex];
        const uint64_t shift = cRookMagicOffsets[squareIndex];

        const Square square(squareIndex);
        const Bitboard attackMask = GetRookAttackMask(square);

        gRookAttacksMasks[squareIndex] = attackMask;

        // compute number of possible occluder layouts
        const uint32_t attackMaskBits = PopCount(attackMask);
        const uint32_t numBlockerSets = 1 << attackMaskBits;

        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            // reconstruct (masked) blockers bitboard
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);

            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            ASSERT(tableIndex < RookAttackTableSize);

            gRookAttackTable[squareIndex][tableIndex] = Bitboard::GenerateRookAttacks_Slow(square, blockerBitboard);
        }

#ifndef CONFIGURATION_FINAL
        // validate
        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);
            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            const Bitboard expected = Bitboard::GenerateRookAttacks_Slow(square, blockerBitboard);
            ASSERT(gRookAttackTable[squareIndex][tableIndex] == expected);
            ASSERT(Bitboard::GenerateRookAttacks(square, blockerBitboard) == expected);
        }
#endif // CONFIGURATION_FINAL
    }
}

static void InitBishopMagicBitboards()
{
    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const uint64_t magic = cBishopMagics[squareIndex];
        const uint64_t shift = cBishopMagicOffsets[squareIndex];

        const Square square(squareIndex);
        const Bitboard attackMask = GetBishopAttackMask(square);

        gBishopAttacksMasks[squareIndex] = attackMask;

        // compute number of possible occluder layouts
        const uint32_t attackMaskBits = PopCount(attackMask);
        const uint32_t numBlockerSets = 1 << attackMaskBits;

        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            // reconstruct (masked) blockers bitboard
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);

            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            ASSERT(tableIndex < BishopAttackTableSize);

            gBishopAttackTable[squareIndex][tableIndex] = Bitboard::GenerateBishopAttacks_Slow(square, blockerBitboard);
        }

#ifndef CONFIGURATION_FINAL
        // validate
        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);
            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            const Bitboard expected = Bitboard::GenerateBishopAttacks_Slow(square, blockerBitboard);
            ASSERT(gBishopAttackTable[squareIndex][tableIndex] == expected);
            ASSERT(Bitboard::GenerateBishopAttacks(square, blockerBitboard) == expected);
        }
#endif // CONFIGURATION_FINAL
    }
}

/*
static void InitMagicBitboards()
{
    const uint32_t attacksTableSizeBits = 12u;
    const uint32_t attacksTableSize = 1u << attacksTableSizeBits;

    uint64_t attacksTable[attacksTableSize];


    for (uint32_t squareIndex = 1; squareIndex < 64; ++squareIndex)
    {
        const Square square(squareIndex);

        std::cout << "Square: " << square.ToString() << std::endl;

        const Bitboard attackMask = GetRookAttackMask(square);

        //std::cout << attackMask.Print() << std::endl;

        // compute number of possible occluder layouts
        const uint32_t attackMaskBits = (uint32_t)__popcnt64(attackMask);
        const uint32_t numBlockerSets = 1 << attackMaskBits;

        uint32_t bestMatchedSets = 0;

        for (uint32_t i = 0; ; ++i)
        {
            uint64_t magic = XorShift();

            memset(attacksTable, 0xFF, sizeof(attacksTable));

            if (i % 100000 == 0)
            {
                std::cout << "Iter " << i << ", matches: " << bestMatchedSets << '/' << numBlockerSets << std::endl;
            }

            bool match = true;

            // iterate each occluder layout
            for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
            {
                // reconstruct (masked) blockers bitboard
                const Bitboard blockerBitboard = ParallelBitsDeposit(blockersIndex + 1u, attackMask);

                const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> (64u - attacksTableSizeBits));

                const uint64_t expected = Bitboard::GenerateRookAttacks_Slow(square, blockerBitboard);

                if (attacksTable[tableIndex] == UINT64_MAX)
                {
                    attacksTable[tableIndex] = expected;
                }
                else if (attacksTable[tableIndex] != expected)
                {
                    match = false;
                    bestMatchedSets = std::max(bestMatchedSets, blockersIndex);
                    break;
                }
            }

            if (match)
            {
                std::cout << "Found magic: " << std::hex << magic << std::dec << " (iter " << i << ")" << std::endl;
                break;
            }
        }
    }
}
*/

static void InitBetweenBitboards()
{
    memset(gBetweenBitboards, 0, sizeof(gBetweenBitboards));

    for (uint32_t squareA = 0; squareA < 64; ++squareA)
    {
        for (uint32_t squareB = 0; squareB < 64; ++squareB)
        {
            if (squareA != squareB)
            {
                if (Bitboard::GetRookAttacks(squareA) & Square(squareB).GetBitboard())
                {
                    gBetweenBitboards[squareA][squareB] |=
                        Bitboard::GenerateRookAttacks(squareA, Square(squareB).GetBitboard()) &
                        Bitboard::GenerateRookAttacks(squareB, Square(squareA).GetBitboard());
                }

                if (Bitboard::GetBishopAttacks(squareA) & Square(squareB).GetBitboard())
                {
                    gBetweenBitboards[squareA][squareB] |=
                        Bitboard::GenerateBishopAttacks(squareA, Square(squareB).GetBitboard()) &
                        Bitboard::GenerateBishopAttacks(squareB, Square(squareA).GetBitboard());
                }
            }
        }
    }
}

void InitBitboards()
{
    InitRays();
    InitPawnAttacks();
    InitKingAttacks();
    InitKnightAttacks();
    InitRookAttacks();
    InitBishopAttacks();

    InitRookMagicBitboards();
    InitBishopMagicBitboards();

    InitBetweenBitboards();

    //InitMagicBitboards();
}
