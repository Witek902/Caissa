#include "Bitboard.hpp"
#include "Square.hpp"

static Bitboard gPawnAttacksBitboard[Square::NumSquares][2];
static Bitboard gKingAttacksBitboard[Square::NumSquares];
static Bitboard gKnightAttacksBitboard[Square::NumSquares];
static Bitboard gRookAttacksBitboard[Square::NumSquares];
static Bitboard gBishopAttacksBitboard[Square::NumSquares];
static Bitboard gRaysBitboard[Square::NumSquares][8];
static Bitboard gBetweenBitboards[Square::NumSquares][Square::NumSquares];

#ifdef USE_BMI2
#define USE_PEXT_ATTACKS
#endif // USE_BMI2


#ifdef USE_PEXT_ATTACKS

struct AttackData
{
    Bitboard mask = 0;
    uint32_t offset = 0;
};

static constexpr uint32_t cRookAttackTableSize = 102400;
static constexpr uint32_t cBishopAttackTableSize = 5248;

static AttackData gRookAttacksData[Square::NumSquares];
static AttackData gBishopAttacksData[Square::NumSquares];
static Bitboard gRookAttackTable[cRookAttackTableSize];
static Bitboard gBishopAttackTable[cBishopAttackTableSize];

#else

struct MagicData
{
    uint64_t mask;   // magic mask (relevant occupancy bits)
    uint64_t magic;  // magic 64-bit multiplier
    uint32_t shift;  // shift right value (= 64 - index bits)
    uint32_t offset; // offset into the flat attack table
};

static MagicData gRookMagicData[Square::NumSquares] =
{
    { 0, 0xf898020605feffffULL, 52, 0 }, // 0 - a1
    { 0, 0xe72003d4544dffffULL, 53, 0 }, // 1 - b1
    { 0, 0x860010a3231dfffeULL, 53, 0 }, // 2 - c1, 11 bits
    { 0, 0xc6000b13534dffffULL, 54, 0 }, // 3 - d1, 11 bits -> 10 bits optimized
    { 0, 0xea0002f2f15fffe8ULL, 53, 0 }, // 4 - e1, 11 bits
    { 0, 0x8a0001797777ffa4ULL, 53, 0 }, // 5 - f1, 11 bits
    { 0, 0x1c00169a9b2fffd8ULL, 54, 0 }, // 6 - g1, 11 bits -> 10 bits optimized
    { 0, 0x4600009494a9fffbULL, 52, 0 }, // 7 - h1

    { 0, 0x8099e00807fbfffcULL, 53, 0 }, // 8 - a2, 11 bits
    { 0, 0x000ef0131327fffaULL, 54, 0 }, // 9 - b2, 10 bits
    { 0, 0xcae2002cac99fffaULL, 55, 0 }, // 10 - c2, 10 bits -> 9 bits optimized
    { 0, 0x003e0017574dfffdULL, 55, 0 }, // 11 - d2, 10 bits -> 9 bits optimized
    { 0, 0x0472000b2b25ffffULL, 55, 0 }, // 12 - e2, 10 bits -> 9 bits optimized
    { 0, 0xddb42003ef33dfffULL, 54, 0 }, // 13 - f2, 10 bits
    { 0, 0xa0f92001617bfff3ULL, 54, 0 }, // 14 - g2, 10 bits
    { 0, 0xf3a61000505bfff0ULL, 53, 0 }, // 15 - h2, 11 bits

    { 0, 0x0022878001e24000ULL, 53, 0 }, // 16
    { 0, 0x1090810021004010ULL, 54, 0 }, // 17
    { 0, 0x0801030040200012ULL, 54, 0 }, // 18
    { 0, 0x0500808008001000ULL, 54, 0 }, // 19
    { 0, 0x0a08018014000880ULL, 54, 0 }, // 20
    { 0, 0x8000808004000200ULL, 54, 0 }, // 21
    { 0, 0x0201008080010200ULL, 54, 0 }, // 22
    { 0, 0x0801020000441091ULL, 53, 0 }, // 23

    { 0, 0x0000800080204005ULL, 53, 0 }, // 24
    { 0, 0x1040200040100048ULL, 54, 0 }, // 25
    { 0, 0x0000120200402082ULL, 54, 0 }, // 26
    { 0, 0x0d14880480100080ULL, 54, 0 }, // 27
    { 0, 0x0012040280080080ULL, 54, 0 }, // 28
    { 0, 0x0100040080020080ULL, 54, 0 }, // 29
    { 0, 0x9020010080800200ULL, 54, 0 }, // 30
    { 0, 0x0813241200148449ULL, 53, 0 }, // 31

    { 0, 0x0491604001800080ULL, 53, 0 }, // 32
    { 0, 0x0100401000402001ULL, 54, 0 }, // 33
    { 0, 0x4820010021001040ULL, 54, 0 }, // 34
    { 0, 0x0400402202000812ULL, 54, 0 }, // 35
    { 0, 0x0209009005000802ULL, 54, 0 }, // 36
    { 0, 0x0810800601800400ULL, 54, 0 }, // 37
    { 0, 0x4301083214000150ULL, 54, 0 }, // 38
    { 0, 0x204026458e001401ULL, 53, 0 }, // 39

    { 0, 0x0040204000808000ULL, 53, 0 }, // 40
    { 0, 0x8001008040010020ULL, 54, 0 }, // 41
    { 0, 0x8410820820420010ULL, 54, 0 }, // 42
    { 0, 0x1003001000090020ULL, 54, 0 }, // 43
    { 0, 0x0804040008008080ULL, 54, 0 }, // 44
    { 0, 0x0012000810020004ULL, 54, 0 }, // 45
    { 0, 0x1000100200040208ULL, 54, 0 }, // 46
    { 0, 0x430000a044020001ULL, 53, 0 }, // 47

    { 0, 0x05fffcfd0baee5a0ULL, 54, 0 }, // 48 - a7, 11 bits -> 10 bits optimized
    { 0, 0xaafff77791519200ULL, 55, 0 }, // 49 - b7, 10 bits -> 9 bits optimized
    { 0, 0x4fbffe95fead3200ULL, 55, 0 }, // 50 - c7, 10 bits -> 9 bits optimized
    { 0, 0x1afffed5fece1a00ULL, 55, 0 }, // 51 - d7, 10 bits -> 9 bits optimized
    { 0, 0xcd0fffe9ffe6ce00ULL, 55, 0 }, // 52 - f7, 10 bits -> 9 bits optimized
    { 0, 0xca7fff75ff734600ULL, 55, 0 }, // 53 - f7, 10 bits -> 9 bits optimized
    { 0, 0xee07ffd7d87058c0ULL, 55, 0 }, // 54 - g7, 10 bits -> 9 bits optimized
    { 0, 0xfa23fff5f63c96a0ULL, 54, 0 }, // 55 - h7, 11 bits -> 10 bits optimized

    { 0, 0x58ffff49ff3dc666ULL, 53, 0 }, // 56 - a8, 12 bits -> 11 bits optimized
    { 0, 0x50fffe7e842f84c6ULL, 54, 0 }, // 57 - b8, 11 bits -> 10 bits optimized
    { 0, 0xf9bfffb1ffaddce6ULL, 54, 0 }, // 58 - c8, 11 bits -> 10 bits optimized
    { 0, 0x8d5fff31ff2fdbeaULL, 54, 0 }, // 59 - d8, 11 bits -> 10 bits optimized
    { 0, 0x2d5ffff9fff6a9aeULL, 54, 0 }, // 60 - e8, 11 bits -> 10 bits optimized
    { 0, 0x2007fffdfffe5b8aULL, 53, 0 }, // 61 - f8, 11 bits
    { 0, 0x3793fff4b7f56564ULL, 54, 0 }, // 62 - g8, 11 bits -> 10 bits optimized
    { 0, 0x0ef3fffd3bfd525aULL, 53, 0 }, // 63 - h8, 12 bits -> 11 bits optimized
};

static MagicData gBishopMagicData[Square::NumSquares] =
{
    { 0, 0x09d64908cc8407ffULL, 59, 0 }, // 0 - a1, 6 bits -> 5 bits optimized
    { 0, 0x021b2781e917fd11ULL, 60, 0 }, // 1 - b1, 5 bits -> 4 bits optimized
    { 0, 0x3ce8453a77fcb215ULL, 59, 0 }, // 2 - c1, 5 bits
    { 0, 0xba482053fab49c54ULL, 59, 0 }, // 3 - d1, 5 bits
    { 0, 0x00038ee0c0110280ULL, 59, 0 }, // 4 - e1
    { 0, 0x0100822020200011ULL, 59, 0 }, // 5 - f1
    { 0, 0x2444112a53ff0218ULL, 60, 0 }, // 6 - g1, 5 bits -> 4 bits optimized
    { 0, 0x641100580589ff04ULL, 59, 0 }, // 7 - h1, 6 bits -> 5 bits optimized

    { 0, 0x080c3406a662c7f3ULL, 60, 0 }, // 8 - a2, 5 bits -> 4 bits optimized
    { 0, 0x1288ea531218c3fcULL, 60, 0 }, // 9 - b2, 5 bits -> 4 bits optimized
    { 0, 0xb651e45b3573fbe4ULL, 59, 0 }, // 10
    { 0, 0x0000082080240060ULL, 59, 0 }, // 11
    { 0, 0x2000840504006000ULL, 59, 0 }, // 12
    { 0, 0x30010c4108405004ULL, 59, 0 }, // 13
    { 0, 0x03107352a620ff8cULL, 60, 0 }, // 14 - g2, 5 bits -> 4 bits optimized
    { 0, 0x0c2022232b157fc0ULL, 60, 0 }, // 15 - h2, 5 bits -> 4 bits optimized

    { 0, 0x0240080b96096fe2ULL, 60, 0 }, // 16 - a3, 5 bits -> 4 bits optimized
    { 0, 0x0020028619dd67e0ULL, 60, 0 }, // 17 - b3, 5 bits -> 4 bits optimized
    { 0, 0x34700187143a27ffULL, 57, 0 }, // 18 - c3
    { 0, 0x1004002802102001ULL, 57, 0 }, // 19
    { 0, 0x0841000820080811ULL, 57, 0 }, // 20
    { 0, 0x0040200200a42008ULL, 57, 0 }, // 21
    { 0, 0x00740400a589bfa1ULL, 60, 0 }, // 22 - g3, 5 bits -> 4 bits optimized
    { 0, 0x8a2a012662753fd0ULL, 60, 0 }, // 23 - h3, 5 bits -> 4 bits optimized

    { 0, 0x0520040470104290ULL, 59, 0 }, // 24
    { 0, 0x1004040051500081ULL, 59, 0 }, // 25
    { 0, 0x2002081833080021ULL, 57, 0 }, // 26
    { 0, 0x100300c10400c200ULL, 55, 0 }, // 27
    { 0, 0x0009040086006104ULL, 55, 0 }, // 28
    { 0, 0x0658810000806011ULL, 57, 0 }, // 29
    { 0, 0x0188071040440a00ULL, 59, 0 }, // 30
    { 0, 0x4800404002011c00ULL, 59, 0 }, // 31

    { 0, 0x0104442040404200ULL, 59, 0 }, // 32
    { 0, 0x0511080202091021ULL, 59, 0 }, // 33
    { 0, 0x0004022401120400ULL, 57, 0 }, // 34
    { 0, 0x80c0040400080120ULL, 55, 0 }, // 35
    { 0, 0x8040010040820802ULL, 55, 0 }, // 36
    { 0, 0x0480810700020090ULL, 57, 0 }, // 37
    { 0, 0x0102008e00040242ULL, 59, 0 }, // 38
    { 0, 0x0809005202050100ULL, 59, 0 }, // 39

    { 0, 0x100fe54d35204045ULL, 60, 0 }, // 40 - a6, 5 bits -> 4 bits optimized
    { 0, 0x4a67f3255a18e020ULL, 60, 0 }, // 41 - b6, 5 bits -> 4 bits optimized
    { 0, 0x0019001802081400ULL, 57, 0 }, // 42
    { 0, 0x0200014208040080ULL, 57, 0 }, // 43
    { 0, 0x3308082008200100ULL, 57, 0 }, // 44
    { 0, 0x041010500040c020ULL, 57, 0 }, // 45
    { 0, 0x0a3fa632476a8407ULL, 60, 0 }, // 46 - g6, 5 bits -> 4 bits optimized
    { 0, 0x021fb54274512202ULL, 60, 0 }, // 47 - h6, 5 bits -> 4 bits optimized

    { 0, 0x981ff28c8c8c8055ULL, 60, 0 }, // 48 - a7, 5 bits -> 4 bits optimized
    { 0, 0x8b0ffe8c94868049ULL, 60, 0 }, // 49 - b7, 5 bits -> 4 bits optimized
    { 0, 0x2101004202410000ULL, 59, 0 }, // 50
    { 0, 0x8200000041108022ULL, 59, 0 }, // 51
    { 0, 0x0000021082088000ULL, 59, 0 }, // 52
    { 0, 0x0002410204010040ULL, 59, 0 }, // 53
    { 0, 0xa37f23f28450a84bULL, 60, 0 }, // 54 - g7, 5 bits -> 4 bits optimized
    { 0, 0x00ff86a2e2e06100ULL, 60, 0 }, // 55 - h7, 5 bits -> 4 bits optimized

    { 0, 0x1613ff34100b0540ULL, 59, 0 }, // 56 - a8, 6 bits -> 5 bits optimized
    { 0, 0x68190bfe9809c511ULL, 60, 0 }, // 57 - b8, 5 bits -> 4 bits optimized
    { 0, 0x0402814422015008ULL, 59, 0 }, // 58
    { 0, 0x0090014004842410ULL, 59, 0 }, // 59
    { 0, 0x0001000042304105ULL, 59, 0 }, // 60
    { 0, 0x0010008830412a00ULL, 59, 0 }, // 61
    { 0, 0x9072ffa33264c2c4ULL, 60, 0 }, // 62 - g8, 5 bits -> 4 bits optimized
    { 0, 0x10ffa630030ca639ULL, 59, 0 }, // 63 - h8, 6 bits -> 5 bits optimized
};

static constexpr uint32_t cRookAttackTableSize = 84480;
static constexpr uint32_t cBishopAttackTableSize = 4800;

static Bitboard gRookAttackTable[cRookAttackTableSize];
static Bitboard gBishopAttackTable[cBishopAttackTableSize];

#endif // USE_PEXT_ATTACKS

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
Bitboard Bitboard::GetPawnAttacks<White>(const Square square)
{
    Bitboard bitboard;
    bitboard = (square.GetBitboard() & ~Bitboard::FileBitboard<0u>()) << 7u;
    bitboard |= (square.GetBitboard() & ~Bitboard::FileBitboard<7u>()) << 9u;
    return bitboard;
}

template<>
Bitboard Bitboard::GetPawnAttacks<Black>(const Square square)
{
    Bitboard bitboard;
    bitboard = (square.GetBitboard() & ~Bitboard::FileBitboard<0u>()) >> 9u;
    bitboard |= (square.GetBitboard() & ~Bitboard::FileBitboard<7u>()) >> 7u;
    return bitboard;
}

template<>
Bitboard Bitboard::GetPawnsAttacks<White>(const Bitboard pawns)
{
    Bitboard bitboard;
    bitboard = (pawns & ~Bitboard::FileBitboard<0u>()) << 7u;
    bitboard |= (pawns & ~Bitboard::FileBitboard<7u>()) << 9u;
    return bitboard;
}

template<>
Bitboard Bitboard::GetPawnsAttacks<Black>(const Bitboard pawns)
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
#ifdef USE_PEXT_ATTACKS
    const AttackData& data = gRookAttacksData[square.Index()];
    const uint32_t index = static_cast<uint32_t>(ParallelBitsExtract(blockers, data.mask));
    return gRookAttackTable[data.offset + index];
#else
    const MagicData& md = gRookMagicData[square.Index()];
    const uint32_t index = static_cast<uint32_t>(((blockers.value & md.mask) * md.magic) >> md.shift);
    return gRookAttackTable[md.offset + index];
#endif // USE_PEXT_ATTACKS
}

Bitboard Bitboard::GenerateBishopAttacks(const Square square, const Bitboard blockers)
{
#ifdef USE_PEXT_ATTACKS
    const AttackData& data = gBishopAttacksData[square.Index()];
    const uint32_t index = static_cast<uint32_t>(ParallelBitsExtract(blockers, data.mask));
    return gBishopAttackTable[data.offset + index];
#else
    const MagicData& md = gBishopMagicData[square.Index()];
    const uint32_t index = static_cast<uint32_t>(((blockers.value & md.mask) * md.magic) >> md.shift);
    return gBishopAttackTable[md.offset + index];
#endif // USE_PEXT_ATTACKS
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

        gPawnAttacksBitboard[squareIndex][0] = Bitboard::GetPawnAttacks<White>(square);
        gPawnAttacksBitboard[squareIndex][1] = Bitboard::GetPawnAttacks<Black>(square);
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

static void InitRookAttacks()
{
    uint32_t tableSize = 0;

    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        gRookAttacksBitboard[squareIndex] =
            Bitboard::RankBitboard(square.Rank()) |
            Bitboard::FileBitboard(square.File());

        const Bitboard attackMask = GetRookAttackMask(square);

        // compute number of possible occluder layouts
        const uint32_t attackMaskBits = PopCount(attackMask);
        const uint32_t numBlockerSets = 1 << attackMaskBits;

#ifdef USE_PEXT_ATTACKS

        gRookAttacksData[squareIndex].mask = attackMask;
        gRookAttacksData[squareIndex].offset = tableSize;

        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            // reconstruct (masked) blockers bitboard
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex), attackMask);
            gRookAttackTable[tableSize + blockersIndex] = Bitboard::GenerateRookAttacks_Slow(square, blockerBitboard);
        }

        tableSize += numBlockerSets;

#else // !USE_PEXT_ATTACKS

        gRookMagicData[squareIndex].mask   = attackMask;
        gRookMagicData[squareIndex].offset = tableSize;
        const uint64_t magic = gRookMagicData[squareIndex].magic;
        const uint32_t shift = gRookMagicData[squareIndex].shift;
        const uint32_t squareTableSize = 1u << (64u - shift);

        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            // reconstruct (masked) blockers bitboard
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);

            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            ASSERT(tableIndex < squareTableSize);
            gRookAttackTable[tableSize + tableIndex] = Bitboard::GenerateRookAttacks_Slow(square, blockerBitboard);
        }

#ifndef CONFIGURATION_FINAL
        // validate (there must be no collisions)
        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);
            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            const Bitboard expected = Bitboard::GenerateRookAttacks_Slow(square, blockerBitboard);
            ASSERT(gRookAttackTable[tableSize + tableIndex] == expected);
            ASSERT(Bitboard::GenerateRookAttacks(square, blockerBitboard) == expected);
        }
#endif // CONFIGURATION_FINAL

        tableSize += squareTableSize;

#endif // USE_PEXT_ATTACKS
    }

#ifndef CONFIGURATION_FINAL
    std::cout << "Rook attack table size: " << tableSize << " entries (" << (tableSize * sizeof(Bitboard)) / 1024 << " KB)" << std::endl;
#endif // CONFIGURATION_FINAL

    ASSERT(tableSize == cRookAttackTableSize);
}

static void InitBishopAttacks()
{
    uint32_t tableSize = 0;

    for (uint32_t squareIndex = 0; squareIndex < Square::NumSquares; ++squareIndex)
    {
        const Square square(squareIndex);

        gBishopAttacksBitboard[squareIndex] =
            Bitboard::GetRay(square, Direction::NorthEast) |
            Bitboard::GetRay(square, Direction::NorthWest) |
            Bitboard::GetRay(square, Direction::SouthEast) |
            Bitboard::GetRay(square, Direction::SouthWest);

        const Bitboard attackMask = GetBishopAttackMask(square);

        // compute number of possible occluder layouts
        const uint32_t attackMaskBits = PopCount(attackMask);
        const uint32_t numBlockerSets = 1 << attackMaskBits;

#ifdef USE_PEXT_ATTACKS

        gBishopAttacksData[squareIndex].mask = attackMask;
        gBishopAttacksData[squareIndex].offset = tableSize;

        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            // reconstruct (masked) blockers bitboard
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex), attackMask);
            gBishopAttackTable[tableSize + blockersIndex] = Bitboard::GenerateBishopAttacks_Slow(square, blockerBitboard);
        }

        tableSize += numBlockerSets;

#else // !USE_PEXT_ATTACKS

        gBishopMagicData[squareIndex].mask   = attackMask;
        gBishopMagicData[squareIndex].offset = tableSize;
        const uint64_t magic = gBishopMagicData[squareIndex].magic;
        const uint32_t shift = gBishopMagicData[squareIndex].shift;
        const uint32_t squareTableSize = 1u << (64u - shift);

        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            // reconstruct (masked) blockers bitboard
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);

            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            ASSERT(tableIndex < squareTableSize);
            gBishopAttackTable[tableSize + tableIndex] = Bitboard::GenerateBishopAttacks_Slow(square, blockerBitboard);
        }

#ifndef CONFIGURATION_FINAL
        // validate (there must be no collisions)
        for (uint32_t blockersIndex = 0; blockersIndex < numBlockerSets; ++blockersIndex)
        {
            const Bitboard blockerBitboard = ParallelBitsDeposit(static_cast<uint64_t>(blockersIndex + 1u), attackMask);
            const uint32_t tableIndex = static_cast<uint32_t>((blockerBitboard * magic) >> shift);
            const Bitboard expected = Bitboard::GenerateBishopAttacks_Slow(square, blockerBitboard);
            ASSERT(gBishopAttackTable[tableSize + tableIndex] == expected);
            ASSERT(Bitboard::GenerateBishopAttacks(square, blockerBitboard) == expected);
        }
#endif // CONFIGURATION_FINAL

        tableSize += squareTableSize;

#endif // USE_PEXT_ATTACKS
    }

#ifndef CONFIGURATION_FINAL
    std::cout << "Bishop attack table size: " << tableSize << " entries (" << (tableSize * sizeof(Bitboard)) / 1024 << " KB)" << std::endl;
#endif // CONFIGURATION_FINAL

    ASSERT(tableSize == cBishopAttackTableSize);
}

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
    InitBetweenBitboards();
}
