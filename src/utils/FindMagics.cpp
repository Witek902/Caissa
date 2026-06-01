#include "Common.hpp"
#include "../backend/Bitboard.hpp"
#include "../backend/Square.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// 0 - find magics for bishops
// 1 - find magics for rooks
#define SEARCH_FOR_ROOK_MAGICS 1

// 0 - search for regular (fancy) magics: (occ & mask) * magic >> shift
// 1 - search for black magics:           (occ | ~mask) * magic >> shift
#define USE_BLACK_MAGIC 0

// Which square to search for
static constexpr uint32_t kSearchSquare = 13;

#if SEARCH_FOR_ROOK_MAGICS
    static constexpr const char* kPieceStr  = "rook";
    static constexpr char        kPieceChar = 'r';
#else
    static constexpr const char* kPieceStr  = "bishop";
    static constexpr char        kPieceChar = 'b';
#endif

// Maximum table size we'll ever try: 2^13 = 8192 entries.
static constexpr uint32_t kMaxTableBits = 13;
static constexpr uint32_t kMaxTableSize = 1u << kMaxTableBits;

// Attack mask computation
static uint64_t ComputeRookMask(Square sq)
{
    ASSERT(sq.IsValid());
    uint64_t mask = 0;
    mask |= (uint64_t)(Bitboard::RankBitboard(sq.Rank()) & ~Bitboard::FileBitboard<0>() & ~Bitboard::FileBitboard<7>());
    mask |= (uint64_t)(Bitboard::FileBitboard(sq.File()) & ~Bitboard::RankBitboard<0>() & ~Bitboard::RankBitboard<7>());
    mask &= ~(uint64_t)sq.GetBitboard();
    ASSERT((mask & sq.GetBitboard()) == 0);
    ASSERT((mask & (uint64_t)(Bitboard::FileBitboard<0>() | Bitboard::FileBitboard<7>() | Bitboard::RankBitboard<0>() | Bitboard::RankBitboard<7>())) == 0);
    return mask;
}

static uint64_t ComputeBishopMask(Square sq)
{
    ASSERT(sq.IsValid());
    uint64_t mask = 0;
    mask |= (uint64_t)Bitboard::GetRay(sq, Direction::NorthEast);
    mask |= (uint64_t)Bitboard::GetRay(sq, Direction::NorthWest);
    mask |= (uint64_t)Bitboard::GetRay(sq, Direction::SouthEast);
    mask |= (uint64_t)Bitboard::GetRay(sq, Direction::SouthWest);
    mask &= ~(uint64_t)sq.GetBitboard();
    mask &= ~(uint64_t)(Bitboard::FileBitboard<0>() | Bitboard::FileBitboard<7>() | Bitboard::RankBitboard<0>() | Bitboard::RankBitboard<7>());
    ASSERT((mask & sq.GetBitboard()) == 0);
    ASSERT((mask & (uint64_t)(Bitboard::FileBitboard<0>() | Bitboard::FileBitboard<7>() | Bitboard::RankBitboard<0>() | Bitboard::RankBitboard<7>())) == 0);
    return mask;
}

// Per-square precomputed data
struct SquareData
{
    uint64_t mask;
    uint64_t notMask; // ~mask, used as black magic key: (occ | notMask)
    uint32_t maskBits;
    std::vector<uint64_t> blockers; // magic keys: (occ & mask) for fancy, (occ | notMask) for black
    std::vector<uint64_t> attacks;  // attacks for each occupancy subset
};

static SquareData gSquareData;

static void PrecomputeSquareData()
{
    const Square square(kSearchSquare);
    SquareData& sd = gSquareData;

#if SEARCH_FOR_ROOK_MAGICS
    sd.mask = ComputeRookMask(square);
#else
    sd.mask = ComputeBishopMask(square);
#endif
    sd.maskBits = PopCount(sd.mask);
    ASSERT(sd.maskBits >= 1 && sd.maskBits <= kMaxTableBits);
    sd.notMask = ~sd.mask;

    const uint32_t numSubsets = 1u << sd.maskBits;
    sd.blockers.resize(numSubsets);
    sd.attacks.resize(numSubsets);

    uint32_t i = 0;
    uint64_t subset = 0;
    do {
        ASSERT((subset & ~sd.mask) == 0);

#if USE_BLACK_MAGIC
        sd.blockers[i] = subset | sd.notMask;
#else
        sd.blockers[i] = subset;
#endif

#if SEARCH_FOR_ROOK_MAGICS
        sd.attacks[i] = (uint64_t)Bitboard::GenerateRookAttacks_Slow(square, subset);
#else
        sd.attacks[i] = (uint64_t)Bitboard::GenerateBishopAttacks_Slow(square, subset);
#endif

        ++i;
        subset = (subset - sd.mask) & sd.mask;
    } while (subset != 0);

    ASSERT(i == numSubsets);
    ASSERT(sd.blockers.size() == numSubsets);
    ASSERT(sd.attacks.size() == numSubsets);
}

// Global search state

// Packed as (bits << 32) | used — lexicographic comparison naturally implements
// "minimize bits first, then minimize used entries within the same bit count".
static std::atomic<uint64_t> gOnesMask{ 0xffffffffffffffffULL };
static std::atomic<uint64_t> gZerosMask{ 0xffffffffffffffffULL };
static std::atomic<uint64_t> gBestState{ 0 };
static std::mutex gOutputMutex;

static constexpr uint64_t PackState(uint32_t bits, uint32_t used) { return (uint64_t(bits) << 32) | used; }
static constexpr uint32_t StateBits(uint64_t s) { return uint32_t(s >> 32); }
static constexpr uint32_t StateUsed(uint64_t s) { return uint32_t(s); }

// Count distinct occupied slots without allocating (generation-based, zero-copy).
static uint32_t CountUsed(const SquareData& sd, uint64_t magic, uint32_t bits)
{
    thread_local uint32_t tblGen[kMaxTableSize] = {};
    thread_local uint32_t gen = 0;
    if (++gen == 0) { std::fill(std::begin(tblGen), std::end(tblGen), 0u); gen = 1; }

    const uint32_t shift = 64u - bits;
    uint32_t count = 0;
    for (const uint64_t blocker : sd.blockers)
    {
        const uint32_t idx = uint32_t(blocker * magic >> shift);
        if (tblGen[idx] != gen) { tblGen[idx] = gen; ++count; }
    }
    return count;
}

// Magic validation (generation-based, zero clearing overhead)
static bool TryMagic(const SquareData& sd, uint64_t magic, uint32_t bits)
{
    ASSERT(bits >= 1 && bits <= kMaxTableBits);
    ASSERT(!sd.blockers.empty());
    ASSERT(sd.blockers.size() == sd.attacks.size());
    ASSERT(sd.blockers.size() == (size_t(1) << sd.maskBits));

    thread_local uint32_t tblGen[kMaxTableSize] = {};
    thread_local uint64_t tblAtk[kMaxTableSize] = {};
    thread_local uint32_t gen = 0;

    if (++gen == 0)
    {
        std::fill(std::begin(tblGen), std::end(tblGen), 0u);
        gen = 1;
    }

    const uint32_t shift = 64u - bits;
    for (uint32_t i = 0; i < (uint32_t)sd.blockers.size(); ++i)
    {
        const uint32_t idx = (uint32_t)((sd.blockers[i] * magic) >> shift);
        ASSERT(idx < (1u << bits));
        if (tblGen[idx] != gen)
        {
            tblGen[idx] = gen;
            tblAtk[idx] = sd.attacks[i];
        }
        else if (tblAtk[idx] != sd.attacks[i])
        {
            return false;
        }
    }
    return true;
}

// Result logging
static void LogAndSaveResult(uint64_t magic, uint32_t bits)
{
    ASSERT(bits >= 1 && bits <= kMaxTableBits);
    ASSERT(magic != 0);

    const SquareData& sd = gSquareData;
    const uint32_t shift     = 64u - bits;
    const uint32_t tableSize = 1u << bits;

    std::vector<uint64_t> table(tableSize, 0);
    std::vector<bool> occupied(tableSize, false);
    for (uint32_t i = 0; i < (uint32_t)sd.blockers.size(); ++i)
    {
        const uint32_t idx = (uint32_t)((sd.blockers[i] * magic) >> shift);
        ASSERT(idx < tableSize);
        ASSERT(!occupied[idx] || table[idx] == sd.attacks[i]);
        table[idx] = sd.attacks[i];
        occupied[idx] = true;
    }
    const uint32_t entriesUsed = (uint32_t)std::count(occupied.begin(), occupied.end(), true);
    ASSERT(entriesUsed >= 1 && entriesUsed <= tableSize);

    {
        std::lock_guard<std::mutex> lk(gOutputMutex);
        std::cout
            << "[" << kPieceStr << " sq=" << kSearchSquare
            << " mask_bits=" << sd.maskBits << "]"
            << "  bits=" << bits
            << "  magic=0x" << std::hex << std::setw(16) << std::setfill('0') << magic
            << "  1mask=0x" << std::hex << std::setw(16) << std::setfill('0') << gOnesMask.load(std::memory_order_relaxed)
            << "  0mask=0x" << std::hex << std::setw(16) << std::setfill('0') << gZerosMask.load(std::memory_order_relaxed) << std::dec
            << "  used=" << entriesUsed << "/" << tableSize << "\n";
    }

    const std::string filename = std::string("magic_") + kPieceChar
        + "_sq" + std::to_string(kSearchSquare)
        + "_bits" + std::to_string(bits) + ".txt";

    std::ofstream f(filename);
    if (!f) return;

    f << "piece=" << kPieceChar << "\n"
      << "square=" << kSearchSquare << "\n"
      << "bits=" << bits << "\n"
      << std::hex
      << "magic=0x"   << std::setw(16) << std::setfill('0') << magic << "\n"
      << "mask=0x"    << std::setw(16) << std::setfill('0') << sd.mask << "\n"
      << "notmask=0x" << std::setw(16) << std::setfill('0') << sd.notMask << "\n"
      << std::dec
      << "mask_bits=" << sd.maskBits << "\n"
      << "num_occupancies=" << sd.blockers.size() << "\n"
      << "total_entries=" << tableSize << "\n"
      << "used_entries=" << entriesUsed << "\n"
      << "unused_entries=" << (tableSize - entriesUsed) << "\n"
#if USE_BLACK_MAGIC
      << "formula: (occ | ~mask) * magic >> " << shift << "\n";
#else
      << "formula: (occ & mask) * magic >> " << shift << "\n";
#endif

    f << "\n# Attack table: index -> attacks\n";
    for (uint32_t i = 0; i < tableSize; ++i)
    {
        f << "T[" << std::setw(4) << std::setfill(' ') << i << "] = ";
        if (occupied[i])
            f << "0x" << std::hex << std::setw(16) << std::setfill('0') << table[i] << std::dec;
        else
            f << "<unused>";
        f << "\n";
    }

    f << "\n# Occupancy map: occupancy -> table_index -> attacks\n";
    for (uint32_t i = 0; i < (uint32_t)sd.blockers.size(); ++i)
    {
        const uint32_t idx = (uint32_t)((sd.blockers[i] * magic) >> shift);
        f << "occ=0x" << std::hex << std::setw(16) << std::setfill('0') << sd.blockers[i]
          << " -> T[" << std::dec << std::setw(4) << std::setfill(' ') << idx
          << "] = 0x" << std::hex << std::setw(16) << std::setfill('0') << sd.attacks[i]
          << "\n" << std::dec;
    }
}

// Worker thread
static void SearchWorker(uint32_t threadId)
{
    std::random_device device;
    std::seed_seq seq{ device() + threadId, device(), device(), device() };
    std::mt19937_64 rng(seq);

    const SquareData& sd = gSquareData;


    while (true)
    {
        const uint64_t curState  = gBestState.load(std::memory_order_relaxed);
        const uint32_t targetBits = StateBits(curState);
        if (targetBits < 1) continue;

        uint64_t candidate = rng();
        candidate &= ~0x00001e0000000000ULL;
        candidate |=  0x00000000000007f0ULL;


        // rook f2 (sq=13)
        //candidate &= ~0x00001ffc00000000ULL;
        //candidate |= 0x00000000000007ffULL;

        // rook b2 (sq=9)
        //candidate &= ~0x00000fe000000000ULL;
        //candidate |= 0x0000000000007ff0ULL;

        // rook g7 (sqr=54)
        //candidate &= ~0x1fULL;
        //candidate |= 0x0003ffc3c0000000ULL;

        // rook b7 (sqr=49)
        //candidate &= ~0x00000800f00001ffULL;
        //candidate |=  0x007ff77790000000ULL;

        // rook c7 (sq=50)
        //candidate &= ~0x000000000000001fULL;
        //candidate |= 0x003ffc0400000000ULL;

        // bishop c1 (sqr=2)
        //candidate |= 0x0000000003f00000ULL; // bits 20-25: always 1
        //candidate &= ~0x0003000000000000ULL; // bits 48-49: always 0

        // bishop c3 (sqr=18)
        //candidate |= 0x00000000000003ffULL; // bits 0-9:      always 1
        //candidate &= ~0x0000fe00e0000000ULL; // bits 29-31, 41-47: always 0

        if (!TryMagic(sd, candidate, targetBits))
            continue;

        // Also try one bit fewer — if it works, use that instead.
        const uint32_t achievedBits = (targetBits > 1 && TryMagic(sd, candidate, targetBits - 1))
            ? targetBits - 1 : targetBits;

        const uint32_t usedEntries = CountUsed(sd, candidate, achievedBits);
        const uint64_t newState = PackState(achievedBits, usedEntries);

        if (newState < curState)
        {
            gOnesMask = 0xffffffffffffffffULL;
            gZerosMask = 0xffffffffffffffffULL;
        }

        if (newState <= curState)
        {
            gOnesMask &= candidate;
            gZerosMask &= ~candidate;
            LogAndSaveResult(candidate, achievedBits);
        }

        uint64_t cur = curState;
        while (newState < cur)
        {
            if (gBestState.compare_exchange_weak(cur, newState, std::memory_order_relaxed))
            {
                //LogAndSaveResult(candidate, achievedBits);
                break;
            }
        }
    }
}

// Entry point for utility function to find Black Magic numbers for rook and bishop move generation.
// Black magic: uses (occ | ~mask) * magic >> shift instead of (occ & mask) * magic >> shift.
// https://www.chessprogramming.org/Magic_Bitboards#Black_Magic_Bitboards
void FindMagics()
{
    std::cout << "FindMagics: searching for " << kPieceStr
#if USE_BLACK_MAGIC
              << " BLACK magics"
#else
              << " FANCY magics"
#endif
              << " on sq=" << kSearchSquare << "\n";

    PrecomputeSquareData();

    ASSERT(gSquareData.maskBits >= 1 && gSquareData.maskBits <= kMaxTableBits);
    ASSERT(gSquareData.blockers.size() == (size_t(1) << gSquareData.maskBits));

    // Start at (maskBits, UINT32_MAX) — first magic at maskBits bits will always improve it.
    gBestState.store(PackState(gSquareData.maskBits, UINT32_MAX), std::memory_order_relaxed);

    std::cout << "  mask_bits=" << gSquareData.maskBits
              << "  starting at bits=" << gSquareData.maskBits << "\n";

    const uint32_t numThreads = std::max(1u, std::thread::hardware_concurrency() - 1);
    std::cout << "  using " << numThreads << " threads\n\n";

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (uint32_t t = 0; t < numThreads; ++t)
        threads.emplace_back(SearchWorker, t);

    for (auto& t : threads)
        t.join();
}
