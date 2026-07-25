#include "Common.hpp"

#include "../backend/Position.hpp"
#include "../backend/PositionUtils.hpp"
#include "../backend/Game.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Material.hpp"

#include <iostream>
#include <random>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>
#include <string>
#include <vector>

// Generates random, quiet, near-equal chess positions and writes them as a plain-FEN opening book
// (one FEN per line, directly loadable by the selfplay tool). Positions are filtered by a fixed-depth
// search so that only positions with |eval| within a configured centipawn band are kept.

struct RandomPositionsConfig
{
    std::string outputPath      = "random_positions.epd";
    uint32_t count              = 1000000;  // target number of accepted positions
    uint32_t minPieces          = 8;        // min total pieces including both kings
    uint32_t maxPieces          = 24;       // max total pieces including both kings
    uint32_t depth              = 16;       // fixed search depth for the eval filter
    int32_t minEval             = 60;       // min |eval| in display centipawns
    int32_t maxEval             = 150;      // max |eval| in display centipawns
    int32_t prefilterMargin     = 50;       // widening (cp) of the eval band for the shallow pre-search
    uint32_t maxPawnsPerFile    = 3;        // per-side pawn-per-file cap
    int32_t maxMaterialDiff     = 5;        // cap on |white material value - black| (P=1,N=B=3,R=5,Q=9)
    uint32_t numThreads         = 0;        // 0 => hardware_concurrency
    uint32_t hashSizeMB         = 4;        // per-thread transposition table size
};

static bool IsValueToken(const std::string& s)
{
    if (s.empty()) return false;
    if (isdigit(static_cast<unsigned char>(s[0]))) return true;
    return (s[0] == '-' || s[0] == '.') && s.size() > 1 && isdigit(static_cast<unsigned char>(s[1]));
}

static RandomPositionsConfig ParseArgs(const std::vector<std::string>& args)
{
    RandomPositionsConfig config;

    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string& arg = args[i];
        if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-')
        {
            const std::string flag = arg.substr(2);
            const bool hasValue = (i + 1 < args.size()) && IsValueToken(args[i + 1]);
            const auto nextUInt = [&]() -> uint32_t { return hasValue ? std::stoul(args[++i]) : 0u; };
            const auto nextInt  = [&]() -> int32_t  { return hasValue ? std::stol(args[++i]) : 0; };

            if      (flag == "count")           config.count            = nextUInt();
            else if (flag == "minPieces")       config.minPieces        = nextUInt();
            else if (flag == "maxPieces")       config.maxPieces        = nextUInt();
            else if (flag == "depth")           config.depth            = nextUInt();
            else if (flag == "minEval")         config.minEval          = nextInt();
            else if (flag == "maxEval")         config.maxEval          = nextInt();
            else if (flag == "prefilterMargin") config.prefilterMargin  = nextInt();
            else if (flag == "maxPawnsPerFile") config.maxPawnsPerFile  = nextUInt();
            else if (flag == "maxMaterialDiff") config.maxMaterialDiff  = nextInt();
            else if (flag == "threads")         config.numThreads       = nextUInt();
            else if (flag == "hash")            config.hashSizeMB       = nextUInt();
            else std::cerr << "Warning: unknown flag --" << flag << "\n";
        }
        else
        {
            config.outputPath = arg;
        }
    }

    return config;
}

// material value used both for the balance check and to reject empty sides
static int32_t MaterialValue(uint32_t pawns, uint32_t knights, uint32_t bishops, uint32_t rooks, uint32_t queens)
{
    return static_cast<int32_t>(pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens);
}

void GenerateRandomPositions(const std::vector<std::string>& args)
{
    const RandomPositionsConfig config = ParseArgs(args);

    if (config.minPieces < 3 || config.maxPieces > 32 || config.minPieces > config.maxPieces)
    {
        std::cerr << "Invalid piece range: " << config.minPieces << ".." << config.maxPieces << "\n";
        return;
    }

    std::ofstream outputFile(config.outputPath);
    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open output file: " << config.outputPath << "\n";
        return;
    }

    const uint32_t numThreads = config.numThreads > 0
        ? config.numThreads
        : std::max<uint32_t>(1, std::thread::hardware_concurrency() - 1);

    std::cout << "Generating " << config.count << " random positions\n"
              << "  pieces:   " << config.minPieces << ".." << config.maxPieces << "\n"
              << "  depth:    " << config.depth << "\n"
              << "  eval band: [" << config.minEval << ", " << config.maxEval << "] cp\n"
              << "  threads:  " << numThreads << "\n"
              << "  output:   " << config.outputPath << "\n";

    std::mutex mutex;
    std::atomic<uint32_t> numPositions{ 0 };

    const auto generate = [&]()
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<uint32_t> pieceCountDistr(config.minPieces, config.maxPieces);
        // weight each piece type by its maximum allowed count so pawns dominate as they do on a full
        // board (~8 of every ~15 non-king units), keeping realistic pawn counts and few over-draws
        // order: WP, WN, WB, WR, WQ, BP, BN, BB, BR, BQ
        std::discrete_distribution<uint32_t> pieceIndexDistr({ 8, 2, 2, 2, 1, 8, 2, 2, 2, 1 });

        Search search;
        TranspositionTable tt{ static_cast<size_t>(config.hashSizeMB) * 1024 * 1024 };

        while (numPositions.load(std::memory_order_relaxed) < config.count)
        {
            // randomize material
            MaterialKey matKey;
            const uint32_t numPieces = pieceCountDistr(gen);
            for (uint32_t j = 0; j < numPieces - 2; ++j)
            {
                switch (pieceIndexDistr(gen))
                {
                case 0: matKey.numWhitePawns++; break;
                case 1: matKey.numWhiteKnights++; break;
                case 2: matKey.numWhiteBishops++; break;
                case 3: matKey.numWhiteRooks++; break;
                case 4: matKey.numWhiteQueens++; break;
                case 5: matKey.numBlackPawns++; break;
                case 6: matKey.numBlackKnights++; break;
                case 7: matKey.numBlackBishops++; break;
                case 8: matKey.numBlackRooks++; break;
                case 9: matKey.numBlackQueens++; break;
                }
            }

            // reject illegal piece counts
            if (matKey.numWhitePawns > 8 || matKey.numBlackPawns > 8 ||
                matKey.numWhiteKnights > 2 || matKey.numBlackKnights > 2 ||
                matKey.numWhiteBishops > 2 || matKey.numBlackBishops > 2 ||
                matKey.numWhiteRooks > 2 || matKey.numBlackRooks > 2 ||
                matKey.numWhiteQueens > 1 || matKey.numBlackQueens > 1)
                continue;

            // keep sides roughly balanced and non-empty
            const int32_t whiteValue = MaterialValue((uint32_t)matKey.numWhitePawns, (uint32_t)matKey.numWhiteKnights, (uint32_t)matKey.numWhiteBishops, (uint32_t)matKey.numWhiteRooks, (uint32_t)matKey.numWhiteQueens);
            const int32_t blackValue = MaterialValue((uint32_t)matKey.numBlackPawns, (uint32_t)matKey.numBlackKnights, (uint32_t)matKey.numBlackBishops, (uint32_t)matKey.numBlackRooks, (uint32_t)matKey.numBlackQueens);
            if (whiteValue == 0 || blackValue == 0)
                continue;
            if (std::abs(whiteValue - blackValue) > config.maxMaterialDiff)
                continue;

            // generate board (kings non-adjacent, pawns off back ranks, white to move, black not in check)
            // keep pawns off the pre-promotion ranks so no side has a promotion in one move
            RandomPosDesc desc{ matKey };
            desc.allowedWhitePawns = ~Bitboard::RankBitboard<6>();
            desc.allowedBlackPawns = ~Bitboard::RankBitboard<1>();

            Position pos;
            GenerateRandomPosition(gen, desc, pos);

            // reject two bishops on the same color square per side
            if ((pos.Whites().bishops & Bitboard::LightSquares()).Count() > 1 ||
                (pos.Whites().bishops & Bitboard::DarkSquares()).Count() > 1 ||
                (pos.Blacks().bishops & Bitboard::LightSquares()).Count() > 1 ||
                (pos.Blacks().bishops & Bitboard::DarkSquares()).Count() > 1)
                continue;

            // reject unreasonable pawn structures (too many pawns on a single file)
            {
                bool badPawns = false;
                for (uint32_t file = 0; file < 8 && !badPawns; ++file)
                {
                    const Bitboard fileMask = Bitboard::FileBitboard(file);
                    if ((pos.Whites().pawns & fileMask).Count() > config.maxPawnsPerFile ||
                        (pos.Blacks().pawns & fileMask).Count() > config.maxPawnsPerFile)
                        badPawns = true;
                }
                if (badPawns)
                    continue;
            }

            // only legal, quiet positions (not in check, no winning capture available), not mate/stalemate
            if (!pos.IsValid() || !pos.IsQuiet() || pos.IsMate() || pos.IsStalemate())
                continue;

            // coarse static pre-filter to avoid running a full search on hopeless positions
            const int32_t staticEval = NormalizeEval(Evaluate(pos));
            if (std::abs(staticEval) > 4 * config.maxEval)
                continue;

            // search filter: keep only positions on the edge
            Game game;
            game.Reset(pos);

            const auto searchToDepth = [&](uint32_t depth) -> int32_t
            {
                tt.NextGeneration();

                SearchParam searchParam{ tt };
                searchParam.debugLog = false;
                searchParam.useRootTablebase = false;
                searchParam.limits.maxDepth = static_cast<uint16_t>(depth);

                SearchResult searchResult;
                search.DoSearch(game, searchParam, searchResult);

                if (searchResult.empty())
                    return InvalidValue;

                return NormalizeEval(searchResult.front().score);
            };

            tt.Clear();

            // cheap shallow pre-search with wider bounds; skip early if clearly off the band
            // (the full search reuses the warmed TT, so no clear in between)
            const int32_t preEval = searchToDepth(std::max<uint32_t>(1, config.depth / 2));
            if (preEval == InvalidValue)
                continue;
            if (std::abs(preEval) < config.minEval - config.prefilterMargin ||
                std::abs(preEval) > config.maxEval + config.prefilterMargin)
                continue;

            // full-depth search against the target bounds
            const int32_t evalCp = searchToDepth(config.depth);
            if (evalCp == InvalidValue)
                continue;
            if (std::abs(evalCp) < config.minEval || std::abs(evalCp) > config.maxEval)
                continue;

            {
                std::lock_guard<std::mutex> lock(mutex);
                if (numPositions.load(std::memory_order_relaxed) >= config.count)
                    return;

                outputFile << pos.ToFEN() << "\n";

                const uint32_t written = numPositions.fetch_add(1, std::memory_order_relaxed) + 1;
                if (written % 10 == 0)
                    std::cout << "Generated " << written << " positions" << std::endl;
            }
        }
    };

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < numThreads; ++i)
        threads.emplace_back(generate);
    for (auto& thread : threads)
        thread.join();

    outputFile.flush();
    std::cout << "Done. Wrote " << numPositions.load() << " positions to " << config.outputPath << "\n";
}
