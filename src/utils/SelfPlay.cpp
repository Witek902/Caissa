#include "Common.hpp"
#include "GameCollection.hpp"

#include "../backend/Position.hpp"
#include "../backend/Game.hpp"
#include "../backend/Score.hpp"
#include "../backend/Move.hpp"
#include "../backend/Search.hpp"
#include "../backend/TranspositionTable.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Tablebase.hpp"

#include <random>
#include <mutex>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

struct SelfPlayConfig
{
    std::vector<std::string> bookPaths;

    uint32_t minNodes                   = 21'000;
    uint32_t maxNodes                   = 21'000;
    uint32_t maxDepth                   = 40;
    int32_t  maxEval                    = 2000;
    int32_t  openingMaxEval             = 500;
    uint32_t minRandomMoves             = 8;
    uint32_t maxRandomMoves             = 10;
    int32_t  drawScoreThreshold         = 3;
    uint32_t drawScoreConsecutiveMoves  = 10;
    uint32_t drawMinHalfMove            = 80;
    uint32_t winAdjMinHalfMove          = 40;
    uint32_t winAdjConsecutiveMoves     = 3;
    uint32_t syzygyProbeLimit           = 5;
    uint32_t consolePgnFrequency        = 1;
    bool     dumpAllPgn                 = false;
    uint32_t numThreads                 = 0; // 0 = hardware_concurrency
};

static bool LoadOpeningPositions(const std::string& path, std::vector<PackedPosition>& outPositions)
{
    std::ifstream file(path);
    if (!file.good())
    {
        std::cout << "Failed to load opening positions file " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        Position pos;
        if (!pos.FromFEN(line))
        {
            std::cout << "Invalid FEN string: " << line << "\n";
            continue;
        }

        if (pos.GetNumPieces() > 32)
        {
            std::cout << "Too many pieces: " << line << "\n";
            continue;
        }

        PackedPosition packedPos;
        PackPosition(pos, packedPos);
        outPositions.push_back(packedPos);
    }

    return true;
}

static Move GetRandomMove(std::mt19937& randomGenerator, const Position& pos)
{
    std::vector<Move> moves;
    pos.GetNumLegalMoves(&moves);

    // don't play losing moves (according to SEE)
    moves.erase(std::remove_if(moves.begin(),
        moves.end(),
        [&](const Move& move) { return !pos.StaticExchangeEvaluation(move); }),
        moves.end());

    if (moves.empty())
        return Move::Invalid();

    Move move = moves.front();
    if (moves.size() > 1)
    {
        std::uniform_int_distribution<size_t> distr(0, moves.size() - 1);
        move = moves[distr(randomGenerator)];
    }

    return move;
}

struct SelfPlayStats
{
    std::atomic<uint32_t> numWhiteWins = 0;
    std::atomic<uint32_t> numBlackWins = 0;
    std::atomic<uint32_t> numDraws = 0;
};

static bool SelfPlayThreadFunc(
    uint32_t threadIndex,
    const SelfPlayConfig& config,
    const std::vector<PackedPosition>& openingPositions,
    std::atomic<uint32_t>& openingCounter,
    std::atomic<uint32_t>& gameCounter,
    GameCollection::Writer& writer,
    std::ofstream* pgnFile,
    std::mutex& pgnMutex,
    SelfPlayStats& stats)
{
    const size_t c_transpositionTableSize = 4ull * 1024ull * 1024ull;

    std::random_device rd;
    std::mt19937 gen(rd());

    Search search;
    TranspositionTable tt{ c_transpositionTableSize };

    for (;;)
    {
        SearchResult searchResult;

        // generate opening position
        Position openingPos;

        const uint32_t index = gameCounter.fetch_add(1, std::memory_order_relaxed);

        if (!openingPositions.empty())
        {
            const uint32_t openingIndex = openingCounter.fetch_add(1, std::memory_order_relaxed) % (uint32_t)openingPositions.size();
            UnpackPosition(openingPositions[openingIndex], openingPos);
        }

        if (config.maxRandomMoves > 0)
        {
            // play few random moves in the opening
            const uint32_t numRandomMoves = std::uniform_int_distribution<uint32_t>(config.minRandomMoves, config.maxRandomMoves)(gen);
            for (uint32_t i = 0; i < numRandomMoves; ++i)
            {
                Move move = GetRandomMove(gen, openingPos);
                if (!move.IsValid())
                    break;

                const bool moveSuccess = openingPos.DoMove(move);
                ASSERT(moveSuccess);
                (void)moveSuccess;
            }
        }

        if (openingPos.IsMate() || openingPos.IsStalemate())
            continue;

        // start new game
        Game game;
        tt.Clear();
        search.Clear();
        game.Reset(openingPos);

        int32_t halfMoveNumber = 0;
        uint32_t drawScoreCounter = 0;
        uint32_t whiteWinsCounter = 0;
        uint32_t blackWinsCounter = 0;

        const uint32_t searchSeed = gen();

        for (;; ++halfMoveNumber)
        {
            SearchParam searchParam{ tt };
            searchParam.debugLog = false;
            searchParam.useRootTablebase = false;
            searchParam.evalRandomization = 1;
            searchParam.seed = searchSeed;
            searchParam.limits.maxDepth = static_cast<uint16_t>(config.maxDepth);
            searchParam.limits.maxNodesSoft = config.minNodes + (config.maxNodes - config.minNodes) * std::max(0, 80 - halfMoveNumber) / 80;
            if (halfMoveNumber < 10) searchParam.limits.maxNodesSoft *= 2; // more nodes in the first moves
            searchParam.limits.maxNodes = 5 * searchParam.limits.maxNodesSoft;

            searchResult.clear();
            tt.NextGeneration();
            search.DoSearch(game, searchParam, searchResult);

            ASSERT(!searchResult.empty());

            // skip game if starting position is unbalanced
            if (halfMoveNumber == 0 && std::abs(searchResult.begin()->score) * 100 / wld::NormalizeToPawnValue > config.openingMaxEval)
                break;

            ASSERT(!searchResult.front().moves.empty());
            Move move = searchResult.front().moves.front();

            ScoreType moveScore = searchResult.front().score;
            ScoreType eval = Evaluate(game.GetPosition());

            if (game.GetSideToMove() == Black)
            {
                moveScore = -moveScore;
                eval = -eval;
            }

            const bool moveSuccess = game.DoMove(move, moveScore);
            ASSERT(moveSuccess);
            (void)moveSuccess;

            if (std::abs(moveScore) < config.drawScoreThreshold)
                drawScoreCounter++;
            else
                drawScoreCounter = 0;

            // adjudicate draw if eval is near-zero for long enough
            if (drawScoreCounter > config.drawScoreConsecutiveMoves && halfMoveNumber >= (int32_t)config.drawMinHalfMove)
            {
                game.SetScore(Game::Score::Draw);
            }

            // adjudicate win
            if (halfMoveNumber >= (int32_t)config.winAdjMinHalfMove)
            {
                if (moveScore > config.maxEval && eval > 0)
                {
                    whiteWinsCounter++;
                    if (whiteWinsCounter > config.winAdjConsecutiveMoves) game.SetScore(Game::Score::WhiteWins);
                }
                else
                {
                    whiteWinsCounter = 0;
                }

                if (moveScore < -config.maxEval && eval < 0)
                {
                    blackWinsCounter++;
                    if (blackWinsCounter > config.winAdjConsecutiveMoves) game.SetScore(Game::Score::BlackWins);
                }
                else
                {
                    blackWinsCounter = 0;
                }
            }

            const bool isCheck = game.GetPosition().IsInCheck();

            // tablebase adjudication
            int32_t wdlScore = 0;
            if (!isCheck && ProbeSyzygy_WDL(game.GetPosition(), &wdlScore))
            {
                const auto stm = game.GetPosition().GetSideToMove();
                if (wdlScore == 1) game.SetScore(stm == White ? Game::Score::WhiteWins : Game::Score::BlackWins);
                if (wdlScore == 0) game.SetScore(Game::Score::Draw);
                if (wdlScore == -1) game.SetScore(stm == White ? Game::Score::BlackWins : Game::Score::WhiteWins);
            }

            if (game.GetPosition().IsMate())
            {
                ASSERT(moveScore >= TablebaseWinValue || moveScore <= -TablebaseWinValue);
            }

            if (game.GetScore() != Game::Score::Unknown)
            {
                if (game.GetScore() == Game::Score::WhiteWins) stats.numWhiteWins++;
                if (game.GetScore() == Game::Score::BlackWins) stats.numBlackWins++;
                if (game.GetScore() == Game::Score::Draw) stats.numDraws++;
                break;
            }
        }

        // save game
        if (halfMoveNumber > 0)
        {
            GameMetadata metadata;
            metadata.roundNumber = index;
            game.SetMetadata(metadata);

            writer.WriteGame(game);

            const bool printToConsole = threadIndex == 0 && config.consolePgnFrequency != 0 && (index % config.consolePgnFrequency == 0);
            if (pgnFile || printToConsole)
            {
                const std::string pgn = game.ToPGN(true);

                if (pgnFile)
                {
                    std::lock_guard<std::mutex> lock(pgnMutex);
                    *pgnFile << pgn << "\n\n";
                    pgnFile->flush();
                }

                if (printToConsole)
                {
                    std::cout << "\n" << pgn << "\n";

                    const uint32_t numGames = stats.numWhiteWins + stats.numBlackWins + stats.numDraws;
                    std::cout << "\n";
                    std::cout << "White wins: " << stats.numWhiteWins << " (" << (stats.numWhiteWins * 100.0 / numGames) << "%)\n";
                    std::cout << "Black wins: " << stats.numBlackWins << " (" << (stats.numBlackWins * 100.0 / numGames) << "%)\n";
                    std::cout << "Draws:      " << stats.numDraws    << " (" << (stats.numDraws    * 100.0 / numGames) << "%)\n";
                }
            }
        }
    }

    return true;
}

static SelfPlayConfig ParseSelfPlayArgs(const std::vector<std::string>& args)
{
    SelfPlayConfig config;

    for (size_t i = 0; i < args.size(); ++i)
    {
        const std::string& arg = args[i];

        if (arg.size() >= 2 && arg[0] == '-' && arg[1] == '-')
        {
            const std::string flag = arg.substr(2);
            auto isValue = [](const std::string& s) -> bool
            {
                // a value token starts with a digit, or '-'/'.' followed by a digit
                return !s.empty() && (std::isdigit((unsigned char)s[0]) ||
                    ((s[0] == '-' || s[0] == '.') && s.size() > 1 && std::isdigit((unsigned char)s[1])));
            };
            const bool hasValue = (i + 1 < args.size()) && isValue(args[i + 1]);

            auto nextUInt = [&]() -> uint32_t
            {
                if (!hasValue) { std::cerr << "Missing value for --" << flag << "\n"; return 0; }
                return static_cast<uint32_t>(std::stoul(args[++i]));
            };
            auto nextInt = [&]() -> int32_t
            {
                if (!hasValue) { std::cerr << "Missing value for --" << flag << "\n"; return 0; }
                return static_cast<int32_t>(std::stol(args[++i]));
            };

            if      (flag == "minNodes")              config.minNodes                  = nextUInt();
            else if (flag == "maxNodes")              config.maxNodes                  = nextUInt();
            else if (flag == "maxDepth")              config.maxDepth                  = nextUInt();
            else if (flag == "maxEval")               config.maxEval                   = nextInt();
            else if (flag == "openingMaxEval")        config.openingMaxEval            = nextInt();
            else if (flag == "minRandomMoves")        config.minRandomMoves            = nextUInt();
            else if (flag == "maxRandomMoves")        config.maxRandomMoves            = nextUInt();
            else if (flag == "drawScoreThreshold")    config.drawScoreThreshold        = nextInt();
            else if (flag == "drawScoreConsecutive")  config.drawScoreConsecutiveMoves = nextUInt();
            else if (flag == "drawMinHalfMove")       config.drawMinHalfMove           = nextUInt();
            else if (flag == "winAdjMinHalfMove")     config.winAdjMinHalfMove         = nextUInt();
            else if (flag == "winAdjConsecutive")     config.winAdjConsecutiveMoves    = nextUInt();
            else if (flag == "syzygyProbeLimit")      config.syzygyProbeLimit          = nextUInt();
            else if (flag == "consolePgnFrequency")   config.consolePgnFrequency       = nextUInt();
            else if (flag == "threads")               config.numThreads                = nextUInt();
            else if (flag == "dumpPgn")               config.dumpAllPgn                = true;
            else
            {
                std::cerr << "Warning: unknown flag --" << flag << ", ignoring\n";
            }
        }
        else
        {
            // positional arg = book path
            config.bookPaths.push_back(arg);
        }
    }

    return config;
}

static std::string BuildOutputBaseName(const SelfPlayConfig& config, uint32_t seed)
{
    // build book stem: concatenate stems of all book paths
    std::string stem;
    for (const std::string& path : config.bookPaths)
    {
        const std::string s = std::filesystem::path(path).stem().string();
        if (!stem.empty()) stem += '_';
        stem += s;
        if (stem.size() > 32)
        {
            stem.resize(32);
            break;
        }
    }
    if (stem.empty())
        stem = "nobook";

    std::ostringstream oss;
    oss << DATA_PATH "selfplayGames/selfplay_"
        << std::hex << seed << std::dec
        << '_' << stem
        << '_' << (config.maxNodes / 1000) << "kn";
    return oss.str();
}

static void WriteConfigFile(const std::string& baseName, const SelfPlayConfig& config, uint32_t seed, uint32_t resolvedNumThreads)
{
    const std::string path = baseName + ".cfg";
    std::ofstream f(path);
    if (!f.is_open())
    {
        std::cerr << "Warning: could not write config file " << path << "\n";
        return;
    }

    f << "seed=" << std::hex << seed << std::dec << "\n";

    f << "bookPaths=";
    for (size_t i = 0; i < config.bookPaths.size(); ++i)
    {
        if (i) f << ';';
        f << config.bookPaths[i];
    }
    f << "\n";

    f << "minNodes="               << config.minNodes                  << "\n";
    f << "maxNodes="               << config.maxNodes                  << "\n";
    f << "maxDepth="               << config.maxDepth                  << "\n";
    f << "maxEval="                << config.maxEval                   << "\n";
    f << "openingMaxEval="         << config.openingMaxEval            << "\n";
    f << "minRandomMoves="         << config.minRandomMoves            << "\n";
    f << "maxRandomMoves="         << config.maxRandomMoves            << "\n";
    f << "drawScoreThreshold="     << config.drawScoreThreshold        << "\n";
    f << "drawScoreConsecutive="   << config.drawScoreConsecutiveMoves << "\n";
    f << "drawMinHalfMove="        << config.drawMinHalfMove           << "\n";
    f << "winAdjMinHalfMove="      << config.winAdjMinHalfMove         << "\n";
    f << "winAdjConsecutive="      << config.winAdjConsecutiveMoves    << "\n";
    f << "syzygyProbeLimit="       << config.syzygyProbeLimit          << "\n";
    f << "syzygyEnabled="          << (HasSyzygyTablebases() ? "true" : "false") << "\n";
    f << "consolePgnFrequency="    << config.consolePgnFrequency       << "\n";
    f << "dumpAllPgn="             << (config.dumpAllPgn ? "true" : "false") << "\n";
    f << "numThreads="             << resolvedNumThreads               << "\n";

    std::cout << "Config written to: " << path << "\n";
}

void SelfPlay(const std::vector<std::string>& args)
{
    SelfPlayConfig config = ParseSelfPlayArgs(args);

    g_syzygyProbeLimit = config.syzygyProbeLimit;

    uint32_t nameSeed = 0;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> distrib;
        nameSeed = distrib(gen);
    }

    std::cout << "Loading opening positions...\n";
    std::vector<PackedPosition> openingPositions;
    for (const std::string& path : config.bookPaths)
    {
        LoadOpeningPositions(path, openingPositions);
    }
    std::cout << "Loaded " << openingPositions.size() << " opening positions\n";

    if (openingPositions.empty())
    {
        std::cout << "No opening positions loaded!\n";
        return;
    }

    // Start at a random offset so every run covers a different segment first, but from there pick sequentially
    std::atomic<uint32_t> openingCounter;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> distrib(0u, (uint32_t)openingPositions.size() - 1u);
        openingCounter = distrib(gen);
    }

    const std::string baseName = BuildOutputBaseName(config, nameSeed);

    // open single shared output file
    const std::string datPath = baseName + ".dat";
    FileOutputStream gamesFile(datPath.c_str());
    if (!gamesFile.IsOK())
    {
        std::cerr << "Failed to open output file: " << datPath << "\n";
        return;
    }
    GameCollection::Writer writer(gamesFile);
    std::cout << "Output: " << datPath << "\n";

    const uint32_t numThreads = config.numThreads > 0
        ? config.numThreads
        : std::max<uint32_t>(1, std::thread::hardware_concurrency());

    // write config file
    WriteConfigFile(baseName, config, nameSeed, numThreads);

    // open optional PGN file
    std::unique_ptr<std::ofstream> pgnFile;
    std::mutex pgnMutex;
    if (config.dumpAllPgn)
    {
        const std::string pgnPath = baseName + ".pgn";
        pgnFile = std::make_unique<std::ofstream>(pgnPath);
        if (!pgnFile->is_open())
        {
            std::cerr << "Failed to open PGN file: " << pgnPath << "\n";
            pgnFile.reset();
        }
        else
        {
            std::cout << "PGN output: " << pgnPath << "\n";
        }
    }

    alignas(CACHELINE_SIZE) SelfPlayStats stats;
    std::atomic<uint32_t> gameCounter{ 0 };

    std::cout << "Starting games...\n";

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([i, &config, &openingPositions, &openingCounter, &gameCounter, &writer, &pgnFile, &pgnMutex, &stats]()
        {
            SelfPlayThreadFunc(i, config, openingPositions, openingCounter, gameCounter, writer, pgnFile.get(), pgnMutex, stats);
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}
