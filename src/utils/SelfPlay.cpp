#include "Common.hpp"
#include "GameCollection.hpp"
#include "../backend/Search.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Tablebase.hpp"

#include <csignal>
#include <fstream>
#include <filesystem>
#include <iostream>

struct SelfPlayConfig
{
    std::vector<std::string> bookPaths;

    uint32_t minNodes                   = 21'000;
    uint32_t maxNodes                   = 21'000;
    uint32_t maxDepth                   = 40;
    int32_t  maxEval                    = InfValue;
    int32_t  openingMaxEval             = 500;
    uint32_t minRandomMoves             = 8;
    uint32_t maxRandomMoves             = 10;
    int32_t  drawScoreThreshold         = 0;
    uint32_t drawScoreConsecutiveMoves  = 10;
    uint32_t drawMinHalfMove            = 80;
    uint32_t winAdjMinHalfMove          = 40;
    uint32_t winAdjConsecutiveMoves     = 3;
    uint32_t syzygyProbeLimit           = 5;
    uint32_t samplePgnFrequency         = 10; // 0 = no sample PGN
    uint32_t statsInterval              = 10; // seconds, 0 = no periodic stats
    uint32_t gamesPerFile               = 1'000'000;
    uint64_t maxGames                   = 0; // 0 = unlimited
    bool     dumpAllPgn                 = false;
    uint32_t numThreads                 = 0; // 0 = hardware_concurrency
};

static std::atomic<bool> s_stopRequested{ false };
static std::atomic<bool> s_interrupted{ false };

static void OnInterrupt(int)
{
    s_interrupted = true;
    s_stopRequested = true;
    // second Ctrl+C terminates immediately
    std::signal(SIGINT, SIG_DFL);
}

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
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;

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

// Writes games to a series of "<baseName>_NNN.dat" files, starting a new file every 'gamesPerFile' games
class SplitGameWriter
{
public:
    SplitGameWriter(const std::string& baseName, uint64_t gamesPerFile, uint64_t maxGames)
        : mBaseName(baseName), mGamesPerFile(gamesPerFile), mMaxGames(maxGames)
    {}

    bool Open()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return OpenNextFile();
    }

    void Close()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mWriter.reset();
        mStream.reset();
    }

    // returns false if the game was not written (game limit reached or output error)
    bool WriteGame(const Game& game)
    {
        std::lock_guard<std::mutex> lock(mMutex);

        if (!mWriter || (mMaxGames > 0 && mNumGames >= mMaxGames))
            return false;

        if (mNumGamesInFile >= mGamesPerFile && !OpenNextFile())
        {
            s_stopRequested = true;
            return false;
        }

        if (!mWriter->WriteGame(game))
        {
            std::cerr << "Failed to write game to " << mCurrentPath << "\n";
            s_stopRequested = true;
            return false;
        }

        mNumGames++;
        mNumGamesInFile++;

        if (mMaxGames > 0 && mNumGames >= mMaxGames)
            s_stopRequested = true;

        return true;
    }

    void GetProgress(uint32_t& outNumFiles, uint64_t& outNumGamesInFile, uint64_t& outNumGames) const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        outNumFiles = mNumFiles;
        outNumGamesInFile = mNumGamesInFile;
        outNumGames = mNumGames;
    }

private:
    bool OpenNextFile()
    {
        mWriter.reset();
        mStream.reset();

        char suffix[16];
        snprintf(suffix, sizeof(suffix), "_%03u.dat", mNumFiles);
        mCurrentPath = mBaseName + suffix;

        mStream = std::make_unique<FileOutputStream>(mCurrentPath.c_str());
        if (!mStream->IsOK())
        {
            std::cerr << "Failed to open output file: " << mCurrentPath << "\n";
            mStream.reset();
            return false;
        }

        mWriter = std::make_unique<GameCollection::Writer>(*mStream);
        mNumFiles++;
        mNumGamesInFile = 0;

        std::cout << "Output: " << mCurrentPath + "\n";
        return true;
    }

    const std::string mBaseName;
    const uint64_t mGamesPerFile;
    const uint64_t mMaxGames;

    mutable std::mutex mMutex;
    std::unique_ptr<FileOutputStream> mStream;
    std::unique_ptr<GameCollection::Writer> mWriter;
    std::string mCurrentPath;
    uint32_t mNumFiles = 0;
    uint64_t mNumGamesInFile = 0;
    uint64_t mNumGames = 0;
};

enum class Termination : uint8_t
{
    Mate,
    DrawRule, // repetition, 50-move rule, insufficient material, stalemate
    Tablebase,
    WinAdjudication,
    DrawAdjudication,
    Count
};

struct SelfPlayStats
{
    std::atomic<uint64_t> numWhiteWins = 0;
    std::atomic<uint64_t> numBlackWins = 0;
    std::atomic<uint64_t> numDraws = 0;
    std::atomic<uint64_t> numPositions = 0;
    std::atomic<uint64_t> numNodes = 0;
    std::atomic<uint64_t> numSkippedOpenings = 0;
    std::atomic<uint64_t> numTerminations[(size_t)Termination::Count] = {};
};

static bool SelfPlayThreadFunc(
    const SelfPlayConfig& config,
    const std::vector<PackedPosition>& openingPositions,
    std::atomic<uint32_t>& openingCounter,
    std::atomic<uint32_t>& gameCounter,
    SplitGameWriter& writer,
    std::ofstream* pgnFile,
    std::mutex& pgnMutex,
    std::ofstream* sampleFile,
    std::mutex& sampleMutex,
    SelfPlayStats& stats)
{
    const size_t c_transpositionTableSize = 4ull * 1024ull * 1024ull;

    std::random_device rd;
    std::mt19937 gen(rd());

    Search search;
    TranspositionTable tt{ c_transpositionTableSize };

    while (!s_stopRequested)
    {
        SearchResult searchResult;
        SearchStats searchStats;

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
        {
            stats.numSkippedOpenings++;
            continue;
        }

        // start new game
        Game game;
        tt.Clear();
        search.Clear();
        game.Reset(openingPos);

        int32_t halfMoveNumber = 0;
        uint32_t drawScoreCounter = 0;
        uint32_t whiteWinsCounter = 0;
        uint32_t blackWinsCounter = 0;
        uint64_t gameNodes = 0;
        Termination termination = Termination::Count;

        const auto adjudicate = [&](Game::Score score, Termination reason)
        {
            game.SetScore(score);
            termination = reason;
        };

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
            search.DoSearch(game, searchParam, searchResult, &searchStats);
            gameNodes += searchStats.nodes;

            ASSERT(!searchResult.empty());

            // skip game if starting position is unbalanced
            if (halfMoveNumber == 0 && std::abs(searchResult.begin()->score) * 100 / wld::NormalizeToPawnValue > config.openingMaxEval)
            {
                stats.numSkippedOpenings++;
                break;
            }

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
                adjudicate(Game::Score::Draw, Termination::DrawAdjudication);
            }

            // adjudicate win
            if (halfMoveNumber >= (int32_t)config.winAdjMinHalfMove)
            {
                if (moveScore > config.maxEval && eval > 0)
                {
                    whiteWinsCounter++;
                    if (whiteWinsCounter > config.winAdjConsecutiveMoves) adjudicate(Game::Score::WhiteWins, Termination::WinAdjudication);
                }
                else
                {
                    whiteWinsCounter = 0;
                }

                if (moveScore < -config.maxEval && eval < 0)
                {
                    blackWinsCounter++;
                    if (blackWinsCounter > config.winAdjConsecutiveMoves) adjudicate(Game::Score::BlackWins, Termination::WinAdjudication);
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
                if (wdlScore == 1) adjudicate(stm == White ? Game::Score::WhiteWins : Game::Score::BlackWins, Termination::Tablebase);
                if (wdlScore == 0) adjudicate(Game::Score::Draw, Termination::Tablebase);
                if (wdlScore == -1) adjudicate(stm == White ? Game::Score::BlackWins : Game::Score::WhiteWins, Termination::Tablebase);
            }

            if (game.GetPosition().IsMate())
            {
                ASSERT(moveScore >= TablebaseWinValue || moveScore <= -TablebaseWinValue);
            }

            if (game.GetScore() != Game::Score::Unknown)
            {
                if (game.GetForcedScore() == Game::Score::Unknown)
                    termination = game.GetPosition().IsMate() ? Termination::Mate : Termination::DrawRule;
                break;
            }
        }

        stats.numNodes += gameNodes;

        // save game
        if (halfMoveNumber > 0)
        {
            GameMetadata metadata;
            metadata.roundNumber = index;
            game.SetMetadata(metadata);

            if (!writer.WriteGame(game))
                break;

            if (game.GetScore() == Game::Score::WhiteWins) stats.numWhiteWins++;
            if (game.GetScore() == Game::Score::BlackWins) stats.numBlackWins++;
            if (game.GetScore() == Game::Score::Draw) stats.numDraws++;
            stats.numPositions += game.GetMoves().size();
            stats.numTerminations[(size_t)termination]++;

            const bool writeSample = sampleFile && config.samplePgnFrequency != 0 && (index % config.samplePgnFrequency == 0);
            if (pgnFile || writeSample)
            {
                const std::string pgn = game.ToPGN(true);

                if (pgnFile)
                {
                    std::lock_guard<std::mutex> lock(pgnMutex);
                    *pgnFile << pgn << "\n\n";
                    pgnFile->flush();
                }

                if (writeSample)
                {
                    std::lock_guard<std::mutex> lock(sampleMutex);
                    *sampleFile << pgn << "\n\n";
                    sampleFile->flush();
                }
            }
        }
    }

    return true;
}

// e.g. 950, 12.3K, 4.56M
static std::string FormatCount(double value, int smallValueDecimals = 0)
{
    char buf[32];
    if (value >= 1.0e9)         snprintf(buf, sizeof(buf), "%.2fG", value * 1.0e-9);
    else if (value >= 1.0e6)    snprintf(buf, sizeof(buf), "%.2fM", value * 1.0e-6);
    else if (value >= 1.0e3)    snprintf(buf, sizeof(buf), "%.1fK", value * 1.0e-3);
    else                        snprintf(buf, sizeof(buf), "%.*f", smallValueDecimals, value);
    return buf;
}

struct StatsSample
{
    uint64_t numGames = 0;
    double seconds = 0.0;
};

// 'prevSample' is the previous report, used for the games/s rate since then; nullptr prints the whole-run rate only
static StatsSample PrintStats(const SelfPlayStats& stats, const SplitGameWriter& writer, const SelfPlayConfig& config, double seconds, const StatsSample* prevSample)
{
    const uint64_t numGames = stats.numWhiteWins + stats.numBlackWins + stats.numDraws;
    const uint64_t numPositions = stats.numPositions;
    const uint64_t numSkipped = stats.numSkippedOpenings;

    uint32_t numFiles = 0;
    uint64_t numGamesInFile = 0;
    uint64_t numGamesWritten = 0;
    writer.GetProgress(numFiles, numGamesInFile, numGamesWritten);

    const auto percent = [](uint64_t count, uint64_t total) { return total > 0 ? 100.0 * (double)count / (double)total : 0.0; };
    const double invSeconds = seconds > 0.0 ? 1.0 / seconds : 0.0;
    const uint32_t totalSeconds = (uint32_t)seconds;

    char timeTag[32];
    const int timeTagLength = snprintf(timeTag, sizeof(timeTag), "[%u:%02u:%02u]", totalSeconds / 3600, (totalSeconds / 60) % 60, totalSeconds % 60);

    std::string gamesRate = FormatCount((double)numGames * invSeconds, 1) + "/s";
    if (prevSample && seconds > prevSample->seconds)
    {
        const double recentRate = (double)(numGames - prevSample->numGames) / (seconds - prevSample->seconds);
        gamesRate += ", last " + std::to_string(config.statsInterval) + "s: " + FormatCount(recentRate, 1) + "/s";
    }

    char buf[512];
    std::string str;

    snprintf(buf, sizeof(buf), "%s games %s (%s) | positions %s (%s/s) | %s nodes/s | avg %.1f plies | white %.1f%% draw %.1f%% black %.1f%%\n",
        timeTag,
        FormatCount((double)numGames).c_str(), gamesRate.c_str(),
        FormatCount((double)numPositions).c_str(), FormatCount((double)numPositions * invSeconds, 1).c_str(),
        FormatCount((double)stats.numNodes * invSeconds).c_str(),
        numGames > 0 ? (double)numPositions / (double)numGames : 0.0,
        percent(stats.numWhiteWins, numGames), percent(stats.numDraws, numGames), percent(stats.numBlackWins, numGames));
    str += buf;

    snprintf(buf, sizeof(buf), "%*s end: mate %.1f%%, rule draw %.1f%%, TB %.1f%%, win adj %.1f%%, draw adj %.1f%% | skipped openings %.1f%% | file %03u: %s/%s games\n",
        timeTagLength, "",
        percent(stats.numTerminations[(size_t)Termination::Mate], numGames),
        percent(stats.numTerminations[(size_t)Termination::DrawRule], numGames),
        percent(stats.numTerminations[(size_t)Termination::Tablebase], numGames),
        percent(stats.numTerminations[(size_t)Termination::WinAdjudication], numGames),
        percent(stats.numTerminations[(size_t)Termination::DrawAdjudication], numGames),
        percent(numSkipped, numSkipped + numGames),
        numFiles > 0 ? numFiles - 1 : 0u,
        FormatCount((double)numGamesInFile).c_str(), FormatCount((double)config.gamesPerFile).c_str());
    str += buf;

    std::cout << str << std::flush;

    return { numGames, seconds };
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
            else if (flag == "samplePgnFrequency")    config.samplePgnFrequency        = nextUInt();
            else if (flag == "statsInterval")         config.statsInterval             = nextUInt();
            else if (flag == "gamesPerFile")          config.gamesPerFile              = std::max(1u, nextUInt());
            else if (flag == "games")                 config.maxGames                  = nextUInt();
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
    f << "samplePgnFrequency="     << config.samplePgnFrequency        << "\n";
    f << "statsInterval="          << config.statsInterval             << "\n";
    f << "gamesPerFile="           << config.gamesPerFile              << "\n";
    f << "games="                  << config.maxGames                  << "\n";
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

    SplitGameWriter writer(baseName, config.gamesPerFile, config.maxGames);
    if (!writer.Open())
        return;

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

    // sample of games for inspection, overwritten on every run
    std::unique_ptr<std::ofstream> sampleFile;
    std::mutex sampleMutex;
    if (config.samplePgnFrequency > 0)
    {
        const std::string samplePath = DATA_PATH "datagen_sample.pgn";
        sampleFile = std::make_unique<std::ofstream>(samplePath);
        if (!sampleFile->is_open())
        {
            std::cerr << "Failed to open PGN sample file: " << samplePath << "\n";
            sampleFile.reset();
        }
        else
        {
            std::cout << "PGN sample (every " << config.samplePgnFrequency << " games): " << samplePath << "\n";
        }
    }

    alignas(CACHELINE_SIZE) SelfPlayStats stats;
    std::atomic<uint32_t> gameCounter{ 0 };

    s_stopRequested = false;
    s_interrupted = false;
    std::signal(SIGINT, OnInterrupt);

    std::cout << "Starting games on " << numThreads << " threads";
    if (config.maxGames > 0) std::cout << " (" << config.maxGames << " games)";
    std::cout << ", press Ctrl+C to stop...\n";

    const auto startTime = std::chrono::steady_clock::now();
    const auto getElapsedSeconds = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count(); };

    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([&]()
        {
            SelfPlayThreadFunc(config, openingPositions, openingCounter, gameCounter, writer, pgnFile.get(), pgnMutex, sampleFile.get(), sampleMutex, stats);
        });
    }

    std::atomic<bool> workersDone{ false };
    std::thread statsThread([&]()
    {
        const auto interval = std::chrono::seconds(config.statsInterval);
        auto nextReportTime = startTime + interval;
        bool interruptReported = false;
        StatsSample prevSample;

        while (!workersDone)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (s_interrupted && !interruptReported)
            {
                std::cout << "Stopping: finishing games in progress (press Ctrl+C again to abort)...\n" << std::flush;
                interruptReported = true;
            }

            if (config.statsInterval > 0 && std::chrono::steady_clock::now() >= nextReportTime)
            {
                prevSample = PrintStats(stats, writer, config, getElapsedSeconds(), &prevSample);
                nextReportTime += interval;
            }
        }
    });

    for (auto& thread : threads)
    {
        thread.join();
    }

    workersDone = true;
    statsThread.join();

    writer.Close();
    std::signal(SIGINT, SIG_DFL);

    uint32_t numFiles = 0;
    uint64_t numGamesInFile = 0;
    uint64_t numGamesWritten = 0;
    writer.GetProgress(numFiles, numGamesInFile, numGamesWritten);

    std::cout << "Finished: " << numGamesWritten << " games written to " << numFiles << " file(s)\n";
    PrintStats(stats, writer, config, getElapsedSeconds(), nullptr);
}
