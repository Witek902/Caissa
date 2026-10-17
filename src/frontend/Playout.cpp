#include "Playout.hpp"
#include "../backend/Evaluate.hpp"
#include "../backend/Game.hpp"
#include "../backend/Search.hpp"
#include "../backend/containers/DynArray.hpp"
#include "../backend/containers/FixedString.hpp"
#include "../backend/containers/Sort.hpp"

#include <chrono>
#include <condition_variable>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <thread>

static constexpr size_t c_transpositionTableSize = 8ull * 1024ull * 1024ull;
static constexpr uint32_t c_reportIntervalSeconds = 5;
static constexpr int32_t c_evalRandomization = 1;
static constexpr uint64_t c_hardNodeLimitMultiplier = 5;

enum class GameEnd : uint8_t
{
    Mate,
    Stalemate,
    Repetition,
    FiftyMoveRule,
    InsufficientMaterial,
};

// returns true if the game has ended, writing the reason to outGameEnd
static bool ClassifyGameEnd(const Game& game, GameEnd& outGameEnd)
{
    const Position& pos = game.GetPosition();

    // mate and stalemate take priority, so that a mate delivered on the 100th halfmove
    // is not reported as a fifty-move-rule draw
    if (pos.IsMate())                       { outGameEnd = GameEnd::Mate;                   return true; }
    if (pos.IsStalemate())                  { outGameEnd = GameEnd::Stalemate;              return true; }
    if (CheckInsufficientMaterial(pos))     { outGameEnd = GameEnd::InsufficientMaterial;   return true; }
    if (game.GetRepetitionCount(pos) >= 3)  { outGameEnd = GameEnd::Repetition;             return true; }
    if (pos.IsFiftyMoveRuleDraw())          { outGameEnd = GameEnd::FiftyMoveRule;          return true; }

    return false;
}

struct PlayoutStats
{
    uint64_t numGames = 0;

    // outcomes from the perspective of the side to move in the root position
    uint64_t numWins = 0;
    uint64_t numDraws = 0;
    uint64_t numLosses = 0;

    uint64_t numStalemates = 0;
    uint64_t numRepetitions = 0;
    uint64_t numFiftyMoveDraws = 0;
    uint64_t numInsufficientMaterial = 0;

    uint64_t totalPlies = 0;
    uint64_t totalNodes = 0;
    uint32_t minPlies = UINT32_MAX;
    uint32_t maxPlies = 0;

    DynArray<uint32_t> gameLengths;
};

static constexpr size_t c_reportWidth = 62;

static FixedString<32> FormatDuration(double seconds)
{
    char buffer[32];
    const uint32_t total = static_cast<uint32_t>(seconds);

    if (total < 60)
        snprintf(buffer, sizeof(buffer), "%us", total);
    else if (total < 3600)
        snprintf(buffer, sizeof(buffer), "%um %02us", total / 60, total % 60);
    else
        snprintf(buffer, sizeof(buffer), "%uh %02um", total / 3600, (total / 60) % 60);

    return buffer;
}

static uint64_t SplitMix64(uint64_t& state)
{
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct WorkerContext
{
    TranspositionTable tt{ c_transpositionTableSize };
    Search search;
    SearchParam param{ tt };
    std::thread thread;
};

struct PlayoutRunner::Impl
{
    Position rootPosition;
    uint64_t softNodeLimit = 0;
    uint64_t maxGames = 0; // 0 = unlimited

    // owned
    DynArray<WorkerContext*> workers;
    std::thread reporterThread;

    std::atomic<bool> stopRequested = false;
    std::atomic<uint32_t> numActiveWorkers = 0;
    std::atomic<uint64_t> gameCounter = 0;

    std::mutex statsMutex;
    PlayoutStats stats;

    std::mutex reporterMutex;
    std::condition_variable reporterCV;

    TimePoint startTime;

    ~Impl()
    {
        for (WorkerContext* worker : workers)
            delete worker;
    }

    void WorkerFunc(WorkerContext& ctx);
    void ReporterFunc();
    void PrintReport(bool isFinal);
};

void PlayoutRunner::Impl::WorkerFunc(WorkerContext& ctx)
{
    // differs between workers and runs
    uint64_t seedState = reinterpret_cast<uintptr_t>(&ctx) ^ static_cast<uint64_t>(1.0e6 * TimePoint::GetCurrent().ToSeconds());

    SearchResult searchResult;
    SearchStats searchStats;

    ctx.param.debugLog = false;
    ctx.param.useRootTablebase = false;
    ctx.param.numThreads = 1;
    ctx.param.numPvLines = 1;
    ctx.param.evalRandomization = c_evalRandomization;
    ctx.param.limits.maxNodesSoft = softNodeLimit;
    ctx.param.limits.maxNodes = c_hardNodeLimitMultiplier * softNodeLimit;

    const Color rootSideToMove = rootPosition.GetSideToMove();

    while (!stopRequested.load(std::memory_order_relaxed))
    {
        if (maxGames > 0 && gameCounter.fetch_add(1, std::memory_order_relaxed) >= maxGames)
            break;

        Game game;
        game.Reset(rootPosition);
        ctx.tt.Clear();
        ctx.search.Clear();

        // a fresh seed per game is the only source of variation between games
        ctx.param.seed = static_cast<uint32_t>(SplitMix64(seedState));

        uint32_t numPlies = 0;
        uint64_t numNodes = 0;
        GameEnd gameEnd = GameEnd::Mate;
        bool aborted = false;

        while (!ClassifyGameEnd(game, gameEnd))
        {
            ctx.param.stopSearch = false;
            searchResult.Clear();
            ctx.tt.NextGeneration();
            ctx.search.DoSearch(game, ctx.param, searchResult, &searchStats);

            if (stopRequested.load(std::memory_order_relaxed))
            {
                aborted = true;
                break;
            }

            ASSERT(!searchResult.Empty());
            ASSERT(!searchResult.Front().moves.Empty());

            numNodes += searchStats.nodes;
            numPlies++;

            const bool moveSuccess = game.DoMove(searchResult.Front().moves.Front());
            ASSERT(moveSuccess);
            (void)moveSuccess;
        }

        if (aborted)
            break;

        std::lock_guard<std::mutex> lock(statsMutex);

        stats.numGames++;
        stats.totalPlies += numPlies;
        stats.totalNodes += numNodes;
        stats.minPlies = Min(stats.minPlies, numPlies);
        stats.maxPlies = Max(stats.maxPlies, numPlies);
        stats.gameLengths.PushBack(numPlies);

        switch (gameEnd)
        {
        case GameEnd::Mate:
            // the side to move is the one being mated
            if (game.GetPosition().GetSideToMove() == rootSideToMove)
                stats.numLosses++;
            else
                stats.numWins++;
            break;
        case GameEnd::Stalemate:
            stats.numDraws++;
            stats.numStalemates++;
            break;
        case GameEnd::Repetition:
            stats.numDraws++;
            stats.numRepetitions++;
            break;
        case GameEnd::FiftyMoveRule:
            stats.numDraws++;
            stats.numFiftyMoveDraws++;
            break;
        case GameEnd::InsufficientMaterial:
            stats.numDraws++;
            stats.numInsufficientMaterial++;
            break;
        }
    }

    if (numActiveWorkers.fetch_sub(1, std::memory_order_acq_rel) == 1)
        reporterCV.notify_all();
}

void PlayoutRunner::Impl::PrintReport(bool isFinal)
{
    PlayoutStats snapshot;
    {
        std::lock_guard<std::mutex> lock(statsMutex);
        snapshot = stats;
    }

    const double elapsed = (TimePoint::GetCurrent() - startTime).ToSeconds();

    char separator[c_reportWidth + 1];
    memset(separator, '=', c_reportWidth);
    separator[c_reportWidth] = 0;

    char header[c_reportWidth + 1];
    const int headerLength = snprintf(header, sizeof(header), "=== playout %s", isFinal ? "(final) " : "");
    memset(header + headerLength, '=', c_reportWidth - headerLength);
    header[c_reportWidth] = 0;
    printf("\n%s\n", header);

    printf("  setup    %llu nodes/move   %u threads   %s to move\n",
        (unsigned long long)softNodeLimit,
        workers.Size(),
        rootPosition.GetSideToMove() == White ? "white" : "black");

    if (snapshot.numGames == 0)
    {
        printf("  games    none completed yet (%s)\n", FormatDuration(elapsed).CStr());
        printf("%s\n", separator);
        fflush(stdout);
        return;
    }

    const double numGames = static_cast<double>(snapshot.numGames);

    // a single game scores 0, 0.5 or 1, so the score variance follows from the outcome counts alone
    const double score = (snapshot.numWins + 0.5 * snapshot.numDraws) / numGames;
    const double secondMoment = (snapshot.numWins + 0.25 * snapshot.numDraws) / numGames;
    const double scoreStdError = sqrt(Max(0.0, secondMoment - score * score) / numGames);

    const uint32_t middle = snapshot.gameLengths.Size() / 2;
    Sort(snapshot.gameLengths.begin(), snapshot.gameLengths.end(), [](uint32_t a, uint32_t b) { return a < b; });
    const uint32_t medianPlies = snapshot.gameLengths[middle];

    // outcome and draw-reason columns are printed side by side, all percentages relative to the total game count
    const char* drawLabels[] = { "3-fold", "insufficient", "50-move", "stalemate" };
    const uint64_t drawCounts[] = { snapshot.numRepetitions, snapshot.numInsufficientMaterial, snapshot.numFiftyMoveDraws, snapshot.numStalemates };

    char left[4][48];
    snprintf(left[0], sizeof(left[0]), "    %-6s %7llu  %5.1f%%", "win",  (unsigned long long)snapshot.numWins,   100.0 * snapshot.numWins / numGames);
    snprintf(left[1], sizeof(left[1]), "    %-6s %7llu  %5.1f%%", "draw", (unsigned long long)snapshot.numDraws,  100.0 * snapshot.numDraws / numGames);
    snprintf(left[2], sizeof(left[2]), "    %-6s %7llu  %5.1f%%", "loss", (unsigned long long)snapshot.numLosses, 100.0 * snapshot.numLosses / numGames);
    snprintf(left[3], sizeof(left[3]), "    %-6s %6.1f%% +/- %.1f%%", "score", 100.0 * score, 100.0 * 1.959964 * scoreStdError);

    printf("\n  outcome                      draws (%% of all games)\n");
    for (uint32_t i = 0; i < 4; ++i)
    {
        printf("%-30s   %-14s %6llu  %5.1f%%\n", left[i], drawLabels[i], (unsigned long long)drawCounts[i], 100.0 * drawCounts[i] / numGames);
    }

    printf("\n  games    %llu in %s   (%.1f games/s)\n",
        (unsigned long long)snapshot.numGames, FormatDuration(elapsed).CStr(), numGames / elapsed);
    printf("  length   avg %.1f   median %u   min %u   max %u plies\n",
        (double)snapshot.totalPlies / numGames, medianPlies, snapshot.minPlies, snapshot.maxPlies);
    printf("  search   %.1f Mnps total   avg %.0f nodes/move\n",
        snapshot.totalNodes / elapsed / 1000000.0,
        (double)snapshot.totalNodes / (double)snapshot.totalPlies);

    fflush(stdout);
}

void PlayoutRunner::Impl::ReporterFunc()
{
    uint64_t lastReportedGames = 0;

    for (;;)
    {
        {
            std::unique_lock<std::mutex> lock(reporterMutex);
            reporterCV.wait_for(lock, std::chrono::seconds(c_reportIntervalSeconds), [this]()
            {
                return stopRequested.load(std::memory_order_relaxed) ||
                       numActiveWorkers.load(std::memory_order_relaxed) == 0;
            });
        }

        if (stopRequested.load(std::memory_order_relaxed) ||
            numActiveWorkers.load(std::memory_order_relaxed) == 0)
            break;

        uint64_t numGames;
        {
            std::lock_guard<std::mutex> lock(statsMutex);
            numGames = stats.numGames;
        }

        if (numGames != lastReportedGames)
        {
            lastReportedGames = numGames;
            PrintReport(false);
        }
    }

    for (auto& worker : workers)
    {
        if (worker->thread.joinable())
            worker->thread.join();
    }

    PrintReport(true);
}

PlayoutRunner::PlayoutRunner()
    : mImpl(new Impl)
{}

PlayoutRunner::~PlayoutRunner()
{
    Stop();
    delete mImpl;
}

bool PlayoutRunner::Start(const Position& rootPosition, uint64_t softNodeLimit, uint64_t maxGames)
{
    Game game;
    game.Reset(rootPosition);

    GameEnd gameEnd;
    if (ClassifyGameEnd(game, gameEnd))
    {
        printf("Root position is already a finished game\n");
        return false;
    }

    const uint32_t numThreads = Max(1u, std::thread::hardware_concurrency());

    mImpl->rootPosition = rootPosition;
    mImpl->softNodeLimit = softNodeLimit;
    mImpl->maxGames = maxGames;
    mImpl->numActiveWorkers = numThreads;
    mImpl->startTime = TimePoint::GetCurrent();

    mImpl->workers.Reserve(numThreads);
    for (uint32_t i = 0; i < numThreads; ++i)
        mImpl->workers.PushBack(new WorkerContext);

    for (WorkerContext* ctx : mImpl->workers)
    {
        ctx->thread = std::thread([this, ctx]() { mImpl->WorkerFunc(*ctx); });
    }

    mImpl->reporterThread = std::thread([this]() { mImpl->ReporterFunc(); });

    printf("Playing games from the current position at %llu nodes/move on %u threads", (unsigned long long)softNodeLimit, numThreads);
    if (maxGames > 0)
        printf(", %llu games", (unsigned long long)maxGames);
    else
        printf(", use 'stop' to finish");
    printf("\n");

    return true;
}

void PlayoutRunner::Stop()
{
    mImpl->stopRequested = true;

    for (auto& worker : mImpl->workers)
        worker->param.stopSearch = true;

    mImpl->reporterCV.notify_all();

    if (mImpl->reporterThread.joinable())
        mImpl->reporterThread.join();
}

bool PlayoutRunner::IsRunning() const
{
    return mImpl->numActiveWorkers.load(std::memory_order_relaxed) > 0;
}
