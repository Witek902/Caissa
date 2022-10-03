#pragma once

#include "Position.hpp"
#include "MoveList.hpp"
#include "MoveOrderer.hpp"
#include "Time.hpp"
#include "Memory.hpp"
#include "NeuralNetworkEvaluator.hpp"

#include <atomic>

#ifndef CONFIGURATION_FINAL
#define COLLECT_SEARCH_STATS
#endif // CONFIGURATION_FINAL

struct SearchLimits
{
    // a time point where search started
    TimePoint startTimePoint = TimePoint::Invalid();

    // minimum time after which root singularity search kicks in
    TimePoint rootSingularityTime = TimePoint::Invalid();

    // suggested search time, it's checked every iteration so can be exceeded
    TimePoint idealTime = TimePoint::Invalid();

    // maximum allowed search time, after that all search must be stopped immediately
    TimePoint maxTime = TimePoint::Invalid();

    // maximum allowed searched nodes
    uint64_t maxNodes = UINT64_MAX;

    // maximum allowed base search depth (excluding quisence, extensions, etc.)
    uint8_t maxDepth = UINT8_MAX;

    // enable mate search, disables all pruning
    bool mateSearch = false;

    // in analysis mode full PV lines are searched
    bool analysisMode = false;
};

// Utility that allows for collecting evaluated positions during the search
// This is used for collecting positions for parameter tuning
class EvalProbingInterface
{
public:
    virtual void ReportPosition(const Position& pos, ScoreType eval) = 0;
};

struct SearchParam
{
    // shared transposition table
    TranspositionTable& transpositionTable;

    // search limits
    SearchLimits limits;

    uint32_t numThreads = 1;

    // number of PV lines to report
    uint32_t numPvLines = 1;

    // exclude this root moves from the search
    std::vector<Move> excludedMoves;

    // in pondering we don't care about limits
    bool isPonder = false;

    // print UCI-style output
    bool debugLog = true;

    // probe tablebases at the root
    bool useRootTablebase = true;

    // use colors in console output to make it more readable
    bool colorConsoleOutput = false;

    // move notation for PV lines printing
    MoveNotation moveNotation = MoveNotation::LAN;

    // print verbose debug stats (not UCI comaptible)
    bool verboseStats = false;

    // optional eval probing interface
    EvalProbingInterface* evalProbingInterface = nullptr;
};

struct PvLine
{
    std::vector<Move> moves;
    ScoreType score = InvalidValue;
    ScoreType tbScore = InvalidValue;
};

using SearchResult = std::vector<PvLine>;

struct NodeInfo
{
    Position position;

    NodeInfo* parentNode = nullptr;

    // ignore given moves in search, used for multi-PV search
    const Move* moveFilter = nullptr;

    // remaining depth
    int32_t depth = 0;

    // depth in ply (depth counting from root)
    uint32_t height = 0;

    uint8_t moveFilterCount = 0;

    ScoreType alpha;
    ScoreType beta;

    ScoreType staticEval = InvalidValue;

    Move previousMove = Move::Invalid();

    uint8_t pvIndex = 0;

    uint16_t pvLength = 0;
    PackedMove pvLine[MaxSearchDepth];

    bool isPvNodeFromPrevIteration = false;
    bool isPvNode = false;
    bool isCutNode = false;
    bool isNullMove = false;
    bool isInCheck = false;
    bool isSingularSearch = false;

    NNEvaluatorContext* nnContext = nullptr;

    bool ShouldCheckMove(const Move move) const
    {
        for (uint32_t i = 0; i < moveFilterCount; ++i)
        {
            if (move == moveFilter[i])
            {
                return false;
            }
        }

        return true;
    }
};

class Search
{
public:

    Search();
    ~Search();

    void Clear();

    void DoSearch(const Game& game, const SearchParam& param, SearchResult& outResult);

    void StopSearch();

    const MoveOrderer& GetMoveOrderer() const;

private:

    Search(const Search&) = delete;

    enum class BoundsType : uint8_t
    {
        Exact = 0,
        LowerBound = 1,
        UpperBound = 2,
    };

    struct ThreadStats
    {
        uint64_t nodes = 0;
        uint64_t quiescenceNodes = 0;
        uint32_t maxDepth = 0;
    };

    struct Stats
    {
        std::atomic<uint64_t> nodes = 0;
        std::atomic<uint64_t> quiescenceNodes = 0;
        std::atomic<uint32_t> maxDepth = 0;

#ifdef COLLECT_SEARCH_STATS
        static const int32_t EvalHistogramMaxValue = 1600;
        static const int32_t EvalHistogramBins = 100;
        uint64_t ttHits = 0;
        uint64_t ttWrites = 0;
        uint64_t tbHits = 0;
        uint64_t betaCutoffHistogram[MoveList::MaxMoves] = { 0 };
        uint64_t evalHistogram[EvalHistogramBins] = { 0 };
#endif // COLLECT_SEARCH_STATS

        void Append(ThreadStats& threadStats, bool flush = false);
    };

    struct SearchContext
    {
        const Game& game;
        const SearchParam& searchParam;
        Stats& stats;
        TimePoint maxTimeSoft = TimePoint::Invalid();
    };

    struct AspirationWindowSearchParam
    {
        const Position& position;
        const SearchParam& searchParam;
        uint32_t depth;
        uint8_t pvIndex;
        SearchContext& searchContext;
        const Move* moveFilter = nullptr;
        uint8_t moveFilterCount = 0;
        ScoreType previousScore = 0;                  // score in previous ID iteration
        uint32_t threadID = 0;
    };

    struct ThreadData
    {
        bool isMainThread = false;

        // search depth at the root node in current iterative deepening step
        uint32_t rootDepth = 0;

        // principial variation lines from previous iterative deepening search
        SearchResult prevPvLines;

        // per-thread search stats
        ThreadStats stats;

        // per-thread move orderer
        MoveOrderer moveOrderer;

        // neural network context for each node height
        std::vector<NNEvaluatorContext, AlignmentAllocator<NNEvaluatorContext, CACHELINE_SIZE>> nnContextStack{ MaxSearchDepth };

        // get PV move from previous depth iteration
        const Move GetPvMove(const NodeInfo& node) const;
    };

    mutable std::atomic<bool> mStopSearch = false;
    
    std::vector<ThreadData, Allocator<ThreadData>> mThreadData;

    static constexpr uint32_t MaxReducedMoves = 64;
    uint8_t mMoveReductionTable[MaxSearchDepth][MaxReducedMoves];

    void BuildMoveReductionTable();

    void ReportPV(const AspirationWindowSearchParam& param, const PvLine& pvLine, BoundsType boundsType, const TimePoint& searchTime) const;
    void ReportCurrentMove(const Move& move, int32_t depth, uint32_t moveNumber) const;

    void Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, const SearchParam& param, Stats& outStats, SearchResult& outResult);

    bool IsSingular(const Position& position, const Move move, ThreadData& thread, SearchContext& ctx) const;

    PvLine AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param) const;

    ScoreType QuiescenceNegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const;
    ScoreType NegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const;

    // returns true if the search needs to be aborted immediately
    bool CheckStopCondition(const ThreadData& thread, const SearchContext& ctx, bool isRootNode) const;
};