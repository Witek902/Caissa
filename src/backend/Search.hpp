#pragma once

#include "Position.hpp"
#include "MoveList.hpp"
#include "MoveOrderer.hpp"

#include "nnue-probe/nnue.h"

#include <chrono>
#include <atomic>

#ifndef CONFIGURATION_FINAL
#define COLLECT_SEARCH_STATS
#endif // CONFIGURATION_FINAL

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct SearchLimits
{
    // minimum time after which root singularity search kicks in
    uint32_t rootSingularityTime = UINT32_MAX;

    // suggested search time in milliseconds, it's checked every iteration so can be exceeded
    uint32_t maxTimeSoft = UINT32_MAX;

    // maximum allowed search time in milliseconds, after that all search must be stopped immediately
    uint32_t maxTime = UINT32_MAX;

    // maximum allowed searched nodes
    uint64_t maxNodes = UINT64_MAX;

    // maximum allowed base search depth (excluding quisence, extensions, etc.)
    uint8_t maxDepth = UINT8_MAX;

    // enable mate search, disables all prunning
    bool mateSearch = false;

    // in analysis mode full PV lines are searched
    bool analysisMode = false;
};

struct SearchParam
{
    // shared transposition table
    TranspositionTable& transpositionTable;

    // used to track total time spend on search
    TimePoint startTimePoint;

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

    // move notation for PV lines printing
    MoveNotation moveNotation = MoveNotation::LAN;

    // print move scores for the root nodes
    bool printMoves = false;

    // print verbose debug stats (not UCI comaptible)
    bool verboseStats = false;

    int64_t GetElapsedTime() const;
};

struct PvLine
{
    std::vector<Move> moves;
    ScoreType score = 0;
};

using SearchResult = std::vector<PvLine>;

struct NodeInfo
{
    Position position;

    NodeInfo* parentNode = nullptr;

    // ignore given moves in search, used for multi-PV search
    const Move* moveFilter = nullptr;

    // remaining depth
    int32_t depth;

    // depth in ply (depth counting from root)
    uint32_t height;

    uint8_t moveFilterCount = 0;

    ScoreType alpha;
    ScoreType beta;

    ScoreType staticEval = InvalidValue;

    Move previousMove = Move::Invalid();

    uint8_t pvIndex;

    bool isPvNode = false;
    bool isNullMove = false;

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

    struct SearchStats
    {
        std::atomic<uint64_t> nodes = 0;
        uint64_t quiescenceNodes = 0;
        uint32_t maxDepth = 0;

#ifdef COLLECT_SEARCH_STATS
        uint64_t ttHits = 0;
        uint64_t ttWrites = 0;
        uint64_t tbHits = 0;
        uint64_t betaCutoffHistogram[MoveList::MaxMoves] = { 0 };
#endif // COLLECT_SEARCH_STATS
    };

    struct SearchContext
    {
        const Game& game;
        const SearchParam& searchParam;
        SearchStats stats;
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
        ScoreType previousScore;                  // score in previous ID iteration
        uint32_t threadID = 0;
    };

    struct ThreadData
    {
        bool isMainThread = false;

        // principial variation moves tracking for current search
        PackedMove pvArray[MaxSearchDepth][MaxSearchDepth];
        uint8_t pvLengths[MaxSearchDepth];

        // principial variation lines from previous iterative deepening search
        SearchResult prevPvLines;

        MoveOrderer moveOrderer;

        // update principal variation line
        void UpdatePvArray(uint32_t depth, const Move move);

        // check if one of generated moves is in PV table
        const Move FindPvMove(const NodeInfo& node, MoveList& moves) const;
    };

    std::atomic<bool> mStopSearch = false;
    
    std::vector<ThreadData> mThreadData;

    static constexpr uint32_t MaxReducedMoves = 64;
    uint8_t mMoveReductionTable[MaxSearchDepth][MaxReducedMoves];

    void BuildMoveReductionTable();

    void ReportPV(const AspirationWindowSearchParam& param, const PvLine& pvLine, BoundsType boundsType, const std::chrono::high_resolution_clock::duration searchTime) const;

    void Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, const SearchParam& param, SearchResult& outResult);

    bool IsDraw(const NodeInfo& node, const Game& game) const;

    bool IsSingular(const Position& position, const Move move, ThreadData& thread, SearchContext& ctx) const;

    PvLine AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param) const;

    ScoreType QuiescenceNegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const;
    ScoreType NegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const;

    // check for repetition in the searched node
    bool IsRepetition(const NodeInfo& node, const Game& game) const;

    // reconstruct PV line from cache and TT table
    static std::vector<Move> GetPvLine(const ThreadData& thread, const Position& pos, const TranspositionTable& tt, uint32_t maxLength);

    // returns true if the search needs to be aborted immediately
    bool CheckStopCondition(const SearchContext& ctx) const;
};