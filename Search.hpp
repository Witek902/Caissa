#pragma once

#include "Position.hpp"
#include "MoveList.hpp"
#include "MoveOrderer.hpp"
#include "TranspositionTable.hpp"

#include <span>
#include <chrono>

class Game;

struct SearchLimits
{
    // maximum allowed base search depth (excluding quisence, extensions, etc.)
    uint32_t maxDepth = 8;

    // suggested search time in milliseconds, it's checked every iteration so can be exceeded
    uint32_t maxTimeSoft = UINT32_MAX;

    // maximum allowed search time in milliseconds, after that all search must be stopped immediately
    uint32_t maxTime = UINT32_MAX;

    // maximum allowed searched nodes
    uint64_t maxNodes = UINT64_MAX;
};

struct SearchParam
{
    // used to track total time spend on search
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

    // search limits
    SearchLimits limits;

    // number of PV lines to report
    uint32_t numPvLines = 1;

    // if not empty, only consider this moves
    std::vector<Move> rootMoves;

    // in pondering we don't care about limits
    bool isPonder = false;

    // print UCI-style output
    bool debugLog = true;

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
    const Position* position = nullptr;
    NodeInfo* parentNode = nullptr;
    ScoreType alpha;
    ScoreType beta;
    Move previousMove = Move::Invalid();
    std::span<const Move> moveFilter;   // ignore given moves in search, used for multi-PV search
    std::span<const Move> rootMoves;    // consider only this moves at root node, used for "searchmoves" UCI command
    int32_t depth;                   // remaining depth
    uint32_t height;                    // depth in ply (depth counting from root)
    uint8_t pvIndex;
    Color color;
    bool isPvNode = false;
    bool isTbNode = false;
    bool isNullMove = false;
};

class Search
{
public:

    Search();
    ~Search();

    void DoSearch(const Game& game, const SearchParam& param, SearchResult& result);

    void StopSearch();

    TranspositionTable& GetTranspositionTable() { return mTranspositionTable; }
    const MoveOrderer& GetMoveOrderer() const { return mMoveOrderer; }

private:

    Search(const Search&) = delete;

    struct SearchStats
    {
        uint64_t fh = 0;
        uint64_t fhf = 0;
        uint64_t nodes = 0;
        uint64_t quiescenceNodes = 0;
        uint64_t ttHits = 0;
        uint64_t ttWrites = 0;
        uint64_t tbHits = 0;
        uint32_t maxDepth = 0;
        uint64_t betaCutoffHistogram[MoveList::MaxMoves] = { 0 };
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
        SearchResult& searchResult;
        uint32_t depth;
        uint32_t pvIndex;
        SearchContext& searchContext;
        std::span<const Move> moveFilter;
        ScoreType previousScore;                  // score in previous ID iteration
    };

    std::atomic<bool> mStopSearch = false;

    // principial variation moves tracking for current search
    PackedMove pvArray[MaxSearchDepth][MaxSearchDepth];
    uint8_t pvLengths[MaxSearchDepth];

    // principial variation lines from previous iterative deepening search
    SearchResult mPrevPvLines;

    TranspositionTable mTranspositionTable;
    MoveOrderer mMoveOrderer;

    bool IsDraw(const NodeInfo& node, const Game& game) const;

    ScoreType AspirationWindowSearch(const AspirationWindowSearchParam& param);

    ScoreType QuiescenceNegaMax(NodeInfo& node, SearchContext& ctx);
    ScoreType NegaMax(NodeInfo& node, SearchContext& ctx);

    // check if one of generated moves is in PV table
    const Move FindPvMove(const NodeInfo& node, MoveList& moves) const;
    void FindTTMove(const PackedMove& ttMove, MoveList& moves) const;

    ScoreType PruneByMateDistance(const NodeInfo& node, ScoreType alpha, ScoreType beta);

    // check for repetition in the searched node
    bool IsRepetition(const NodeInfo& node, const Game& game) const;

    // update principal variation line
    void UpdatePvArray(uint32_t depth, const Move move);

    // reconstruct PV line from cache and TT table
    std::vector<Move> GetPvLine(const Position& pos, uint32_t maxLength) const;

    // returns true if the search needs to be aborted immediately
    bool CheckStopCondition(const SearchContext& ctx) const;
};