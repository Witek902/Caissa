#pragma once

#include "Position.hpp"
#include "Move.hpp"
#include "TranspositionTable.hpp"

#include <unordered_map>
#include <span>

struct SearchParam
{
    uint32_t maxDepth = 8;
    uint32_t numPvLines = 1;
    std::vector<Move> rootMoves;
    bool debugLog = true;
};

struct PvLine
{
    std::vector<Move> moves;
    int32_t score = 0;
};

using SearchResult = std::vector<PvLine>;

class Search
{
public:

    using ScoreType = int32_t;
    static constexpr int32_t CheckmateValue = 100000;
    static constexpr int32_t InfValue       = 10000000;
    static constexpr int32_t InvalidValue   = 9999999;

    static constexpr int32_t MaxSearchDepth = 64;

    Search();

    void DoSearch(const Position& position, const SearchParam& param, SearchResult& result);

    void RecordBoardPosition(const Position& position);

    // check if a position was repeated 2 times
    bool IsPositionRepeated(const Position& position, uint32_t repetitionCount = 2u) const;

    TranspositionTable& GetTranspositionTable() { return mTranspositionTable; }

private:

    Search(const Search&) = delete;

    struct NodeInfo
    {
        const Position* position = nullptr;
        const NodeInfo* parentNode = nullptr;
        ScoreType alpha;
        ScoreType beta;
        std::span<const Move> moveFilter; // ignore given moves in search, used for multi-PV search
        std::span<const Move> rootMoves;  // consider only this moves at root node, used for "searchmoves" UCI command
        uint16_t depth;
        uint16_t maxDepth;
        uint8_t pvIndex;
        Color color;
        bool isPvNode = false;
    };
    
    struct SearchContext
    {
        uint64_t fh = 0;
        uint64_t fhf = 0;
        uint64_t nodes = 0;
        uint64_t quiescenceNodes = 0;
        uint64_t pseudoMovesPerNode = 0;
        uint64_t ttHits = 0;
        uint32_t maxDepth = 0;
    };

    struct PvLineEntry
    {
        uint64_t positionHash;
        Move move;
    };

    struct GameHistoryPosition
    {
        Position pos;           // board position
        uint32_t count = 0;     // how many times it occurred during the game
    };

    std::atomic<bool> mStopSearch = false;

    using GameHistoryPositionEntry = std::vector<GameHistoryPosition>;

    // principial variation moves tracking for current search
    PackedMove pvArray[MaxSearchDepth][MaxSearchDepth];
    uint16_t pvLengths[MaxSearchDepth];

    // principial variation lines from previous iterative deepening search
    SearchResult mPrevPvLines;

    TranspositionTable mTranspositionTable;

    uint32_t searchHistory[2][6][64];

    static constexpr uint32_t NumKillerMoves = 4;
    PackedMove killerMoves[MaxSearchDepth][NumKillerMoves];

    std::unordered_map<uint64_t, GameHistoryPositionEntry> historyGamePositions;

    ScoreType QuiescenceNegaMax(const NodeInfo& node, SearchContext& ctx);
    ScoreType NegaMax(const NodeInfo& node, SearchContext& ctx);

    // check if one of generated moves is in PV table
    const Move FindPvMove(const NodeInfo& node, MoveList& moves) const;
    void FindHistoryMoves(Color color, MoveList& moves) const;
    void FindKillerMoves(uint32_t depth, MoveList& moves) const;

    int32_t PruneByMateDistance(const NodeInfo& node, int32_t alpha, int32_t beta);

    // check for repetition in the searched node
    bool IsRepetition(const NodeInfo& node) const;

    // update principal variation line
    void UpdatePvArray(uint32_t depth, const Move move);

    void UpdateSearchHistory(const NodeInfo& node, const Move move);
    void RegisterKillerMove(const NodeInfo& node, const Move move);
};