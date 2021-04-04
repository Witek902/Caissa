#pragma once

#include "Position.hpp"
#include "Move.hpp"

#include <unordered_map>

class Search
{
public:

    using ScoreType = int32_t;
    static constexpr int32_t CheckmateValue = -1000000;
    static constexpr int32_t InfValue       = 10000000;

    static constexpr int32_t MaxSearchDepth = 64;

    Search();

    ScoreType DoSearch(const Position& position, Move& outBestMove);

private:

    Search(const Search&) = delete;

    struct NegaMaxParam
    {
        const Position* position = nullptr;
        const NegaMaxParam* parentParam = nullptr;
        uint64_t positionHash = 0;
        uint16_t depth;
        uint16_t maxDepth;
        ScoreType alpha;
        ScoreType beta;
        Color color;
    };
    
    struct SearchContext
    {
        uint64_t fh = 0;
        uint64_t fhf = 0;
        uint64_t nodes = 0;
        uint64_t quiescenceNodes = 0;
        Move moves[MaxSearchDepth];
    };

    struct PvTableEntry
    {
        uint64_t positionHash = 0;
        Move move;
    };

    // TODO adjust size depending on depth?
    static constexpr uint32_t PvTableSize = 4 * 1024 * 1024;
    std::vector<PvTableEntry> pvTable;

    uint64_t searchHistory[2][6][64];

    static constexpr uint32_t NumKillerMoves = 3;
    Move killerMoves[MaxSearchDepth][NumKillerMoves];

    ScoreType QuiescenceNegaMax(const NegaMaxParam& param, SearchContext& ctx);
    ScoreType NegaMax(const NegaMaxParam& param, SearchContext& ctx, Move* outBestMove = nullptr);

    // check if one of generated moves is in PV table
    void FindPvMove(const uint64_t positionHash, MoveList& moves) const;
    void FindHistoryMoves(Color color, MoveList& moves) const;
    void FindKillerMoves(uint32_t depth, MoveList& moves) const;

    static bool IsRepetition(const NegaMaxParam& param);

    void UpdatePvEntry(uint64_t positionHash, const Move move);
};