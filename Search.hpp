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

    ScoreType DoSearch(const Position& position, Move& outBestMove);

    static ScoreType Evaluate(const Position& position);

private:

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
        Move move;
        int32_t score;
        uint32_t depth;
    };

    std::unordered_map<uint64_t, PvTableEntry> pvTable;

    uint64_t searchHistory[2][6][64];

    ScoreType QuiescenceNegaMax(const NegaMaxParam& param, SearchContext& ctx);
    ScoreType NegaMax(const NegaMaxParam& param, SearchContext& ctx, Move* outBestMove = nullptr);

    // check if one of generated moves is in PV table
    void FindPvMove(const uint64_t positionHash, MoveList& moves);
    void FindHistoryMoves(Color color, MoveList& moves);

    static bool IsRepetition(const NegaMaxParam& param);

    void UpdatePvEntry(uint32_t depth, uint64_t positionHash, const Move move, int32_t score);
};