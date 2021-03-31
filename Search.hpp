#pragma once

#include "Position.hpp"
#include "Move.hpp"

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
        uint16_t depth;
        uint16_t maxDepth;
        ScoreType alpha;
        ScoreType beta;
        Color color;
    };
    
    struct SearchContext
    {
        uint64_t nodes = 0;
        uint64_t quiescenceNodes = 0;
        Move moves[MaxSearchDepth];
    };

    ScoreType QuiescenceNegaMax(const Position& position, const NegaMaxParam& param, SearchContext& ctx);
    ScoreType NegaMax(const Position& position, const NegaMaxParam& param, SearchContext& ctx, Move& outBestMove);
};