#pragma once

#include "Search.hpp"

struct TimeManagerInitData
{
    int32_t moveTime;

    int32_t remainingTime;
    int32_t timeIncrement;
    int32_t theirRemainingTime;
    int32_t theirTimeIncrement;

    uint32_t movesToGo;

    int32_t moveOverhead;
};

struct TimeManagerUpdateData
{
    uint32_t depth = 0;
    const SearchResult& currResult;
    const SearchResult& prevResult;
    float bestMoveNodeFraction = 0.0f;
};

class TimeManager
{
public:
    // init time limits at the beginning of a search
    static void Init(const Game& game, const TimeManagerInitData& data, SearchLimits& limits);

    // update time limits after one search iteration
    static void Update(const Game& game, const TimeManagerUpdateData& data, SearchLimits& limits);
};
