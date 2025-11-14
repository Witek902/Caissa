#pragma once

#include "Search.hpp"

enum class PreviousSearchHint
{
    Hit,
    Miss,
    Unknown,
};

struct TimeManagerInitData
{
    int32_t moveTime;
    int32_t remainingTime;
    int32_t timeIncrement;
    int32_t theirRemainingTime;
    int32_t theirTimeIncrement;
    uint32_t movesToGo;
    int32_t moveOverhead;
    PreviousSearchHint previousSearchHint = PreviousSearchHint::Unknown;
};

struct TimeManagerUpdateData
{
    uint32_t depth;
    const SearchResult& currResult;
    const SearchResult& prevResult;
    double bestMoveNodeFraction = 0.0;
};

struct TimeManagerState
{
    uint32_t stabilityCounter = 0;
};

// init time limits at the beginning of a search
void InitTimeManager(const Game& game, const TimeManagerInitData& data, SearchLimits& limits);

// update time limits after one search iteration
void UpdateTimeManager(const TimeManagerUpdateData& data, const Game& game, SearchLimits& limits, TimeManagerState& state);
