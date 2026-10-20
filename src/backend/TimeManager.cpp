#include "TimeManager.hpp"
#include "Game.hpp"
#include "Tuning.hpp"

#include <math.h>
#include <stdio.h>

DEFINE_PARAM(TM_MovesLeftMidpoint, 35, 25, 60);
DEFINE_PARAM(TM_MovesLeftSteepness, 219, 150, 260);
DEFINE_PARAM(TM_IdealTimeFactor, 823, 700, 1000);
DEFINE_PARAM(TM_MaxTimeFactor, 450, 100, 1000);
DEFINE_PARAM(TM_NodesCountScale, 208, 160, 260);
DEFINE_PARAM(TM_NodesCountOffset, 63, 10, 90);
DEFINE_PARAM(TM_StabilityScale, 58, 0, 200);
DEFINE_PARAM(TM_StabilityOffset, 1549, 1000, 2000);
DEFINE_PARAM(TM_PredictedMoveHitScale, 915, 800, 1000);
DEFINE_PARAM(TM_PredictedMoveMissScale, 1132, 1000, 1400);
DEFINE_PARAM(TM_OverheadReserveFrac, 750, 400, 800);
DEFINE_PARAM(TM_ComplexityBase, 700, 400, 1000);
DEFINE_PARAM(TM_ComplexityScale, 500, 0, 1000);
DEFINE_PARAM(TM_ComplexityCap, 250, 100, 600);
DEFINE_PARAM(TM_ClockDisadvantageScale, 750, 0, 1000);

static float EstimateMovesLeft(const uint32_t moves)
{
    // based on LeelaChessZero
    const float midpoint = static_cast<float>(TM_MovesLeftMidpoint);
    const float steepness = static_cast<float>(TM_MovesLeftSteepness) / 100.0f;
    return midpoint * powf(1.0f + 1.5f * powf((float)moves / midpoint, steepness), 1.0f / steepness) - (float)moves;
}

void InitTimeManager(const Game& game, const TimeManagerInitData& data, SearchLimits& limits)
{
    const int32_t moveOverhead = data.moveOverhead;
    const float movesLeft = data.movesToGo != UINT32_MAX ? (float)data.movesToGo : EstimateMovesLeft(game.GetPosition().GetMoveCount());

    // soft limit
    if (data.remainingTime != INT32_MAX)
    {
        const float remainingTime = (float)data.remainingTime;
        const float idealTimeFactor = static_cast<float>(TM_IdealTimeFactor) / 1000.0f;
        const float maxTimeFactor = static_cast<float>(TM_MaxTimeFactor) / 100.0f;
        float idealTime = idealTimeFactor * (remainingTime / movesLeft + (float)data.timeIncrement);
        float maxTime = maxTimeFactor * (remainingTime / movesLeft + (float)data.timeIncrement);

        const float minMoveTime = 0.00001f;
        const float timeMargin = 0.8f;

        // The hard limit must leave the communication latency on the clock, otherwise a search running
        // to the very end of it loses on time. Never reserve more than half of the clock: an almost
        // exhausted clock must still leave a budget the search can do something with.
        // The soft limit is left alone.
        const float reserveFrac = static_cast<float>(TM_OverheadReserveFrac) / 1000.0f;
        const float overheadReserve = Min((float)moveOverhead, reserveFrac * remainingTime);
        const float hardTimeLimit = Max(minMoveTime, Min(timeMargin * remainingTime, remainingTime - overheadReserve));

        maxTime = Clamp(maxTime, 0.0f, hardTimeLimit);
        idealTime = Clamp(idealTime, 0.0f, Max(minMoveTime, timeMargin * remainingTime));

        // spend less time when behind on the clock
        if (data.theirRemainingTime != INT32_MAX && data.movesToGo != 1)
        {
            const float theirRemainingTime = (float)data.theirRemainingTime;
            const float clockAdvantage = (remainingTime - theirRemainingTime) / (1.0f + remainingTime + theirRemainingTime);
            idealTime *= 1.0f + static_cast<float>(TM_ClockDisadvantageScale) / 1000.0f * Min(clockAdvantage, 0.0f);
        }

        // reduce time if opponent played a move predicted by the previous search, increase otherwise
        if (data.previousSearchHint == PreviousSearchHint::Hit)
            idealTime *= static_cast<float>(TM_PredictedMoveHitScale) / 1000.0f;
        else if (data.previousSearchHint == PreviousSearchHint::Miss)
            idealTime *= static_cast<float>(TM_PredictedMoveMissScale) / 1000.0f;

#ifndef CONFIGURATION_FINAL
        printf("info string idealTime=%gms maxTime=%gms\n", idealTime, maxTime);
#endif // CONFIGURATION_FINAL

        limits.idealTimeBase = limits.idealTimeCurrent = TimePoint::FromSeconds(0.001f * idealTime);

        // abort search if significantly exceeding ideal allocated time
        limits.maxTime = TimePoint::FromSeconds(0.001f * maxTime);

        // activate root singularity search after some portion of estimated time passed
        limits.rootSingularityTime = TimePoint::FromSeconds(0.001f * idealTime * 0.2f);
    }

    // fixed move time
    if (data.moveTime != INT32_MAX)
    {
        limits.idealTimeBase = limits.idealTimeCurrent = TimePoint::FromSeconds(0.001f * data.moveTime);
        limits.maxTime = TimePoint::FromSeconds(0.001f * data.moveTime);
    }
}

void UpdateTimeManager(const TimeManagerUpdateData& data, SearchLimits& limits, TimeManagerState& state)
{
    ASSERT(!data.currResult.Empty());
    ASSERT(!data.currResult[0].moves.Empty());

    if (!limits.idealTimeBase.IsValid() || data.prevResult.Empty() || data.prevResult[0].moves.Empty())
        return;
    
    // don't update TM at low depths
    if (data.depth < 5)
        return;

    limits.idealTimeCurrent = limits.idealTimeBase;

    // decrease time if PV move is stable
    {
        // update PV move stability counter
        if (data.prevResult[0].moves.Front() == data.currResult[0].moves.Front())
            state.stabilityCounter++;
        else
            state.stabilityCounter = 0;

        const double stabilityFactor = static_cast<double>(TM_StabilityScale) / 1000.0;
        const double stabilityOffset = static_cast<double>(TM_StabilityOffset) / 1000.0;
        const double stabilityTimeFactor = stabilityOffset - stabilityFactor * Min(10u, state.stabilityCounter);
        limits.idealTimeCurrent *= stabilityTimeFactor;
    }

    // decrease time if nodes fraction spent on best move is high
    {
        const double nonBestMoveNodeFraction = 1.0 - data.bestMoveNodeFraction;
        const double scale = static_cast<double>(TM_NodesCountScale) / 100.0;
        const double offset = static_cast<double>(TM_NodesCountOffset) / 100.0;
        const double nodeCountFactor = nonBestMoveNodeFraction * scale + offset;
        limits.idealTimeCurrent *= nodeCountFactor;
    }

    // more time when the search score disagrees with the root static eval, less otherwise (averages ~1.0)
    const ScoreType score = data.currResult[0].score;
    if (data.rootStaticEval != InvalidValue && Abs(score) < KnownWinValue)
    {
        const double cap = static_cast<double>(TM_ComplexityCap);
        const double complexity = Min(Abs(score - data.rootStaticEval) * log(static_cast<double>(data.depth)), cap);
        const double complexityFactor = static_cast<double>(TM_ComplexityBase) / 1000.0 + static_cast<double>(TM_ComplexityScale) / 1000.0 * complexity / cap;
        limits.idealTimeCurrent *= complexityFactor;
    }

#ifndef CONFIGURATION_FINAL
    printf("info string ideal time %g ms\n", limits.idealTimeCurrent.ToSeconds() * 1000.0f);
#endif // CONFIGURATION_FINAL
}
