#include "TimeManager.hpp"
#include "Evaluate.hpp"
#include "Game.hpp"
#include "Tuning.hpp"

#include <math.h>
#include <stdio.h>

DEFINE_PARAM(TM_MovesLeftMidpoint, 35, 25, 60);
DEFINE_PARAM(TM_MovesLeftSteepness, 219, 150, 260);
DEFINE_PARAM(TM_IdealTimeFactor, 823, 700, 1000);
DEFINE_PARAM(TM_MaxTimeFactor, 450, 100, 1000);
DEFINE_PARAM(TM_PredictedMoveHitScale, 915, 800, 1000);
DEFINE_PARAM(TM_PredictedMoveMissScale, 1132, 1000, 1400);
DEFINE_PARAM(TM_OverheadReserveFrac, 750, 400, 800);
DEFINE_PARAM(TM_ComplexityCap, 250, 100, 600);
DEFINE_PARAM(TM_ClockDisadvantageScale, 750, 0, 1000);

// soft limit factor exp(sum(weight * input) / 1000); weights are a logistic fit of "4x more nodes changes the
// best move" on 40K LTC positions, rescaled to the spread of the previous factors, bias set so clocked LTC
// searches spend the same total time as with the previous factors
DEFINE_PARAM(TM_ModelBias, 723, 300, 1200);
DEFINE_PARAM(TM_ModelStability, -52, -200, 50);
DEFINE_PARAM(TM_ModelNodeShare, -1033, -2500, 0);
DEFINE_PARAM(TM_ModelMoveChanges, 187, -100, 600);
DEFINE_PARAM(TM_ModelComplexity, -44, -300, 300);
DEFINE_PARAM(TM_ModelScoreFalling, 49, -300, 400);
DEFINE_PARAM(TM_ModelScoreRising, -311, -800, 200);
DEFINE_PARAM(TM_ModelLegalMoves, 149, -200, 500);
DEFINE_PARAM(TM_ModelPieces, -34, -300, 200);
DEFINE_PARAM(TM_ModelNearDraw, -246, -800, 300);
DEFINE_PARAM(TM_ModelNearDrawMargin, 15, 0, 50);
DEFINE_PARAM(TM_ModelScoreDriftCap, 500, 100, 1000);

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

    if (!limits.idealTimeBase.IsValid())
        return;

    ASSERT(data.depth <= MaxSearchDepth);
    const ScoreType score = data.currResult[0].score;
    state.iterationScores[data.depth] = score;

    if (data.prevResult.Empty() || data.prevResult[0].moves.Empty())
        return;

    // don't update TM at low depths
    if (data.depth < 5)
        return;

    if (data.prevResult[0].moves.Front() == data.currResult[0].moves.Front())
    {
        state.stabilityCounter++;
    }
    else
    {
        state.stabilityCounter = 0;
        state.bestMoveChanges++;
    }

    // disagreement between the search score and the root static eval
    double complexity = 0.0;
    if (data.rootStaticEval != InvalidValue && Abs(score) < KnownWinValue)
        complexity = Min(Abs(score - data.rootStaticEval) * log(static_cast<double>(data.depth)), static_cast<double>(TM_ComplexityCap));

    // score change since the middle of the search
    const int32_t scoreDrift = Clamp<int32_t>(score - state.iterationScores[data.depth / 2 + 1], -TM_ModelScoreDriftCap, TM_ModelScoreDriftCap);

    const bool nearDraw = Abs(NormalizeEval(score)) <= TM_ModelNearDrawMargin;

    const double z =
        static_cast<double>(TM_ModelBias) +
        static_cast<double>(TM_ModelStability) * Min(10u, state.stabilityCounter) +
        static_cast<double>(TM_ModelNodeShare) * data.bestMoveNodeFraction +
        static_cast<double>(TM_ModelMoveChanges) * log1p(static_cast<double>(state.bestMoveChanges)) +
        static_cast<double>(TM_ModelComplexity) * complexity / 100.0 +
        static_cast<double>(TM_ModelScoreFalling) * Max(-scoreDrift, 0) / 100.0 +
        static_cast<double>(TM_ModelScoreRising) * Max(scoreDrift, 0) / 100.0 +
        static_cast<double>(TM_ModelLegalMoves) * log(static_cast<double>(Max(1u, state.rootLegalMoves))) +
        static_cast<double>(TM_ModelPieces) * state.rootPieces / 10.0 +
        (nearDraw ? static_cast<double>(TM_ModelNearDraw) : 0.0);

    limits.idealTimeCurrent = limits.idealTimeBase;
    limits.idealTimeCurrent *= exp(z / 1000.0);

#ifndef CONFIGURATION_FINAL
    printf("info string ideal time %g ms\n", limits.idealTimeCurrent.ToSeconds() * 1000.0f);
#endif // CONFIGURATION_FINAL
}
