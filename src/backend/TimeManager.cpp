#include "TimeManager.hpp"
#include "Game.hpp"
#include "Tuning.hpp"


DEFINE_PARAM(TM_MovesLeftMidpoint, 36, 30, 60);
DEFINE_PARAM(TM_MovesLeftSteepness, 210, 150, 260);
DEFINE_PARAM(TM_IdealTimeFactor, 823, 700, 1000);
DEFINE_PARAM(TM_MaxTimeFactor, 493, 100, 1000);
DEFINE_PARAM(TM_NodesCountScale, 205, 160, 260);
DEFINE_PARAM(TM_NodesCountOffset, 63, 10, 90);

DEFINE_PARAM(TM_StabilityFactor0, 1520, 0, 200);
DEFINE_PARAM(TM_StabilityFactor1, 1459, 0, 200);
DEFINE_PARAM(TM_StabilityFactor2, 1398, 0, 200);
DEFINE_PARAM(TM_StabilityFactor3, 1337, 0, 200);
DEFINE_PARAM(TM_StabilityFactor4, 1276, 0, 200);
DEFINE_PARAM(TM_StabilityFactor5, 1215, 0, 200);
DEFINE_PARAM(TM_StabilityFactor6, 1154, 0, 200);
DEFINE_PARAM(TM_StabilityFactor7, 1093, 0, 200);

DEFINE_PARAM(TM_PredictedMoveHitScale, 915, 800, 1000);
DEFINE_PARAM(TM_PredictedMoveMissScale, 1128, 1000, 1400);

static float EstimateMovesLeft(const uint32_t moves)
{
    // based on LeelaChessZero
    const float midpoint = static_cast<float>(TM_MovesLeftMidpoint);
    const float steepness = static_cast<float>(TM_MovesLeftSteepness) / 100.0f;
    return midpoint * std::pow(1.0f + 1.5f * std::pow((float)moves / midpoint, steepness), 1.0f / steepness) - (float)moves;
}

void InitTimeManager(const Game& game, const TimeManagerInitData& data, SearchLimits& limits)
{
    const int32_t moveOverhead = data.moveOverhead;
    const float movesLeft = data.movesToGo != UINT32_MAX ? (float)data.movesToGo : EstimateMovesLeft(game.GetPosition().GetMoveCount());

    // soft limit
    if (data.remainingTime != INT32_MAX)
    {
        const float idealTimeFactor = static_cast<float>(TM_IdealTimeFactor) / 1000.0f;
        const float maxTimeFactor = static_cast<float>(TM_MaxTimeFactor) / 100.0f;
        float idealTime = idealTimeFactor * (data.remainingTime / movesLeft + (float)data.timeIncrement);
        float maxTime = maxTimeFactor * ((data.remainingTime - moveOverhead) / movesLeft + (float)data.timeIncrement);

        const float minMoveTime = 0.00001f;
        const float timeMargin = 0.75f; // don't spend more than 75% of remaining time on a single move
        maxTime = std::clamp(maxTime, 0.0f, std::max(minMoveTime, timeMargin * (float)data.remainingTime));
        idealTime = std::clamp(idealTime, 0.0f, std::max(minMoveTime, timeMargin * (float)data.remainingTime));

        // reduce time if opponent played a move predicted by the previous search, increase otherwise
        if (data.previousSearchHint == PreviousSearchHint::Hit)
            idealTime *= static_cast<float>(TM_PredictedMoveHitScale) / 1000.0f;
        else if (data.previousSearchHint == PreviousSearchHint::Miss)
            idealTime *= static_cast<float>(TM_PredictedMoveMissScale) / 1000.0f;

#ifndef CONFIGURATION_FINAL
        std::cout << "info string idealTime=" << idealTime << "ms maxTime=" << maxTime << "ms" << std::endl;
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
    ASSERT(!data.currResult.empty());
    ASSERT(!data.currResult[0].moves.empty());

    if (!limits.idealTimeBase.IsValid() || data.prevResult.empty() || data.prevResult[0].moves.empty())
        return;
    
    // don't update TM at low depths
    if (data.depth < 5)
        return;

    limits.idealTimeCurrent = limits.idealTimeBase;

    // decrease time if PV move is stable
    {
        // update PV move stability counter
        if (data.prevResult[0].moves.front() == data.currResult[0].moves.front())
            state.stabilityCounter++;
        else
            state.stabilityCounter = 0;

        const double stabilityFactors[] =
        {
            static_cast<double>(TM_StabilityFactor0) / 1000.0,
            static_cast<double>(TM_StabilityFactor1) / 1000.0,
            static_cast<double>(TM_StabilityFactor2) / 1000.0,
            static_cast<double>(TM_StabilityFactor3) / 1000.0,
            static_cast<double>(TM_StabilityFactor4) / 1000.0,
            static_cast<double>(TM_StabilityFactor5) / 1000.0,
            static_cast<double>(TM_StabilityFactor6) / 1000.0,
            static_cast<double>(TM_StabilityFactor7) / 1000.0,
        };

        limits.idealTimeCurrent *= stabilityFactors[std::min(7u, state.stabilityCounter)];
    }

    // decrease time if nodes fraction spent on best move is high
    {
        const double nonBestMoveNodeFraction = 1.0 - data.bestMoveNodeFraction;
        const double scale = static_cast<double>(TM_NodesCountScale) / 100.0;
        const double offset = static_cast<double>(TM_NodesCountOffset) / 100.0;
        const double nodeCountFactor = nonBestMoveNodeFraction * scale + offset;
        limits.idealTimeCurrent *= nodeCountFactor;
    }

#ifndef CONFIGURATION_FINAL
    std::cout << "info string ideal time " << limits.idealTimeCurrent.ToSeconds() * 1000.0f << " ms" << std::endl;
#endif // CONFIGURATION_FINAL
}
