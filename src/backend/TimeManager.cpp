#include "TimeManager.hpp"

#include "Game.hpp"
#include "Evaluate.hpp"

#include <algorithm>

static float EstimateMovesLeft(const uint32_t moves)
{
    // based on LeelaChessZero 
    const float midpoint = 60.0f;
    const float steepness = 2.5f;
    return midpoint * std::pow(1.0f + 1.5f * std::pow((float)moves / midpoint, steepness), 1.0f / steepness) - (float)moves;
}

void TimeManager::Init(const Game& game, const TimeManagerInitData& data, SearchLimits& limits)
{
    const int32_t moveOverhead = data.moveOverhead;
    const float movesLeft = data.movesToGo != UINT32_MAX ? (float)data.movesToGo : EstimateMovesLeft(game.GetPosition().GetMoveCount());

    // soft limit
    if (data.remainingTime != INT32_MAX)
    {
        float factor = 1.0f;

        if (movesLeft > 1)
        {
            // scale based on number of moves (40 moves -> 100%, 80 moves -> 125%, 0 moves -> 75%)
            {
                const uint32_t numLegalMoves = game.GetPosition().GetNumLegalMoves();
                const float avgNumMoves = 40.0f;
                factor *= 0.75f + 0.25f * static_cast<float>(numLegalMoves) / avgNumMoves;
            }

            // decrease search time if in check, because it should have been extended during previous search
            if (game.GetPosition().IsInCheck())
            {
                factor *= 0.75f;
            }
        }

        const float marigin = 0.98f;
        const float increment = marigin * (float)data.timeIncrement;

        const float idealTime = std::clamp(0.85f * factor * (data.remainingTime - moveOverhead) / movesLeft + increment, 0.0f, marigin * (float)data.remainingTime);
        const float maxTime = std::clamp(factor * (data.remainingTime - moveOverhead) / sqrtf(movesLeft) + increment, 0.0f, marigin * (float)data.remainingTime);

#ifndef CONFIGURATION_FINAL
        std::cout << "info string idealTime=" << idealTime << "ms maxTime=" << maxTime << "ms" << std::endl;
#endif // CONFIGURATION_FINAL

        limits.idealTime = limits.startTimePoint + TimePoint::FromSeconds(0.001f * idealTime);

        // abort search if significantly exceeding ideal allocated time
        limits.maxTime = limits.startTimePoint + TimePoint::FromSeconds(0.001f * maxTime);

        // activate root singularity search after some portion of estimated time passed
        limits.rootSingularityTime = limits.startTimePoint + TimePoint::FromSeconds(0.001f * idealTime * 0.2f);
    }

    // hard limit
    int32_t hardLimitMs = std::min(data.remainingTime, data.moveTime);
    if (hardLimitMs != INT32_MAX)
    {
        hardLimitMs = std::max(0, hardLimitMs - moveOverhead);
        const TimePoint hardLimitTimePoint = limits.startTimePoint + TimePoint::FromSeconds(hardLimitMs * 0.001f);

        if (!limits.maxTime.IsValid() ||
            limits.maxTime >= hardLimitTimePoint)
        {
            limits.maxTime = hardLimitTimePoint;
        }

#ifndef CONFIGURATION_FINAL
        std::cout << "info string hardLimitTime=" << hardLimitMs << "ms" << std::endl;
#endif // CONFIGURATION_FINAL
    }
}

void TimeManager::Update(const Game& game, const TimeManagerUpdateData& data, TimePoint& maxTimeSoft)
{
    const uint32_t startDepth = 5;

    ASSERT(!data.currResult.empty());
    ASSERT(!data.currResult[0].moves.empty());

    if (!maxTimeSoft.IsValid() || data.prevResult.empty() || data.prevResult[0].moves.empty())
    {
        return;
    }
    
    // don't update TM at low depths
    if (data.depth < startDepth)
    {
        return;
    }

    const int32_t prevScore = data.depth > startDepth ? data.prevResult[0].score : 0;
    const int32_t currScore = data.currResult[0].score;
    const Move currMove = data.currResult[0].moves[0];

    TimePoint t = maxTimeSoft - data.limits.startTimePoint;

    if (data.depth == startDepth)
    {
        const int32_t goodScoreTreshold = 300;

        // reduce time on recapture
        if (std::abs(currScore) < goodScoreTreshold &&
            currMove.IsCapture() &&
            game.GetPosition().StaticExchangeEvaluation(currMove, 100))
        {
            const int32_t staticRootEval = Evaluate(game.GetPosition()) * ColorMultiplier(game.GetSideToMove());
            if (currScore > staticRootEval + 500) t *= 0.5;
            if (currScore > staticRootEval + 250) t *= 0.75;
        }

        // reduce time on good position
        if (currScore > goodScoreTreshold) t *= 0.5 + 0.5 * pow(2.0, -(currScore - goodScoreTreshold) / 100.0);

        // reduce time more on winning position
        if (currScore > KnownWinValue) t *= 0.5;
    }

    // increase time if score dropped
    if (currScore < prevScore) t *= pow(2.0, std::clamp(prevScore - currScore, -1000, 1000) / 1000.0);

    // decrease time if score jumped
    if (prevScore < currScore) t *= pow(2.0, std::clamp(prevScore - currScore, -1000, 1000) / 2000.0);

    // increase time if PV line changes
    {
        const size_t pvLength = std::min(data.prevResult[0].moves.size(), data.currResult[0].moves.size());
        for (size_t i = 0; i < std::min<size_t>(pvLength, 8); ++i)
        {
            if (data.prevResult[0].moves[i] != data.currResult[0].moves[i])
            {
                t *= 1.0 + 0.075 / (1 + i);
                break;
            }
        }
    }

    const TimePoint newMaxTimeSoft = data.limits.startTimePoint + t;

    if (newMaxTimeSoft != maxTimeSoft)
    {
#ifndef CONFIGURATION_FINAL
        std::cout << "info string ideal time " << t.ToSeconds() * 1000.0f << " ms" << std::endl;
#endif // CONFIGURATION_FINAL

        maxTimeSoft = newMaxTimeSoft;
    }
}