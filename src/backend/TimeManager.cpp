#include "TimeManager.hpp"

#include "Game.hpp"
#include "Evaluate.hpp"
#include "Tuning.hpp"

#include <algorithm>

void TimeManager::Init(const Game& game, const TimeManagerInitData& data, SearchLimits& limits)
{
    UNUSED(game);

    const int32_t moveOverhead = data.moveOverhead;

    // soft limit
    if (data.remainingTime != INT32_MAX)
    {
        float idealTime, maxTime;

        if (data.movesToGo != UINT32_MAX) // "remainingTime / movesToGo + increment" time control
        {
            idealTime = 2.0f * (data.remainingTime - moveOverhead) / (data.movesToGo + 5) + (float)data.timeIncrement;
            maxTime = 10.0f * (data.remainingTime - moveOverhead) / (data.movesToGo + 10) + (float)data.timeIncrement;
        }
        else // "remainingTime + increment" time control
        {
            idealTime = (data.remainingTime - moveOverhead) / 20.0f + (float)data.timeIncrement / 2.0f;
            maxTime = (data.remainingTime - moveOverhead) / 5.0f + (float)data.timeIncrement / 2.0f;
        }

        // clamp to max remaining time
        idealTime = std::clamp(idealTime, 0.0f, (float)data.remainingTime - moveOverhead);
        maxTime = std::clamp(maxTime, 0.0f, (float)data.remainingTime - moveOverhead);

#ifndef CONFIGURATION_FINAL
        std::cout << "info string idealTime=" << idealTime << "ms maxTime=" << maxTime << "ms" << std::endl;
#endif // CONFIGURATION_FINAL

        limits.idealTime = TimePoint::FromSeconds(0.001f * idealTime);
        limits.maxTime = TimePoint::FromSeconds(0.001f * maxTime);

        // activate root singularity search after some portion of estimated time passed
        limits.rootSingularityTime = TimePoint::FromSeconds(0.001f * idealTime * 0.2f);
    }

    // fixed move time
    if (data.moveTime != INT32_MAX)
    {
        limits.idealTime = TimePoint::FromSeconds(0.001f * data.moveTime);
        limits.maxTime = TimePoint::FromSeconds(0.001f * data.moveTime);
    }
}

void TimeManager::Update(const Game& game, const TimeManagerUpdateData& data, SearchLimits& limits)
{
    const uint32_t startDepth = 5;

    ASSERT(!data.currResult.empty());
    ASSERT(!data.currResult[0].moves.empty());

    if (!limits.idealTime.IsValid() || data.prevResult.empty() || data.prevResult[0].moves.empty())
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

    TimePoint t = limits.idealTime;

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

    if (t != limits.idealTime)
    {
#ifndef CONFIGURATION_FINAL
        std::cout << "info string ideal time " << t.ToSeconds() * 1000.0f << " ms" << std::endl;
#endif // CONFIGURATION_FINAL

        limits.idealTime = t;
    }
}