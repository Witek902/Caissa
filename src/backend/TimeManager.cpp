#include "TimeManager.hpp"

#include "Game.hpp"
#include "Evaluate.hpp"
#include "Tuning.hpp"

#include <algorithm>

DEFINE_PARAM(TM_MovesLeftMidpoint, 41, 30, 60);
DEFINE_PARAM(TM_MovesLeftSteepness, 213, 150, 260);
DEFINE_PARAM(TM_IdealTimeFactor, 830, 700, 1000);
DEFINE_PARAM(TM_NodesCountScale, 199, 160, 260);
DEFINE_PARAM(TM_NodesCountOffset, 53, 10, 90);
DEFINE_PARAM(TM_StabilityScale, 37, 0, 200);
DEFINE_PARAM(TM_StabilityOffset, 1254, 1000, 2000);

DEFINE_PARAM(TM_OurPawnFactor, -2, -50, 50);
DEFINE_PARAM(TM_OurKnightFactor, 0, -50, 50);
DEFINE_PARAM(TM_OurBishopFactor, 5, -50, 50);
DEFINE_PARAM(TM_OurRookFactor, 1, -50, 50);
DEFINE_PARAM(TM_OurQueenFactor, 3, -50, 50);
DEFINE_PARAM(TM_TheirPawnFactor, -2, -50, 50);
DEFINE_PARAM(TM_TheirKnightFactor, -8, -50, 50);
DEFINE_PARAM(TM_TheirBishopFactor, -4, -50, 50);
DEFINE_PARAM(TM_TheirRookFactor, -5, -50, 50);
DEFINE_PARAM(TM_TheirQueenFactor, 12, -50, 50);

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
        float idealTime = idealTimeFactor * (data.remainingTime / movesLeft + (float)data.timeIncrement);
        float maxTime = (data.remainingTime - moveOverhead) / sqrtf(movesLeft) + (float)data.timeIncrement;

        const float minMoveTime = 0.00001f;
        const float timeMargin = 0.5f;
        maxTime = std::clamp(maxTime, 0.0f, std::max(minMoveTime, timeMargin * (float)data.remainingTime - moveOverhead));
        idealTime = std::clamp(idealTime, 0.0f, std::max(minMoveTime, timeMargin * (float)data.remainingTime - moveOverhead));

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

void UpdateTimeManager(const Game& game, const TimeManagerUpdateData& data, SearchLimits& limits, TimeManagerState& state)
{
    ASSERT(!data.currResult.empty());
    ASSERT(!data.currResult[0].moves.empty());

    const Position& pos = game.GetPosition();
    const Move bestMove = data.currResult[0].moves.front();

    if (!limits.idealTimeBase.IsValid() || data.prevResult.empty() || data.prevResult[0].moves.empty())
        return;
    
    // don't update TM at low depths
    if (data.depth < 5)
        return;

    limits.idealTimeCurrent = limits.idealTimeBase;

    // decrease time if PV move is stable
    {
        // update PV move stability counter
        if (data.prevResult[0].moves.front() == bestMove)
            state.stabilityCounter++;
        else
            state.stabilityCounter = 0;

        const double stabilityFactor = static_cast<double>(TM_StabilityScale) / 1000.0;
        const double stabilityOffset = static_cast<double>(TM_StabilityOffset) / 1000.0;
        const double stabilityTimeFactor = stabilityOffset - stabilityFactor * std::min(10u, state.stabilityCounter);
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

    // adjust time based on material
    {
        const double ourPawnFactor = static_cast<double>(TM_OurPawnFactor) / 100.0;
        const double ourKnightFactor = static_cast<double>(TM_OurKnightFactor) / 100.0;
        const double ourBishopFactor = static_cast<double>(TM_OurBishopFactor) / 100.0;
        const double ourRookFactor = static_cast<double>(TM_OurRookFactor) / 100.0;
        const double ourQueenFactor = static_cast<double>(TM_OurQueenFactor) / 100.0;
        const double theirPawnFactor = static_cast<double>(TM_TheirPawnFactor) / 100.0;
        const double theirKnightFactor = static_cast<double>(TM_TheirKnightFactor) / 100.0;
        const double theirBishopFactor = static_cast<double>(TM_TheirBishopFactor) / 100.0;
        const double theirRookFactor = static_cast<double>(TM_TheirRookFactor) / 100.0;
        const double theirQueenFactor = static_cast<double>(TM_TheirQueenFactor) / 100.0;

        const double maxMaterialFactor =
            8 * (ourPawnFactor + theirPawnFactor) +
            2 * (ourKnightFactor + theirKnightFactor) +
            2 * (ourBishopFactor + theirBishopFactor) +
            2 * (ourRookFactor + theirRookFactor) +
            1 * (ourQueenFactor + theirQueenFactor);

        const auto& us = pos.GetCurrentSide();
        const auto& them = pos.GetOpponentSide();

        const double materialFactor =
            us.pawns.Count() * ourPawnFactor + us.knights.Count() * ourKnightFactor + us.bishops.Count() * ourBishopFactor + us.rooks.Count() * ourRookFactor + us.queens.Count() * ourQueenFactor +
            them.pawns.Count() * theirPawnFactor + them.knights.Count() * theirKnightFactor + them.bishops.Count() * theirBishopFactor + them.rooks.Count() * theirRookFactor + them.queens.Count() * theirQueenFactor;

        limits.idealTimeCurrent *= std::clamp(1.0 + (materialFactor - maxMaterialFactor / 2.0) / 42.0, 0.5, 2.0);

        std::cout << "info string material factor " << std::clamp(1.0 + (materialFactor - maxMaterialFactor / 2.0) / 42.0, 0.5, 2.0) << std::endl;
    }

#ifndef CONFIGURATION_FINAL
    std::cout << "info string ideal time " << limits.idealTimeCurrent.ToSeconds() * 1000.0f << " ms" << std::endl;
#endif // CONFIGURATION_FINAL
}
