#include "Search.hpp"
#include "SearchUtils.hpp"
#include "MovePicker.hpp"
#include "Game.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"
#include "TranspositionTable.hpp"
#include "Tablebase.hpp"
#include "TimeManager.hpp"
#include "PositionHash.hpp"

#ifdef USE_TABLE_BASES
    #include "tablebase/tbprobe.h"
#endif // USE_TABLE_BASES

#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <thread>
#include <math.h>

static const float CurrentMoveReportDelay = 10.0f;

static const uint8_t SingularitySearchPvIndex = UINT8_MAX;
static const int32_t SingularitySearchMinDepth = 8;
static const int32_t SingularitySearchScoreTresholdMin = 200;
static const int32_t SingularitySearchScoreTresholdMax = 400;
static const int32_t SingularitySearchScoreStep = 25;

static const uint32_t DefaultMaxPvLineLength = 20;
static const uint32_t MateCountStopCondition = 5;

static const int32_t TablebaseProbeDepth = 4;

static const int32_t NullMoveReductionsStartDepth = 2;
static const int32_t NullMoveReductions_NullMoveDepthReduction = 4;
static const int32_t NullMoveReductions_ReSearchDepthReduction = 4;

static const int32_t LateMoveReductionStartDepth = 3;

static const int32_t AspirationWindowMax = 80;
static const int32_t AspirationWindowMin = 25;
static const int32_t AspirationWindowStep = 5;

static const int32_t SingularExtensionScoreMarigin = 10;

static const int32_t BetaPruningDepth = 8;
static const int32_t BetaMarginMultiplier = 80;
static const int32_t BetaMarginBias = 10;

static const int32_t QSearchSeeMargin = -50;

static const int32_t HistoryPruningScoreBase = 0;

INLINE static uint32_t GetLateMovePruningTreshold(uint32_t depth)
{
    return 3 + depth + depth * depth / 2;
}

INLINE static int32_t GetHistoryPruningTreshold(int32_t depth)
{
    return HistoryPruningScoreBase - 256 * depth - 64 * depth * depth;
}

void Search::Stats::Append(ThreadStats& threadStats, bool flush)
{
    if (threadStats.nodes >= 64 || flush)
    {
        nodes += threadStats.nodes;
        quiescenceNodes += threadStats.quiescenceNodes;
        AtomicMax(maxDepth, threadStats.maxDepth);

        threadStats = ThreadStats{};
    }
}

Search::Search()
{
    BuildMoveReductionTable();
    mThreadData.resize(1);
}

Search::~Search()
{
}

void Search::BuildMoveReductionTable()
{
    for (int32_t depth = 0; depth < MaxSearchDepth; ++depth)
    {
        for (uint32_t moveIndex = 0; moveIndex < MaxReducedMoves; ++moveIndex)
        {
            const int32_t reduction = int32_t(0.5f * logf(float(depth + 1)) * logf(float(moveIndex + 1)));

            ASSERT(reduction <= 64);
            mMoveReductionTable[depth][moveIndex] = (uint8_t)std::min<int32_t>(reduction, UINT8_MAX);
        }
    }
}

void Search::Clear()
{
    for (ThreadData& threadData : mThreadData)
    {
        threadData.moveOrderer.Clear();
    }
}

const MoveOrderer& Search::GetMoveOrderer() const
{
    return mThreadData.front().moveOrderer;
}

void Search::StopSearch()
{
    mStopSearch = true;
}

NO_INLINE bool Search::CheckStopCondition(const ThreadData& thread, const SearchContext& ctx, bool isRootNode) const
{
    if (mStopSearch.load(std::memory_order_relaxed))
    {
        return true;
    }

    if (!ctx.searchParam.isPonder)
    {
        if (ctx.searchParam.limits.maxNodes < UINT64_MAX &&
            ctx.stats.nodes >= ctx.searchParam.limits.maxNodes)
        {
            // nodes limit exceeded
            mStopSearch = true;
            return true;
        }

        // check inner nodes periodically
        if (isRootNode || (thread.stats.nodes % 256 == 0))
        {
            if (ctx.searchParam.limits.maxTime.IsValid() &&
                TimePoint::GetCurrent() >= ctx.searchParam.limits.maxTime)
            {
                // time limit exceeded
                mStopSearch = true;
                return true;
            }
        }
    }

    return false;
}

void Search::DoSearch(const Game& game, const SearchParam& param, SearchResult& outResult)
{
    outResult.clear();

    if (param.limits.maxDepth == 0 || !game.GetPosition().IsValid())
    {
        return;
    }

    mStopSearch = false;

    // clamp number of PV lines (there can't be more than number of max moves)
    static_assert(MoveList::MaxMoves <= UINT8_MAX, "Max move count must fit uint8");
    std::vector<Move> legalMoves;
    const uint32_t numLegalMoves = game.GetPosition().GetNumLegalMoves(&legalMoves);
    const uint32_t numPvLines = std::min(param.numPvLines, numLegalMoves);

    outResult.resize(numPvLines);

    if (numPvLines == 0u)
    {
        // early exit in case of no legal moves
        if (param.debugLog)
        {
            if (!game.GetPosition().IsInCheck(game.GetPosition().GetSideToMove()))
            {
                std::cout << "info depth 0 score cp 0" << std::endl;
            }
            if (game.GetPosition().IsInCheck(game.GetPosition().GetSideToMove()))
            {
                std::cout << "info depth 0 score mate 0" << std::endl;
            }
        }
        return;
    }

    if (!param.limits.analysisMode)
    {
        // if we have time limit and there's only a single legal move, return it immediately without evaluation
        if (param.limits.maxTime.IsValid() && numLegalMoves == 1)
        {
            outResult.front().moves.push_back(legalMoves.front());
            outResult.front().score = 0;
            return;
        }

        // try returning tablebase move immediately
        if (numPvLines == 1)
        {
            Move tbMove;
            if (ProbeTablebase_Root(game.GetPosition(), tbMove))
            {
                ASSERT(tbMove.IsValid());
                outResult.front().moves.push_back(tbMove);
                outResult.front().score = 0;
                return;
            }
        }
    }

    Stats globalStats;

    mThreadData.resize(param.numThreads);
    mThreadData[0].isMainThread = true;

    if (param.numThreads > 1)
    {
        std::vector<std::thread> threads;
        threads.reserve(param.numThreads);

        for (uint32_t i = param.numThreads; i-- > 0; )
        {
            // NOTE: can't capture everything by reference, because lambda is running in a thread
            threads.emplace_back([this, i, numPvLines, &game, &param, &globalStats, &outResult]() INLINE_LAMBDA
            {
                Search_Internal(i, numPvLines, game, param, globalStats, outResult);
            });
        }

        for (uint32_t threadID = 0; threadID < param.numThreads; ++threadID)
        {
            threads[threadID].join();
        }
    }
    else
    {
        Search_Internal(0, numPvLines, game, param, globalStats, outResult);
    }
}

void Search::ReportPV(const AspirationWindowSearchParam& param, const PvLine& pvLine, BoundsType boundsType, const TimePoint& searchTime) const
{
    std::stringstream ss{ std::ios_base::out };

    ss << "info depth " << param.depth;
    ss << " seldepth " << (uint32_t)param.searchContext.stats.maxDepth;
    if (param.searchParam.numPvLines > 1)
    {
        ss << " multipv " << (param.pvIndex + 1);
    }

    if (pvLine.score > CheckmateValue - (int32_t)MaxSearchDepth)
    {
        ss << " score mate " << (CheckmateValue - pvLine.score + 1) / 2;
    }
    else if (pvLine.score < -CheckmateValue + (int32_t)MaxSearchDepth)
    {
        ss << " score mate -" << (CheckmateValue + pvLine.score + 1) / 2;
    }
    else
    {
        ss << " score cp " << pvLine.score;
    }

    if (boundsType == BoundsType::LowerBound)
    {
        ss << " lowerbound";
    }
    if (boundsType == BoundsType::UpperBound)
    {
        ss << " upperbound";
    }

    const float timeInSeconds = searchTime.ToSeconds();
    const uint64_t numNodes = param.searchContext.stats.nodes.load();

    ss << " nodes " << numNodes;

    if (timeInSeconds > 0.01f && numNodes > 100)
    {
        ss << " nps " << (int64_t)((double)numNodes / (double)timeInSeconds);
    }

#ifdef COLLECT_SEARCH_STATS
    if (param.searchContext.stats.tbHits)
    {
        ss << " tbhit " << param.searchContext.stats.tbHits;
    }
#endif // COLLECT_SEARCH_STATS

    ss << " time " << static_cast<int64_t>(0.5f + 1000.0f * timeInSeconds);

    ss << " pv ";
    {
        Position tempPosition = param.position;
        for (size_t i = 0; i < pvLine.moves.size(); ++i)
        {
            const Move move = pvLine.moves[i];
            ASSERT(move.IsValid());

            if (i == 0 && param.searchParam.colorConsoleOutput) ss << "\033[93m";

            ss << tempPosition.MoveToString(move, param.searchParam.moveNotation);

            if (i == 0 && param.searchParam.colorConsoleOutput) ss << "\033[0m";

            if (i + 1 < pvLine.moves.size()) ss << ' ';
            tempPosition.DoMove(move);
        }
    }

#ifdef COLLECT_SEARCH_STATS
    if (param.searchParam.verboseStats)
    {
        {
            uint32_t maxMoveIndex = 0;
            uint64_t sum = 0;
            double average = 0.0;
            for (uint32_t i = 0; i < MoveList::MaxMoves; ++i)
            {
                if (param.searchContext.stats.betaCutoffHistogram[i])
                {
                    sum += param.searchContext.stats.betaCutoffHistogram[i];
                    average += (double)i * (double)param.searchContext.stats.betaCutoffHistogram[i];
                    maxMoveIndex = std::max(maxMoveIndex, i);
                }
            }
            average /= sum;
            printf("Average cutoff move index: %.3f\n", average);

            printf("Beta cutoff histogram\n");
            for (uint32_t i = 0; i <= maxMoveIndex; ++i)
            {
                const uint64_t value = param.searchContext.stats.betaCutoffHistogram[i];
                printf("    %u : %" PRIu64 " (%.2f%%)\n", i, value, 100.0f * float(value) / float(sum));
            }
        }


        {
            printf("Eval value histogram\n");
            for (uint32_t i = 0; i < Stats::EvalHistogramBins; ++i)
            {
                const int32_t lowEval = -Stats::EvalHistogramMaxValue + i * 2 * Stats::EvalHistogramMaxValue / Stats::EvalHistogramBins;
                const int32_t highEval = lowEval + 2 * Stats::EvalHistogramMaxValue / Stats::EvalHistogramBins;
                const uint64_t value = param.searchContext.stats.evalHistogram[i];

                printf("    %4d...%4d %llu\n", lowEval, highEval, value);
            }
        }
    }
#endif // COLLECT_SEARCH_STATS

    std::cout << std::move(ss.str()) << std::endl;
}

void Search::ReportCurrentMove(const Move& move, int32_t depth, uint32_t moveNumber) const
{
    std::cout
        << "info depth " << depth
        << " currmove " << move.ToString()
        << " currmovenumber " << moveNumber
        << std::endl;
}

static bool IsMate(const ScoreType score)
{
    return score > CheckmateValue - (int32_t)MaxSearchDepth || score < -CheckmateValue + (int32_t)MaxSearchDepth;
}

void Search::Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, const SearchParam& param, Stats& outStats, SearchResult& outResult)
{
    const bool isMainThread = threadID == 0;
    ThreadData& thread = mThreadData[threadID];

    std::vector<Move> pvMovesSoFar;
    pvMovesSoFar.reserve(param.excludedMoves.size() + numPvLines);

    outResult.resize(numPvLines);

    thread.stats = ThreadStats{};
    thread.moveOrderer.NewSearch();
    thread.prevPvLines.clear();
    thread.prevPvLines.resize(numPvLines);

    uint32_t mateCounter = 0;

    SearchContext searchContext{ game, param, outStats, param.limits.idealTime };

    // main iterative deepening loop
    for (uint32_t depth = 1; depth <= param.limits.maxDepth; ++depth)
    {
        SearchResult tempResult;
        tempResult.resize(numPvLines);

        pvMovesSoFar.clear();
        pvMovesSoFar = param.excludedMoves;

        thread.rootDepth = depth;

        bool finishSearchAtDepth = false;

        for (uint32_t pvIndex = 0; pvIndex < numPvLines; ++pvIndex)
        {
            PvLine& prevPvLine = thread.prevPvLines[pvIndex];

            // use previous iteration score as starting aspiration window
            // if it's the first iteration - try score from transposition table
            ScoreType prevScore = prevPvLine.score;
            if (depth <= 1 && pvIndex == 0)
            {
                TTEntry ttEntry;
                if (param.transpositionTable.Read(game.GetPosition(), ttEntry))
                {
                    if (ttEntry.IsValid())
                    {
                        prevScore = ScoreFromTT(ttEntry.score, 0, game.GetPosition().GetHalfMoveCount());
                    }
                }
            }

            const AspirationWindowSearchParam aspirationWindowSearchParam =
            {
                game.GetPosition(),
                param,
                depth,
                (uint8_t)pvIndex,
                searchContext,
                !pvMovesSoFar.empty() ? pvMovesSoFar.data() : nullptr,
                (uint8_t)(!pvMovesSoFar.empty() ? pvMovesSoFar.size() : 0u),
                prevScore,
                threadID,
            };

            PvLine pvLine = AspirationWindowSearch(thread, aspirationWindowSearchParam);

            // stop search only at depth 2 and more
            if (depth > 1 && CheckStopCondition(thread, searchContext, true))
            {
                finishSearchAtDepth = true;
                break;
            }

            ASSERT(pvLine.score > -CheckmateValue && pvLine.score < CheckmateValue);
            ASSERT(!pvLine.moves.empty());

            // only main thread writes out final PV line
            if (isMainThread)
            {
                outResult[pvIndex] = pvLine;
            }

            // update mate counter
            if (pvIndex == 0)
            {
                if (IsMate(pvLine.score))
                {
                    mateCounter++;
                }
                else
                {
                    mateCounter = 0;
                }
            }

            // store for multi-PV filtering in next iteration
            for (const Move prevMove : pvMovesSoFar)
            {
                ASSERT(prevMove != pvLine.moves.front());
            }
            pvMovesSoFar.push_back(pvLine.moves.front());

            tempResult[pvIndex] = std::move(pvLine);
        }

        if (finishSearchAtDepth)
        {
            if (isMainThread)
            {
                // make sure all PV lines are correct
                for (uint32_t i = 0; i < numPvLines; ++i)
                {
                    ASSERT(outResult[i].score > -CheckmateValue && outResult[i].score < CheckmateValue);
                    ASSERT(!outResult[i].moves.empty());
                }

                // stop other threads
                StopSearch();
            }
            break;
        }

        const ScoreType primaryMoveScore = tempResult.front().score;
        const Move primaryMove = !tempResult.front().moves.empty() ? tempResult.front().moves.front() : Move::Invalid();

        // update time manager
        if (isMainThread &&
            !param.isPonder && !param.limits.analysisMode)
        {
            const TimeManagerUpdateData data{ depth, tempResult, thread.prevPvLines, param.limits };
            TimeManager::Update(game, data, searchContext.maxTimeSoft);
        }

        // rememeber PV lines so they can be used in next iteration
        thread.prevPvLines = std::move(tempResult);

        // check soft time limit every depth iteration
        if (isMainThread &&
            !param.isPonder &&
            searchContext.maxTimeSoft.IsValid() &&
            TimePoint::GetCurrent() >= searchContext.maxTimeSoft)
        {
            StopSearch();
            break;
        }

        // stop the search if found mate in multiple depths in a row
        if (isMainThread &&
            !param.isPonder && !param.limits.analysisMode &&
            mateCounter >= MateCountStopCondition &&
            param.limits.maxDepth == UINT8_MAX)
        {
            StopSearch();
            break;
        }

        // check for singular root move
        if (isMainThread &&
            numPvLines == 1 &&
            depth >= SingularitySearchMinDepth &&
            std::abs(primaryMoveScore) < 1000 &&
            param.limits.rootSingularityTime.IsValid() &&
            TimePoint::GetCurrent() >= param.limits.rootSingularityTime)
        {
            const int32_t scoreTreshold = std::max<int32_t>(SingularitySearchScoreTresholdMin, SingularitySearchScoreTresholdMax - SingularitySearchScoreStep * (depth - SingularitySearchMinDepth));

            const uint32_t singularDepth = depth / 2;
            const ScoreType singularBeta = primaryMoveScore - (ScoreType)scoreTreshold;

            NodeInfo rootNode;
            rootNode.position = game.GetPosition();
            rootNode.isInCheck = rootNode.position.IsInCheck();
            rootNode.depth = singularDepth;
            rootNode.pvIndex = SingularitySearchPvIndex;
            rootNode.alpha = singularBeta - 1;
            rootNode.beta = singularBeta;
            rootNode.moveFilter = &primaryMove;
            rootNode.moveFilterCount = 1;

            ScoreType score = NegaMax(thread, rootNode, searchContext);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            if (score < singularBeta || CheckStopCondition(thread, searchContext, true))
            {
                StopSearch();
                break;
            }
        }
    }
}

PvLine Search::AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param) const
{
    ASSERT(param.pvIndex != SingularitySearchPvIndex);

    int32_t alpha = -InfValue;
    int32_t beta = InfValue;

    // decrease aspiration window with increasing depth
    int32_t aspirationWindow = AspirationWindowMax - param.depth * AspirationWindowStep;
    aspirationWindow = std::max<int32_t>(AspirationWindowMin, aspirationWindow);
    ASSERT(aspirationWindow > 0);

    // scale aspiration window based on score
    aspirationWindow = (128 + std::abs(param.previousScore) / 4) * aspirationWindow / 128;

    // start applying aspiration window at given depth
    if (param.depth > 5 &&
        param.previousScore != InvalidValue &&
        !IsMate(param.previousScore) &&
        !CheckStopCondition(thread, param.searchContext, true))
    {
        alpha = std::max<int32_t>(param.previousScore - aspirationWindow, -InfValue);
        beta = std::min<int32_t>(param.previousScore + aspirationWindow, InfValue);
    }

    PvLine pvLine; // working copy
    PvLine finalPvLine;

    const uint32_t maxPvLine = param.searchParam.limits.analysisMode ? param.depth : std::min(param.depth, DefaultMaxPvLineLength);

    for (;;)
    {
        NodeInfo rootNode;
        rootNode.position = param.position;
        rootNode.isInCheck = param.position.IsInCheck();
        rootNode.isPvNode = true;
        rootNode.isPvNodeFromPrevIteration = true;
        rootNode.depth = param.depth;
        rootNode.pvIndex = param.pvIndex;
        rootNode.alpha = ScoreType(alpha);
        rootNode.beta = ScoreType(beta);
        rootNode.moveFilter = param.moveFilter;
        rootNode.moveFilterCount = param.moveFilterCount;

        pvLine.score = NegaMax(thread, rootNode, param.searchContext);
        ASSERT(pvLine.score >= -CheckmateValue && pvLine.score <= CheckmateValue);
        SearchUtils::GetPvLine(param.searchContext.game, rootNode, param.searchParam.transpositionTable, maxPvLine, pvLine.moves);

        // flush pending per-thread stats
        param.searchContext.stats.Append(thread.stats);

        BoundsType boundsType = BoundsType::Exact;

        aspirationWindow *= 2;
        if (aspirationWindow > 400) aspirationWindow = CheckmateValue;
        
        // out of aspiration window, redo the search in wider score range
        if (pvLine.score <= alpha)
        {
            pvLine.score = ScoreType(alpha);
            alpha -= aspirationWindow;
            alpha = std::max<int32_t>(alpha, -CheckmateValue);
            boundsType = BoundsType::UpperBound;
        }
        else if (pvLine.score >= beta)
        {
            pvLine.score = ScoreType(beta);
            beta += aspirationWindow;
            beta = std::min<int32_t>(beta, CheckmateValue);
            boundsType = BoundsType::LowerBound;
        }

        const bool stopSearch = param.depth > 1 && CheckStopCondition(thread, param.searchContext, true);
        const bool isMainThread = param.threadID == 0;

        ASSERT(!pvLine.moves.empty());
        ASSERT(pvLine.moves.front().IsValid());

        if (isMainThread && param.searchParam.debugLog)
        {
            const TimePoint searchTime = TimePoint::GetCurrent() - param.searchParam.limits.startTimePoint;
            ReportPV(param, pvLine, boundsType, searchTime);
        }

        // don't return line if search was aborted, because the result comes from incomplete search
        if (!stopSearch)
        {
            finalPvLine = std::move(pvLine);
        }

        // stop the search when exact score is found
        if (boundsType == BoundsType::Exact || stopSearch)
        {
            break;
        }
    }

    return finalPvLine;
}

const Move Search::ThreadData::GetPvMove(const NodeInfo& node) const
{
    if (!node.isPvNodeFromPrevIteration || prevPvLines.empty() || node.pvIndex == SingularitySearchPvIndex)
    {
        return Move::Invalid();
    }

    const std::vector<Move>& pvLine = prevPvLines[node.pvIndex].moves;
    if (node.height >= pvLine.size())
    {
        return Move::Invalid();
    }

    const Move pvMove = pvLine[node.height];
    ASSERT(pvMove.IsValid());

    return pvMove;
}

ScoreType Search::QuiescenceNegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const
{
    ASSERT(node.depth <= 0);
    ASSERT(node.alpha <= node.beta);
    ASSERT(node.isPvNode || node.alpha == node.beta - 1);
    ASSERT(node.moveFilterCount == 0);
    ASSERT(node.height > 0);

    // clear PV line
    node.pvLength = 0;

    // update stats
    thread.stats.nodes++;
    thread.stats.quiescenceNodes++;
    thread.stats.maxDepth = std::max<uint32_t>(thread.stats.maxDepth, node.height);
    ctx.stats.Append(thread.stats);

    // Not checking for draw by repetition in the quiesence search
    if (CheckInsufficientMaterial(node.position))
    {
        return 0;
    }

    const Position& position = node.position;

    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;
    ScoreType bestValue = -InfValue;
    ScoreType staticEval = InvalidValue;

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
        ASSERT(ttScore > -CheckmateValue && ttScore < CheckmateValue);

        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

            if (ttEntry.bounds == TTEntry::Bounds::Exact)                           return ttScore;
            else if (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= alpha)  return alpha;
            else if (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= beta)   return beta;
        }
    }

    // do not consider stand pat if in check
    if (!node.isInCheck)
    {
        if (staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(position);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);
            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;

#ifdef COLLECT_SEARCH_STATS
            int32_t binIndex = (evalScore + Stats::EvalHistogramMaxValue) * Stats::EvalHistogramBins / (2 * Stats::EvalHistogramMaxValue);
            binIndex = std::clamp<int32_t>(binIndex, 0, Stats::EvalHistogramBins);
            ctx.stats.evalHistogram[binIndex]++;
#endif // COLLECT_SEARCH_STATS
        }

        // try to use TT score for better evaluation estimate
        if (std::abs(ttScore) < KnownWinValue)
        {
            if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Exact))
            {
                staticEval = ttScore;
            }
        }

        bestValue = staticEval;

        if (bestValue >= beta)
        {
            ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, 0, TTEntry::Bounds::Lower);
            return bestValue;
        }

        if (bestValue > alpha)
        {
            alpha = bestValue;
        }
    }

    ScoreType oldAlpha = alpha;

    NodeInfo childNode;
    childNode.parentNode = &node;
    childNode.pvIndex = node.pvIndex;
    childNode.isPvNode = node.isPvNode;
    childNode.depth = node.depth - 1;
    childNode.height = node.height + 1;

    uint32_t moveGenFlags = 0;
    if (!node.isInCheck)
    {
        moveGenFlags |= MOVE_GEN_ONLY_TACTICAL;
        moveGenFlags |= MOVE_GEN_ONLY_QUEEN_PROMOTIONS;
    }

    const Move pvMove = thread.GetPvMove(node);

    if (pvMove.IsValid() && ttEntry.moves[0].IsValid())
    {
        ASSERT(pvMove == ttEntry.moves[0]);
    }

    MovePicker movePicker(position, thread.moveOrderer, ttEntry, pvMove, moveGenFlags);

    int32_t moveScore = 0;
    Move move;

    Move bestMoves[TTEntry::NumMoves];
    uint32_t numBestMoves = 0;
    int32_t moveIndex = 0;
    uint32_t numQuietCheckEvasion = 0;
    bool searchAborted = false;

    while (movePicker.PickMove(node, ctx.game, move, moveScore))
    {
        ASSERT(move.IsValid());

        // skip losing captures
        if (!node.isInCheck && bestValue > -KnownWinValue && moveScore < MoveOrderer::GoodCaptureValue)
        {
            if (!position.StaticExchangeEvaluation(move, QSearchSeeMargin))
            {
                continue;
            }
        }

        childNode.position = position;
        if (!childNode.position.DoMove(move))
        {
            continue;
        }

        // don't try all check evasions
        if (node.isInCheck && move.IsQuiet())
        {
            if (bestValue > -KnownWinValue && numQuietCheckEvasion > 1)
            {
                continue;
            }
            numQuietCheckEvasion++;
        }

        ctx.searchParam.transpositionTable.Prefetch(childNode.position);

        moveIndex++;

        // Move Count Pruning
        // skip everything after some sane amount of moves has been tried
        // there shouldn't be many "good" captures available in a "normal" chess positions
        if (bestValue > -KnownWinValue)
        {
                 if (node.depth < -4 && moveIndex > 1) break;
            else if (node.depth < -2 && moveIndex > 2) break;
            else if (node.depth <  0 && moveIndex > 4) break;
        }

        childNode.isInCheck = childNode.position.IsInCheck();

        childNode.alpha = -beta;
        childNode.beta = -alpha;
        const ScoreType score = -QuiescenceNegaMax(thread, childNode, ctx);
        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        if (score > bestValue) // new best move found
        {
            // push new best move to the beginning of the list
            for (uint32_t j = TTEntry::NumMoves; j-- > 1; )
            {
                bestMoves[j] = bestMoves[j - 1];
            }
            numBestMoves = std::min(TTEntry::NumMoves, numBestMoves + 1);
            bestMoves[0] = move;
            bestValue = score;
        }

        if (score >= beta)
        {
            break;
        }

        if (score > alpha)
        {
            alpha = score;
        }

        if (CheckStopCondition(thread, ctx, false))
        {
            // abort search of further moves
            searchAborted = true;
            break;
        }
    }

    // no legal moves - checkmate
    if (!searchAborted && node.isInCheck && moveIndex == 0)
    {
        return -CheckmateValue + (ScoreType)node.height;
    }

    // store value in transposition table
    if (!searchAborted)
    {
        // if we didn't beat alpha and had valid TT entry, don't overwrite it
        if (bestValue <= oldAlpha && ttEntry.IsValid() && ttEntry.depth > 0)
        {
            return bestValue;
        }

        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::Lower :
            bestValue > oldAlpha ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::Upper;

        MovesArray<PackedMove, TTEntry::NumMoves> packedBestMoves;
        for (uint32_t i = 0; i < numBestMoves; ++i)
        {
            ASSERT(bestMoves[i].IsValid());
            packedBestMoves[i] = bestMoves[i];
        }
        numBestMoves = packedBestMoves.MergeWith(ttEntry.moves);

        ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, 0, bounds, numBestMoves, packedBestMoves.Data());

#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS
    }

    return bestValue;
}

ScoreType Search::NegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const
{
    ASSERT(node.alpha <= node.beta);
    ASSERT(node.isPvNode || node.alpha == node.beta - 1);

    // clear PV line
    node.pvLength = 0;

    // update stats
    thread.stats.nodes++;
    thread.stats.maxDepth = std::max<uint32_t>(thread.stats.maxDepth, node.height);
    ctx.stats.Append(thread.stats);

    const Position& position = node.position;
    const bool isRootNode = node.height == 0; // root node is the first node in the chain (best move)
    const bool isPvNode = node.isPvNode;
    const bool hasMoveFilter = node.moveFilterCount > 0u;

    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;

    // check if we can draw by repetition in losing position
    if (!isRootNode && alpha < 0 && SearchUtils::CanReachGameCycle(node))
    {
        alpha = 0;
        if (alpha >= beta) return alpha;
    }

    // maximum search depth reached, enter quisence search to find final evaluation
    if (node.depth <= 0)
    {
        return QuiescenceNegaMax(thread, node, ctx);
    }

    // Check for draw
    // Skip root node as we need some move to be reported in PV
    if (!isRootNode)
    {
        if (node.position.GetHalfMoveCount() >= 100 ||
            CheckInsufficientMaterial(node.position) ||
            SearchUtils::IsRepetition(node, ctx.game))
        {
            return true;
        }
    }

    // TODO use proper stack
    const NodeInfo* prevNodes[4] = { nullptr };
    prevNodes[0] = node.parentNode;
    prevNodes[1] = prevNodes[0] ? prevNodes[0]->parentNode : nullptr;
    prevNodes[2] = prevNodes[1] ? prevNodes[1]->parentNode : nullptr;
    prevNodes[3] = prevNodes[2] ? prevNodes[2]->parentNode : nullptr;
    
    ASSERT(node.isInCheck == position.IsInCheck(position.GetSideToMove()));

    const ScoreType oldAlpha = node.alpha;
    ScoreType bestValue = -InfValue;
    ScoreType staticEval = InvalidValue;
    bool tbHit = false;

    // mate distance pruning
    if (!isRootNode)
    {
        alpha = std::max<ScoreType>(-CheckmateValue + (ScoreType)node.height, alpha);
        beta = std::min<ScoreType>(CheckmateValue - (ScoreType)node.height - 1, beta);
        if (alpha >= beta) return alpha;
    }

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
        ASSERT(ttScore > -CheckmateValue && ttScore < CheckmateValue);

        // don't prune in PV nodes, because TT does not contain path information
        if (ttEntry.depth >= node.depth &&
            (node.depth <= 0 || !isPvNode) &&
            !hasMoveFilter &&
            position.GetHalfMoveCount() < 90)
        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

            if (ttEntry.bounds == TTEntry::Bounds::Exact)                           return ttScore;
            else if (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= alpha)  return alpha;
            else if (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= beta)   return beta;
        }
    }

    // try probing Win-Draw-Loose endgame tables
    {
        int32_t wdl = 0;
        if (!isRootNode &&
            (node.depth >= TablebaseProbeDepth || !node.previousMove.IsQuiet()) &&
            ProbeTablebase_WDL(position, &wdl))
        {
            tbHit = true;
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.tbHits++;
#endif // COLLECT_SEARCH_STATS

            // convert the WDL value to a score
            const ScoreType tbValue =
                wdl < 0 ? -ScoreType(TablebaseWinValue - node.height) :
                wdl > 0 ? ScoreType(TablebaseWinValue - node.height) : 0;
            ASSERT(tbValue > -CheckmateValue && tbValue < CheckmateValue);

            // only draws are exact, we don't know exact value for win/loss just based on WDL value
            TTEntry::Bounds bounds =
                wdl < 0 ? TTEntry::Bounds::Upper :
                wdl > 0 ? TTEntry::Bounds::Lower :
                TTEntry::Bounds::Exact;

            if (bounds == TTEntry::Bounds::Exact ||
                (bounds == TTEntry::Bounds::Lower && tbValue >= beta) ||
                (bounds == TTEntry::Bounds::Upper && tbValue <= alpha))
            {
                if (!ttEntry.IsValid())
                {
                    ctx.searchParam.transpositionTable.Write(position, ScoreToTT(tbValue, node.height), staticEval, node.depth, bounds);
                }

#ifdef COLLECT_SEARCH_STATS
                ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS

                return tbValue;
            }
        }
    }

    // evaluate position if it wasn't evaluated
    bool wasPositionEvaluated = true;
    if (!node.isInCheck)
    {
        if (staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(position);
            ASSERT(evalScore < TablebaseWinValue&& evalScore > -TablebaseWinValue);
            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;
            wasPositionEvaluated = false;
        }

        // try to use TT score for better evaluation estimate
        if (std::abs(ttScore) < KnownWinValue)
        {
            if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Exact))
            {
                staticEval = ttScore;
            }
        }

        node.staticEval = staticEval;
    }

    // check how much static evaluation improved between current position and position in previous turn
    // if we were in check in previous turn, use position prior to it
    int32_t evalImprovement = 0;
    if (prevNodes[1] && prevNodes[1]->staticEval != InvalidValue)
    {
        evalImprovement = staticEval - prevNodes[1]->staticEval;
    }
    else if (prevNodes[3] && prevNodes[3]->staticEval != InvalidValue)
    {
        evalImprovement = staticEval - prevNodes[3]->staticEval;
    }
    const bool isImproving = evalImprovement >= -5; // leave some small marigin

    // Futility/Beta Pruning
    if (!isPvNode &&
        !node.isInCheck &&
        node.depth <= BetaPruningDepth &&
        std::abs(staticEval) <= KnownWinValue &&
        staticEval >= (beta + BetaMarginBias + BetaMarginMultiplier * (node.depth - isImproving)))
    {
        return staticEval;
    }

    // Null Move Reductions
    if (!isRootNode &&
        !isPvNode &&
        !node.isInCheck &&
        !hasMoveFilter &&
        staticEval >= beta &&
        node.depth >= NullMoveReductionsStartDepth &&
        (!ttEntry.IsValid() || (ttEntry.bounds != TTEntry::Bounds::Upper) || (ttScore >= beta)) &&
        position.HasNonPawnMaterial(position.GetSideToMove()))
    {
        // don't allow null move if parent or grandparent node was null move
        bool doNullMove = !node.isNullMove;
        if (node.parentNode && node.parentNode->isNullMove) doNullMove = false;

        if (doNullMove)
        {
            const int32_t depthReduction =
                NullMoveReductions_NullMoveDepthReduction +
                node.depth / 4 +
                std::min(3, int32_t(staticEval - beta) / 256);

            NodeInfo childNode;
            childNode.parentNode = &node;
            childNode.pvIndex = node.pvIndex;
            childNode.position = position;
            childNode.alpha = -beta;
            childNode.beta = -beta + 1;
            childNode.isNullMove = true;
            childNode.height = node.height + 1;
            childNode.depth = node.depth - depthReduction;

            childNode.position.DoNullMove();

            ScoreType nullMoveScore = -NegaMax(thread, childNode, ctx);

            if (nullMoveScore >= beta)
            {
                if (nullMoveScore >= TablebaseWinValue)
                    nullMoveScore = beta;

                if (std::abs(beta) < KnownWinValue && node.depth < 10)
                    return nullMoveScore;

                node.depth -= NullMoveReductions_ReSearchDepthReduction;

                if (node.depth <= 0)
                {
                    return QuiescenceNegaMax(thread, node, ctx);
                }
            }
        }
    }

    // reduce depth if position was not found in transposition table
    if (node.depth >= 4 && !ttEntry.IsValid())
    {
        node.depth--;
        if (node.depth >= 6 && node.isPvNode) node.depth--;
    }

    // determine global depth reduction for quiet moves
    int32_t globalDepthReduction = 0;
    {
        // reduce non-PV nodes more
        if (!node.isPvNode) globalDepthReduction++;

        // reduce more if eval is dropping
        if (!isImproving) globalDepthReduction++;

        if (tbHit) globalDepthReduction++;
    }

    NodeInfo childNode;
    childNode.parentNode = &node;
    childNode.height = node.height + 1;
    childNode.pvIndex = node.pvIndex;

    int32_t extension = 0;

    // check extension
    if (node.isInCheck)
    {
        extension++;
    }

    const Move pvMove = thread.GetPvMove(node);

    MovePicker movePicker(position, thread.moveOrderer, ttEntry, pvMove);

    // randomize move order for root node on secondary threads
    if (isRootNode && !thread.isMainThread)
    {
        movePicker.Shuffle();
    }

    int32_t moveScore = 0;
    Move move;

    Move bestMoves[TTEntry::NumMoves];
    for (uint32_t i = 0; i < TTEntry::NumMoves; ++i) bestMoves[i] = Move::Invalid();
    uint32_t numBestMoves = 0;

    uint32_t moveIndex = 0;
    uint32_t quietMoveIndex = 0;
    bool searchAborted = false;
    bool filteredSomeMove = false;
    int32_t singularScoreDiff = 0;

    Move quietMovesTried[MoveList::MaxMoves];
    uint32_t numQuietMovesTried = 0;

    while (movePicker.PickMove(node, ctx.game, move, moveScore))
    {
        ASSERT(move.IsValid());

        // apply node filter (multi-PV search, singularity search, etc.)
        if (!node.ShouldCheckMove(move))
        {
            filteredSomeMove = true;
            continue;
        }

        childNode.position = position;
        if (!childNode.position.DoMove(move))
        {
            continue;
        }

        // start prefetching child node's TT entry
        ctx.searchParam.transpositionTable.Prefetch(childNode.position);

        childNode.isInCheck = childNode.position.IsInCheck();

        moveIndex++;
        if (move.IsQuiet()) quietMoveIndex++;

        // report current move to UCI
        if (isRootNode && thread.isMainThread && ctx.searchParam.debugLog && node.pvIndex == 0)
        {
            const float timeElapsed = (TimePoint::GetCurrent() - ctx.searchParam.limits.startTimePoint).ToSeconds();
            if (timeElapsed > CurrentMoveReportDelay)
            {
                ReportCurrentMove(move, node.depth, moveIndex + node.pvIndex);
            }
        }

        if (!isRootNode && bestValue > -KnownWinValue)
        {
            // Late Move Pruning
            // skip quiet moves that are far in the list
            // the higher depth is, the less agressing pruning is
            if (move.IsQuiet() &&
                !node.isInCheck &&
                node.depth < 9 &&
                quietMoveIndex >= GetLateMovePruningTreshold(node.depth) + isImproving + node.isPvNode)
            {
                continue;
            }

            // History Pruning
            // if a move score is really bad, do not consider this move at low depth
            if (move.IsQuiet() &&
                !node.isInCheck &&
                node.depth < 9 &&
                quietMoveIndex > 1 &&
                moveScore < GetHistoryPruningTreshold(node.depth))
            {
                continue;
            }

            // Futility Pruning
            // skip quiet move that have low chance to beat alpha
            if (move.IsQuiet() &&
                quietMoveIndex > 1 &&
                !node.isPvNode &&
                !node.isInCheck &&
                node.depth > 1 && node.depth < 9 &&
                std::abs(staticEval) <= KnownWinValue &&
                staticEval + 32 * node.depth * node.depth < alpha)
            {
                continue;
            }

            // Static Exchange Evaluation pruning
            // skip all moves that are bad according to SEE
            // the higher depth is, the less agressing pruning is
            if (!node.isInCheck)
            {
                if (move.IsCapture() || childNode.isInCheck)
                {
                    if (!position.StaticExchangeEvaluation(move, -256 * node.depth)) continue;
                }
                else
                {
                    if (!position.StaticExchangeEvaluation(move, -16 * node.depth * node.depth)) continue;
                }
            }
        }

        int32_t moveExtension = extension;
        {
            // promotion extension
            if (move.GetPromoteTo() == Piece::Queen)
            {
                moveExtension++;
            }

            // pawn advanced to 6th row so is about to promote
            if (move.GetPiece() == Piece::Pawn &&
                move.ToSquare().RelativeRank(position.GetSideToMove()) == 6)
            {
                moveExtension++;
            }
        }

        // Singular move detection
        if (!isRootNode &&
            !hasMoveFilter &&
            move == ttEntry.moves[0] &&
            node.depth >= SingularitySearchMinDepth &&
            std::abs(ttScore) < KnownWinValue &&
            ((ttEntry.bounds & TTEntry::Bounds::Lower) != TTEntry::Bounds::Invalid) &&
            ttEntry.depth >= node.depth - 2)
        {
            const ScoreType singularBeta = (ScoreType)std::max(-CheckmateValue, (int32_t)ttScore - SingularExtensionScoreMarigin - 2 * node.depth);

            NodeInfo singularChildNode = node;
            singularChildNode.isPvNode = false;
            singularChildNode.depth = node.depth / 2;
            singularChildNode.pvIndex = SingularitySearchPvIndex;
            singularChildNode.alpha = singularBeta - 1;
            singularChildNode.beta = singularBeta;
            singularChildNode.moveFilter = &move;
            singularChildNode.moveFilterCount = 1;

            const ScoreType singularScore = NegaMax(thread, singularChildNode, ctx);

            if (singularScore < singularBeta)
            {
                singularScoreDiff = singularBeta - singularScore;

                if (node.height < 2 * thread.rootDepth)
                {
                    moveExtension++;
                }
            }
            else if (singularScore >= beta)
            {
                // if second best move beats current beta, there most likely would be beta cutoff
                // when searching it at full depth
                return singularScore;
            }
            else if (ttScore >= beta)
            {
                moveExtension = 0;
            }
        }

        if (move.IsQuiet())
        {
            quietMovesTried[numQuietMovesTried++] = move;
        }

        // avoid extending search too much (maximum 2x depth at root node)
        if (node.height < 2 * thread.rootDepth)
        {
            moveExtension = std::clamp(moveExtension, 0, 2);
        }
        else
        {
            moveExtension = 0;
        }

        childNode.previousMove = move;
        childNode.isPvNodeFromPrevIteration = node.isPvNodeFromPrevIteration && (move == pvMove);

        int32_t depthReduction = 0;

        // Late Move Reduction
        // don't reduce PV moves, while in check, good captures, promotions, etc.
        if (moveIndex > 1 &&
            node.depth >= LateMoveReductionStartDepth &&
            !node.isInCheck &&
            (moveScore < MoveOrderer::GoodCaptureValue && !move.IsPromotion()))
        {
            depthReduction = globalDepthReduction;

            // reduce depth gradually
            depthReduction += mMoveReductionTable[node.depth][std::min(moveIndex, MaxReducedMoves - 1)];

            // reduce more if TT move is singular move
            if (move != ttEntry.moves[0] && singularScoreDiff > 100) depthReduction++;
            if (move != ttEntry.moves[0] && singularScoreDiff > 400) depthReduction++;

            // reduce good moves less
            if (moveScore < -8000) depthReduction++;
            if (moveScore > 0) depthReduction--;
            if (moveScore > 8000) depthReduction--;

            // reduce less if move gives check
            if (childNode.isInCheck) depthReduction--;

            if (node.previousMove.IsCapture() && std::abs(staticEval) >= KnownWinValue) depthReduction++;
        }

        // don't drop into QS
        depthReduction = std::clamp(depthReduction, 0, node.depth + moveExtension - 1);

        ScoreType score = InvalidValue;

        bool doFullDepthSearch = !(isPvNode && moveIndex == 1);

        // PVS search at reduced depth
        if (depthReduction > 0)
        {
            childNode.depth = node.depth + moveExtension - 1 - (int32_t)depthReduction;
            childNode.alpha = -alpha - 1;
            childNode.beta = -alpha;
            childNode.isPvNode = false;

            score = -NegaMax(thread, childNode, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            doFullDepthSearch = score > alpha;
        }

        // PVS search at full depth
        // TODO: internal aspiration window?
        if (doFullDepthSearch)
        {
            childNode.depth = node.depth + moveExtension - 1;
            childNode.alpha = -alpha - 1;
            childNode.beta = -alpha;
            childNode.isPvNode = false;

            score = -NegaMax(thread, childNode, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
        }

        // full search for PV nodes
        if (isPvNode)
        {
            if (moveIndex == 1 || (score > alpha && score < beta))
            {
                childNode.depth = node.depth + moveExtension - 1;
                childNode.alpha = -beta;
                childNode.beta = -alpha;
                childNode.isPvNode = true;

                score = -NegaMax(thread, childNode, ctx);
            }
        }

        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        if (score > bestValue) // new best move found
        {
            // push new best move to the beginning of the list
            for (uint32_t j = TTEntry::NumMoves; j-- > 1; )
            {
                bestMoves[j] = bestMoves[j - 1];
            }
            numBestMoves = std::min(TTEntry::NumMoves, numBestMoves + 1);
            bestMoves[0] = move;
            bestValue = score;

            // update PV line
            if (isPvNode)
            {
                node.pvLength = std::min<uint16_t>(1u + childNode.pvLength, MaxSearchDepth);
                node.pvLine[0] = move;
                memcpy(node.pvLine + 1, childNode.pvLine, sizeof(PackedMove) * std::min<uint16_t>(childNode.pvLength, MaxSearchDepth - 1));
            }
        }

        if (score >= beta)
        {
            ASSERT(moveIndex > 0);

#ifdef COLLECT_SEARCH_STATS
            ctx.stats.betaCutoffHistogram[moveIndex - 1]++;
#endif // COLLECT_SEARCH_STATS

            break;
        }

        if (score > alpha)
        {
            alpha = score;
        }

        if (!isRootNode && CheckStopCondition(thread, ctx, false))
        {
            // abort search of further moves
            searchAborted = true;
            break;
        }
    }

    // update move orderer
    if (bestValue >= beta)
    {
        if (bestMoves[0].IsQuiet())
        {
            thread.moveOrderer.UpdateQuietMovesHistory(node, quietMovesTried, numQuietMovesTried, bestMoves[0], node.depth);
            thread.moveOrderer.UpdateKillerMove(node, bestMoves[0]);
        }
    }

    // no legal moves
    if (!searchAborted && moveIndex == 0u)
    {
        if (filteredSomeMove)
        {
            return -InfValue;
        }
        else
        {
            bestValue = node.isInCheck ? -CheckmateValue + (ScoreType)node.height : 0;

            // write TT entry so it will overwrite any incorrect entry coming from QSearch
            ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), bestValue, INT8_MAX, TTEntry::Bounds::Exact);

            return bestValue;
        }
    }

    ASSERT(bestValue >= -CheckmateValue && bestValue <= CheckmateValue);

    if (isRootNode)
    {
        ASSERT(numBestMoves > 0);
        ASSERT(!isPvNode || node.pvLength > 0);
        ASSERT(!isPvNode || node.pvLine[0] == bestMoves[0]);
    }

    // update transposition table
    // don't write if:
    // - time is exceeded as evaluation may be inaccurate
    // - some move was skipped due to filtering, because 'bestMove' may not be "the best" for the current position
    if (!filteredSomeMove && !CheckStopCondition(thread, ctx, false))
    {
        ASSERT(numBestMoves > 0);

        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::Lower :
            bestValue > oldAlpha ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::Upper;

        // only PV nodes can have exact score
        ASSERT(isPvNode || bounds != TTEntry::Bounds::Exact);

        MovesArray<PackedMove, TTEntry::NumMoves> packedBestMoves;
        for (uint32_t i = 0; i < numBestMoves; ++i)
        {
            ASSERT(bestMoves[i].IsValid());
            packedBestMoves[i] = bestMoves[i];
        }
        numBestMoves = packedBestMoves.MergeWith(ttEntry.moves);

        ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, node.depth, bounds, numBestMoves, packedBestMoves.Data());

#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS
    }

    return bestValue;
}
