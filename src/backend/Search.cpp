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
#include "Score.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <thread>
#include <math.h>

// #define ENABLE_SEARCH_TRACE

static const float CurrentMoveReportDelay = 10.0f;

static const int32_t SingularitySearchMinDepth = 8;
static const int32_t SingularitySearchScoreTresholdMin = 200;
static const int32_t SingularitySearchScoreTresholdMax = 400;
static const int32_t SingularitySearchScoreStep = 25;

static const uint32_t DefaultMaxPvLineLength = 20;
static const uint32_t MateCountStopCondition = 5;

static const int32_t WdlTablebaseProbeDepth = 4;
static const int32_t WdlTablebaseProbeMaxNumPieces = 7;

static const int32_t NullMoveReductionsStartDepth = 2;
static const int32_t NullMoveReductions_NullMoveDepthReduction = 4;
static const int32_t NullMoveReductions_ReSearchDepthReduction = 4;

static const int32_t MaxExtension = 2;

static const int32_t MaxDepthReduction = 12;
static const int32_t LateMoveReductionStartDepth = 2;

static const int32_t AspirationWindowDepthStart = 6;
static const int32_t AspirationWindowMaxSize = 500;
static const int32_t AspirationWindowStart = 40;
static const int32_t AspirationWindowEnd = 20;
static const int32_t AspirationWindowStep = 4;

static const int32_t SingularExtensionScoreMarigin = 5;

static const int32_t BetaPruningDepth = 7;
static const int32_t BetaMarginMultiplier = 135;
static const int32_t BetaMarginBias = 5;

static const int32_t AlphaPruningDepth = 5;
static const int32_t AlphaMarginMultiplier = 256;
static const int32_t AlphaMarginBias = 2000;

static const int32_t RazoringStartDepth = 3;
static const int32_t RazoringMarginMultiplier = 128;
static const int32_t RazoringMarginBias = 20;

class SearchTrace
{
public:
    enum class ExitReason
    {
        Regular,
        Draw,
        GameCycle,
        MateDistancePruning,
        TBHit,
        TTCutoff,
        BetaPruning,
        AlphaPruning,
        Razoring,
        NullMovePruning,
        SingularPruning,
    };

    SearchTrace(const NodeInfo& node) : m_node(node) { }

    void OnNodeExit(ExitReason reason, ScoreType score, Move bestMove = Move::Invalid())
    {
        FILE* f = GetOutputFile();

        const char* exitReasonStr = "";
        switch (reason)
        {
        case ExitReason::Draw:                  exitReasonStr = "Draw"; break;
        case ExitReason::GameCycle:             exitReasonStr = "GameCycle"; break;
        case ExitReason::MateDistancePruning:   exitReasonStr = "MateDistancePruning"; break;
        case ExitReason::TBHit:                 exitReasonStr = "TBHit"; break;
        case ExitReason::TTCutoff:              exitReasonStr = "TTCutoff"; break;
        case ExitReason::BetaPruning:           exitReasonStr = "BetaPruning"; break;
        case ExitReason::AlphaPruning:          exitReasonStr = "AlphaPruning"; break;
        case ExitReason::Razoring:              exitReasonStr = "Razoring"; break;
        case ExitReason::NullMovePruning:       exitReasonStr = "NullMovePruning"; break;
        case ExitReason::SingularPruning:       exitReasonStr = "SingularPruning"; break;
        }

        // write indent
        char spaceBuffer[MaxSearchDepth];
        memset(spaceBuffer, '\t', m_node.height);
        fwrite(spaceBuffer, 1, m_node.height, f);

        fprintf(f, "%s [%s] d=%d, a=%d, b=%d, e=%d | %s score=%d bestMove=%s\n",
                m_node.previousMove.ToString().c_str(),
                m_node.position.ToFEN().c_str(),
                m_node.depth, m_node.alpha, m_node.beta, m_node.staticEval,
                exitReasonStr, score, bestMove.ToString().c_str());
    }

    static void OnRootSearchBegin()
    {
        FILE* f = GetOutputFile();
        fprintf(f, "ROOT SEARCH START\n");
    }

private:

    static FILE* GetOutputFile()
    {
        static FILE* f = fopen("searchTrace.txt", "w");
        return f;
    }

    const NodeInfo& m_node;
};

INLINE static uint32_t GetLateMovePruningTreshold(uint32_t depth)
{
    return 3 + depth + depth * depth / 2;
}

INLINE static int32_t GetHistoryPruningTreshold(int32_t depth)
{
    return 0 - 256 * depth - 64 * depth * depth;
}

void Search::Stats::Append(ThreadStats& threadStats, bool flush)
{
    if (threadStats.nodes >= 64 || flush)
    {
        nodes += threadStats.nodes;
        quiescenceNodes += threadStats.quiescenceNodes;
        AtomicMax(maxDepth, threadStats.maxDepth);
        tbHits += threadStats.tbHits;

        threadStats = ThreadStats{};
    }
}

Search::Search()
{
    BuildMoveReductionTable();

    mThreadData.emplace_back(std::make_unique<ThreadData>());
    mThreadData.front()->isMainThread = true;
}

Search::~Search()
{
    StopWorkerThreads();
}

void Search::StopWorkerThreads()
{
    for (size_t i = 1; i < mThreadData.size(); ++i)
    {
        const ThreadDataPtr& threadData = mThreadData[i];
        std::unique_lock<std::mutex> lock(threadData->newTaskMutex);
        threadData->stopThread = true;
        threadData->newTaskCV.notify_one();
    }

    for (size_t i = 1; i < mThreadData.size(); ++i)
    {
        mThreadData[i]->thread.join();
    }

    mThreadData.erase(mThreadData.begin() + 1, mThreadData.end());
}

void Search::BuildMoveReductionTable()
{
    memset(mMoveReductionTable, 0, sizeof(mMoveReductionTable));

    for (uint32_t depth = 1; depth < LMRTableSize; ++depth)
    {
        for (uint32_t moveIndex = 1; moveIndex < LMRTableSize; ++moveIndex)
        {
            const int32_t reduction = int32_t(0.25f + 0.5f * logf(float(depth)) * logf(float(moveIndex)));
            ASSERT(reduction <= 64);
            mMoveReductionTable[depth][moveIndex] = (uint8_t)std::clamp<int32_t>(reduction, 0, 64);
        }
    }
}

void Search::Clear()
{
    for (const ThreadDataPtr& threadData : mThreadData)
    {
        ASSERT(threadData);
        threadData->moveOrderer.Clear();
        threadData->stats = ThreadStats{};
    }
}

const MoveOrderer& Search::GetMoveOrderer() const
{
    return mThreadData.front()->moveOrderer;
}

bool Search::CheckStopCondition(const ThreadData& thread, const SearchContext& ctx, bool isRootNode)
{
    SearchParam& param = ctx.searchParam;

    if (param.stopSearch.load(std::memory_order_relaxed))
    {
        return true;
    }

    if (!param.isPonder.load(std::memory_order_acquire))
    {
        if (param.limits.maxNodes < UINT64_MAX &&
            ctx.stats.nodes > param.limits.maxNodes)
        {
            // nodes limit exceeded
            param.stopSearch = true;
            return true;
        }

        // check inner nodes periodically
        if (isRootNode || (thread.stats.nodes % 256 == 0))
        {
            if (param.limits.maxTime.IsValid() &&
                param.limits.startTimePoint.IsValid() &&
                TimePoint::GetCurrent() >= param.limits.startTimePoint + param.limits.maxTime)
            {
                // time limit exceeded
                param.stopSearch = true;
                return true;
            }
        }
    }

    return false;
}

void Search::DoSearch(const Game& game, SearchParam& param, SearchResult& outResult)
{
    ASSERT(!param.stopSearch);

    outResult.clear();

    if (!game.GetPosition().IsValid())
    {
        return;
    }

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
        if (param.useRootTablebase && numPvLines == 1)
        {
            int32_t wdl = 0;
            Move tbMove;

            if (ProbeGaviota_Root(game.GetPosition(), tbMove, nullptr, &wdl))
            {
                ASSERT(tbMove.IsValid());
                outResult.front().moves.push_back(tbMove);
                outResult.front().tbScore = static_cast<ScoreType>(wdl * TablebaseWinValue);
                return;
            }

            if (ProbeSyzygy_Root(game.GetPosition(), tbMove, nullptr, &wdl))
            {
                ASSERT(tbMove.IsValid());
                outResult.front().moves.push_back(tbMove);
                outResult.front().tbScore = static_cast<ScoreType>(wdl * TablebaseWinValue);
                return;
            }
        }
    }

    Stats globalStats;

    // Quiescence search debugging 
    if (param.limits.maxDepth == 0)
    {
        ThreadData& thread = *mThreadData.front();

        NodeInfo rootNode;
        rootNode.position = game.GetPosition();
        rootNode.isInCheck = game.GetPosition().IsInCheck();
        rootNode.isPvNodeFromPrevIteration = true;
        rootNode.alpha = -InfValue;
        rootNode.beta = InfValue;
        rootNode.nnContext = thread.GetNNEvaluatorContext(rootNode.height);
        rootNode.nnContext->MarkAsDirty();

        SearchContext searchContext{ game, param, globalStats };
        outResult.resize(1);
        outResult.front().score = QuiescenceNegaMax(thread, rootNode, searchContext);
        SearchUtils::GetPvLine(rootNode, DefaultMaxPvLineLength, outResult.front().moves);

        // flush pending stats
        searchContext.stats.Append(thread.stats, true);

        const AspirationWindowSearchParam aspirationWindowSearchParam =
        {
            game.GetPosition(),
            param,
            0,
            0,
            searchContext,
        };

        ReportPV(aspirationWindowSearchParam, outResult[0], BoundsType::Exact, TimePoint());
    }

    // kick off worker threads
    for (uint32_t i = 1; i < param.numThreads; ++i)
    {
        // spawn missing threads
        if (mThreadData.size() < param.numThreads)
        {
            mThreadData.emplace_back(std::make_unique<ThreadData>());
            mThreadData.back()->thread = std::thread(Search::WorkerThreadCallback, mThreadData.back().get());
        }

        const ThreadDataPtr& threadData = mThreadData[i];
        {
            std::unique_lock<std::mutex> lock(threadData->newTaskMutex);
            ASSERT(!threadData->callback);
            threadData->callback = [this, i, numPvLines, &game, &param, &globalStats]()
            {
                Search_Internal(i, numPvLines, game, param, globalStats);
            };
        }
        threadData->newTaskCV.notify_one();
    }
        
    // do search on main thread
    Search_Internal(0, numPvLines, game, param, globalStats);

    // wait for worker threads
    for (uint32_t i = 1; i < param.numThreads; ++i)
    {
        const ThreadDataPtr& threadData = mThreadData[i];
        std::unique_lock<std::mutex> lock(threadData->taskFinishedMutex);
        threadData->taskFinishedCV.wait(lock, [&threadData]() { return threadData->taskFinished; });
        threadData->taskFinished = false;
    }

    // select best PV line from finished threads
    {
        uint32_t bestThreadIndex = 0;
        uint16_t bestDepth = 0;
        ScoreType bestScore = -InfValue;

        for (uint32_t i = 0; i < param.numThreads; ++i)
        {
            const ThreadDataPtr& threadData = mThreadData[i];
            ASSERT(!threadData->pvLines.empty());
            ASSERT(threadData->pvLines.size() == numPvLines);

#ifndef CONFIGURATION_FINAL
            // make sure all PV lines are correct
            for (const PvLine& pvLine : threadData->pvLines)
            {
                ASSERT(pvLine.score > -CheckmateValue && pvLine.score < CheckmateValue);
                ASSERT(!pvLine.moves.empty());
            }
#endif // CONFIGURATION_FINAL

            const PvLine& pvLine = threadData->pvLines.front();

            if ((threadData->depthCompleted > bestDepth && pvLine.score > bestScore) ||
                (threadData->depthCompleted > bestDepth && !IsMate(bestScore)) ||
                (IsMate(pvLine.score) && pvLine.score > bestScore))
            {
                bestDepth = threadData->depthCompleted;
                bestScore = pvLine.score;
                bestThreadIndex = i;
            }
        }

        outResult = std::move(mThreadData[bestThreadIndex]->pvLines);
    }
}

void Search::WorkerThreadCallback(ThreadData* threadData)
{
    while (!threadData->stopThread)
    {
        {
            // wait for task
            std::function<void()> callback;
            {
                std::unique_lock<std::mutex> lock(threadData->newTaskMutex);
                threadData->newTaskCV.wait(lock, [threadData]() { return threadData->callback || threadData->stopThread; });
                if (threadData->stopThread) break;
                callback = std::move(threadData->callback);
            }

            callback();
        }

        // notify main thread
        {
            std::unique_lock<std::mutex> lock(threadData->taskFinishedMutex);
            ASSERT(!threadData->taskFinished);
            threadData->taskFinished = true;
        }
        threadData->taskFinishedCV.notify_one();
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

    if (param.searchContext.stats.tbHits)
    {
        ss << " tbhits " << param.searchContext.stats.tbHits;
    }

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
        const Stats& stats = param.searchContext.stats;

        {
            const float sum = float(stats.numPvNodes + stats.numAllNodes + stats.numCutNodes);
            printf("Num PV-Nodes:  %" PRIu64 " (%.2f%%)\n", stats.numPvNodes, 100.0f * float(stats.numPvNodes) / sum);
            printf("Num Cut-Nodes: %" PRIu64 " (%.2f%%)\n", stats.numCutNodes, 100.0f * float(stats.numCutNodes) / sum);
            printf("Num All-Nodes: %" PRIu64 " (%.2f%%)\n", stats.numAllNodes, 100.0f * float(stats.numAllNodes) / sum);

            printf("Expected Cut-Nodes Hits: %.2f%%\n", 100.0f * float(stats.expectedCutNodesSuccess) / float(stats.expectedCutNodesSuccess + stats.expectedCutNodesFailure));
        }

        {
            uint32_t maxMoveIndex = 0;
            uint64_t sum = 0;
            double average = 0.0;
            for (uint32_t i = 0; i < MoveList::MaxMoves; ++i)
            {
                if (stats.betaCutoffHistogram[i])
                {
                    sum += stats.betaCutoffHistogram[i];
                    average += (double)i * (double)stats.betaCutoffHistogram[i];
                    maxMoveIndex = std::max(maxMoveIndex, i);
                }
            }
            average /= sum;
            printf("Average cutoff move index: %.3f\n", average);
            printf("Beta cutoff histogram\n");
            for (uint32_t i = 0; i < maxMoveIndex; ++i)
            {
                const uint64_t value = stats.betaCutoffHistogram[i];
                printf("    %u : %" PRIu64 " (%.2f%%)\n", i, value, 100.0f * float(value) / float(sum));
            }
        }


        {
            printf("Eval value histogram\n");
            for (uint32_t i = 0; i < Stats::EvalHistogramBins; ++i)
            {
                const int32_t lowEval = -Stats::EvalHistogramMaxValue + i * 2 * Stats::EvalHistogramMaxValue / Stats::EvalHistogramBins;
                const int32_t highEval = lowEval + 2 * Stats::EvalHistogramMaxValue / Stats::EvalHistogramBins;
                const uint64_t value = stats.evalHistogram[i];

                printf("    %4d...%4d %" PRIu64 "\n", lowEval, highEval, value);
            }
        }
    }
#endif // COLLECT_SEARCH_STATS

    std::cout << std::move(ss.str()) << std::endl;
}

void Search::ReportCurrentMove(const Move& move, int32_t depth, uint32_t moveNumber) const
{
    std::stringstream ss{ std::ios_base::out };

    ss << "info depth " << depth;
    ss << " currmove " << move.ToString();
    ss << " currmovenumber " << moveNumber;

    std::cout << std::move(ss.str()) << std::endl;
}

void Search::Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, SearchParam& param, Stats& outStats)
{
    const bool isMainThread = threadID == 0;
    ThreadData& thread = *(mThreadData[threadID]);

    std::vector<Move> pvMovesSoFar;
    pvMovesSoFar.reserve(param.excludedMoves.size() + numPvLines);

    // clear per-thread data for new search
    thread.stats = ThreadStats{};
    thread.depthCompleted = 0;
    thread.pvLines.clear();
    thread.pvLines.resize(numPvLines);
    thread.moveOrderer.NewSearch();

    uint32_t mateCounter = 0;

    SearchContext searchContext{ game, param, outStats };

    // main iterative deepening loop
    for (uint16_t depth = 1; depth <= param.limits.maxDepth; ++depth)
    {
        SearchResult tempResult;
        tempResult.resize(numPvLines);

        pvMovesSoFar.clear();
        pvMovesSoFar = param.excludedMoves;

        thread.rootDepth = depth;

        bool abortSearch = false;

        for (uint32_t pvIndex = 0; pvIndex < numPvLines; ++pvIndex)
        {
            PvLine& prevPvLine = thread.pvLines[pvIndex];

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
                abortSearch = true;
                break;
            }

            ASSERT(pvLine.score > -CheckmateValue && pvLine.score < CheckmateValue);
            ASSERT(!pvLine.moves.empty());

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
#ifndef CONFIGURATION_FINAL
            for (const Move prevMove : pvMovesSoFar)
            {
                ASSERT(prevMove != pvLine.moves.front());
            }
#endif // CONFIGURATION_FINAL
            pvMovesSoFar.push_back(pvLine.moves.front());

            tempResult[pvIndex] = std::move(pvLine);
        }

        if (abortSearch)
        {
            if (isMainThread)
            {
                // stop other threads
                param.stopSearch = true;
            }
            break;
        }

        const ScoreType primaryMoveScore = tempResult.front().score;
        const Move primaryMove = !tempResult.front().moves.empty() ? tempResult.front().moves.front() : Move::Invalid();

        // update time manager
        if (isMainThread && !param.limits.analysisMode)
        {
            const TimeManagerUpdateData data{ depth, tempResult, thread.pvLines };
            TimeManager::Update(game, data, searchContext.searchParam.limits);
        }

        // remember PV lines so they can be used in next iteration
        thread.depthCompleted = depth;
        thread.pvLines = std::move(tempResult);

        if (isMainThread &&
            !param.isPonder.load(std::memory_order_acquire))
        {
            // check soft time limit every depth iteration
            if (param.limits.idealTime.IsValid() &&
                param.limits.startTimePoint.IsValid() &&
                TimePoint::GetCurrent() >= param.limits.startTimePoint + param.limits.idealTime)
            {
                param.stopSearch = true;
                break;
            }

            // stop the search if found mate in multiple depths in a row
            if (!param.limits.analysisMode &&
                mateCounter >= MateCountStopCondition &&
                param.limits.maxDepth == UINT16_MAX)
            {
                param.stopSearch = true;
                break;
            }
        }

        // check for singular root move
        if (isMainThread &&
            numPvLines == 1 &&
            depth >= SingularitySearchMinDepth &&
            std::abs(primaryMoveScore) < 1000 &&
            param.limits.rootSingularityTime.IsValid() &&
            param.limits.startTimePoint.IsValid() &&
            TimePoint::GetCurrent() >= param.limits.startTimePoint + param.limits.rootSingularityTime)
        {
            const int32_t scoreTreshold = std::max<int32_t>(SingularitySearchScoreTresholdMin, SingularitySearchScoreTresholdMax - SingularitySearchScoreStep * (depth - SingularitySearchMinDepth));

            const uint16_t singularDepth = depth / 2;
            const ScoreType singularBeta = primaryMoveScore - (ScoreType)scoreTreshold;

            NodeInfo rootNode;
            rootNode.position = game.GetPosition();
            rootNode.isInCheck = rootNode.position.IsInCheck();
            rootNode.isSingularSearch = true;
            rootNode.depth = singularDepth;
            rootNode.alpha = singularBeta - 1;
            rootNode.beta = singularBeta;
            rootNode.moveFilter = &primaryMove;
            rootNode.moveFilterCount = 1;
            rootNode.nnContext = thread.nnContextStack[0].get();
            rootNode.nnContext->MarkAsDirty();

            ScoreType score = NegaMax(thread, rootNode, searchContext);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            if (score < singularBeta || CheckStopCondition(thread, searchContext, true))
            {
                param.stopSearch = true;
                break;
            }
        }
    }
}

PvLine Search::AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param) const
{
    int32_t alpha = -InfValue;
    int32_t beta = InfValue;
    uint32_t depth = param.depth;

    // decrease aspiration window with increasing depth
    int32_t window = AspirationWindowStart - (param.depth - AspirationWindowDepthStart) * AspirationWindowStep;
    window = std::max<int32_t>(AspirationWindowEnd, window);
    ASSERT(window > 0);

    // increase window based on score
    window += std::abs(param.previousScore) / 10;

    // start applying aspiration window at given depth
    if (param.searchParam.useAspirationWindows &&
        param.depth >= AspirationWindowDepthStart &&
        param.previousScore != InvalidValue &&
        !IsMate(param.previousScore) &&
        !CheckStopCondition(thread, param.searchContext, true))
    {
        alpha = std::max<int32_t>(param.previousScore - window, -InfValue);
        beta = std::min<int32_t>(param.previousScore + window, InfValue);
    }

    PvLine pvLine; // working copy
    PvLine finalPvLine;

    const uint32_t maxPvLine = param.searchParam.limits.analysisMode ? UINT32_MAX : std::min(param.depth, DefaultMaxPvLineLength);

    for (;;)
    {
        //std::cout << "Window: " << alpha << " ... " << beta << std::endl;

        NodeInfo rootNode;
        rootNode.position = param.position;
        rootNode.isInCheck = param.position.IsInCheck();
        rootNode.isPvNodeFromPrevIteration = true;
        rootNode.depth = static_cast<int16_t>(depth);
        rootNode.pvIndex = param.pvIndex;
        rootNode.alpha = ScoreType(alpha);
        rootNode.beta = ScoreType(beta);
        rootNode.moveFilter = param.moveFilter;
        rootNode.moveFilterCount = param.moveFilterCount;
        rootNode.nnContext = thread.GetNNEvaluatorContext(rootNode.height);
        rootNode.nnContext->MarkAsDirty();

#ifdef ENABLE_SEARCH_TRACE
        SearchTrace::OnRootSearchBegin();
#endif

        pvLine.score = NegaMax(thread, rootNode, param.searchContext);
        ASSERT(pvLine.score >= -CheckmateValue && pvLine.score <= CheckmateValue);
        SearchUtils::GetPvLine(rootNode, maxPvLine, pvLine.moves);

        // flush pending per-thread stats
        param.searchContext.stats.Append(thread.stats, true);

        // increase window, fallback to full window after some treshold
        window = 2 * window + 5;
        if (window > AspirationWindowMaxSize) window = CheckmateValue;

        BoundsType boundsType = BoundsType::Exact;

        // out of aspiration window, redo the search in wider score range
        if (pvLine.score <= alpha)
        {
            pvLine.score = ScoreType(alpha);
            beta = (alpha + beta + 1) / 2;
            alpha = pvLine.score - window;
            alpha = std::max<int32_t>(alpha, -CheckmateValue);
            boundsType = BoundsType::UpperBound;
        }
        else if (pvLine.score >= beta)
        {
            pvLine.score = ScoreType(beta);
            beta += window;
            beta = std::min<int32_t>(beta, CheckmateValue);
            boundsType = BoundsType::LowerBound;

            // reduce re-search depth
            if (depth > AspirationWindowDepthStart && depth + 3 > param.depth)
            {
                depth--;
            }
        }

        const bool stopSearch = param.depth > 1 && CheckStopCondition(thread, param.searchContext, true);
        const bool isMainThread = param.threadID == 0;

        ASSERT(!pvLine.moves.empty());
        ASSERT(pvLine.moves.front().IsValid());

        if (isMainThread && param.searchParam.debugLog && !param.searchParam.stopSearch)
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

Search::ThreadData::ThreadData()
{
    constexpr uint32_t InitialNNEvaluatorStackSize = 32;

    for (uint32_t i = 0; i < InitialNNEvaluatorStackSize; ++i)
    {
        GetNNEvaluatorContext(i);
    }

    randomSeed = 0x4abf372b;
}

NNEvaluatorContext* Search::ThreadData::GetNNEvaluatorContext(uint32_t height)
{
    ASSERT(height < MaxSearchDepth);

    if (!nnContextStack[height])
    {
        nnContextStack[height] = std::make_unique<NNEvaluatorContext>();
    }

    return nnContextStack[height].get();
}

const Move Search::ThreadData::GetPvMove(const NodeInfo& node) const
{
    if (!node.isPvNodeFromPrevIteration || pvLines.empty() || node.isSingularSearch)
    {
        return Move::Invalid();
    }

    const std::vector<Move>& pvLine = pvLines[node.pvIndex].moves;
    if (node.height >= pvLine.size())
    {
        return Move::Invalid();
    }

    const Move pvMove = pvLine[node.height];
    ASSERT(pvMove.IsValid());
    ASSERT(node.position.IsMoveLegal(pvMove));

    return pvMove;
}

uint32_t Search::ThreadData::GetRandomUint()
{
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
    randomSeed ^= randomSeed << 13;
    randomSeed ^= randomSeed >> 17;
    randomSeed ^= randomSeed << 5;
    return randomSeed;
}

static ScoreType AdjustEvalScore(const ScoreType rawScore, const NodeInfo& node)
{
    // TODO analyze history moves, scale down when moving same piece all the time

    int32_t adjustedScore = rawScore;
    
    if (std::abs(rawScore) < KnownWinValue)
    {
        // scale down when approaching 50-move draw
        adjustedScore = (int32_t)rawScore * (128 - std::max(0, (int32_t)node.position.GetHalfMoveCount() - 4)) / 128;
    }

    return static_cast<ScoreType>(adjustedScore);
}

static void RefreshPsqtScore(NodeInfo& node)
{
    // refresh PSQT score (incrementally, if possible)
    if (node.height == 0 ||
        node.previousMove.GetPiece() == Piece::King ||
        !node.nnContext)
    {
        node.psqtScore = ComputePSQT(node.position);
    }
    else // incremental update
    {
        node.psqtScore = node.parentNode->psqtScore;
        ASSERT(node.psqtScore.mg != INT32_MIN && node.psqtScore.eg != INT32_MIN);
        ComputeIncrementalPSQT(node.psqtScore, node.position, node.nnContext->dirtyPieces, node.nnContext->numDirtyPieces);
    }
}

ScoreType Search::QuiescenceNegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const
{
    ASSERT(node.alpha < node.beta);
    ASSERT(node.moveFilterCount == 0);

    const bool isPvNode = node.IsPV();

    // clear PV line
    node.pvLength = 0;

    // update stats
    thread.stats.nodes++;
    thread.stats.quiescenceNodes++;
    thread.stats.maxDepth = std::max<uint32_t>(thread.stats.maxDepth, node.height + 1);
    ctx.stats.Append(thread.stats);

    // Not checking for draw by repetition in the quiescence search
    if (CheckInsufficientMaterial(node.position))
    {
        return 0;
    }

    const Position& position = node.position;

    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;
    ScoreType bestValue = -CheckmateValue + (ScoreType)node.height;
    ScoreType staticEval = InvalidValue;
    ScoreType futilityBase = -InfValue;

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
        ASSERT(ttScore > -CheckmateValue && ttScore < CheckmateValue);

#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

        // don't prune in PV nodes, because TT does not contain path information
        if (!isPvNode)
        {
            if (ttEntry.bounds == TTEntry::Bounds::Exact)                           return ttScore;
            else if (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= alpha)  return alpha;
            else if (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= beta)   return beta;
        }
    }

    // make sure PSQT score is up to date before calling Evaluate()
    RefreshPsqtScore(node);

    const bool maxDepthReached = false; // node.height + 1 >= MaxSearchDepth;

    // do not consider stand pat if in check
    if (!node.isInCheck || maxDepthReached)
    {
        if (staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(position, &node);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);

            if (ctx.searchParam.evalProbingInterface)
            {
                ctx.searchParam.evalProbingInterface->ReportPosition(position, evalScore);
            }

            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;

#ifdef COLLECT_SEARCH_STATS
            int32_t binIndex = (evalScore + Stats::EvalHistogramMaxValue) * Stats::EvalHistogramBins / (2 * Stats::EvalHistogramMaxValue);
            binIndex = std::clamp<int32_t>(binIndex, 0, Stats::EvalHistogramBins - 1);
            ctx.stats.evalHistogram[binIndex]++;
#endif // COLLECT_SEARCH_STATS
        }

        ASSERT(staticEval != InvalidValue);

        const ScoreType adjustedEvalScore = AdjustEvalScore(staticEval, node);

        bestValue = adjustedEvalScore;

        // try to use TT score for better score estimate
        if (std::abs(ttScore) < KnownWinValue)
        {
            if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > adjustedEvalScore) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < adjustedEvalScore) ||
                (ttEntry.bounds == TTEntry::Bounds::Exact))
            {
                bestValue = ttScore;
            }
        }

        if (bestValue >= beta || maxDepthReached)
        {
            if (!ttEntry.IsValid())
            {
                ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, 0, TTEntry::Bounds::Lower);
            }
            return bestValue;
        }

        if (bestValue > alpha)
        {
            alpha = bestValue;
        }

        futilityBase = bestValue + 150;
    }

    ScoreType oldAlpha = alpha;

    NodeInfo childNode;
    childNode.parentNode = &node;
    childNode.pvIndex = node.pvIndex;
    childNode.depth = node.depth - 1;
    childNode.height = node.height + 1;
    childNode.nnContext = thread.GetNNEvaluatorContext(childNode.height);
    childNode.nnContext->MarkAsDirty();

    uint32_t moveGenFlags = MOVE_GEN_MASK_CAPTURES|MOVE_GEN_MASK_PROMOTIONS;
    if (node.isInCheck)
    {
        moveGenFlags |= MOVE_GEN_MASK_QUIET;
    }

    MovePicker movePicker(position, thread.moveOrderer, ttEntry, Move::Invalid(), moveGenFlags);

    int32_t moveScore = 0;
    Move move;

    Move bestMoves[TTEntry::NumMoves];
    uint32_t numBestMoves = 0;
    int32_t moveIndex = 0;
    uint32_t numQuietCheckEvasion = 0;
    bool searchAborted = false;

    Move captureMovesTried[MoveList::MaxMoves];
    uint32_t numCaptureMovesTried = 0;

    while (movePicker.PickMove(node, ctx.game, move, moveScore))
    {
        ASSERT(move.IsValid());

        if (!node.isInCheck &&
            bestValue > -TablebaseWinValue)
        {
            ASSERT(!move.IsQuiet());

            // skip underpromotions
            if (move.IsUnderpromotion()) continue;

            // skip losing captures
            if (moveScore < MoveOrderer::GoodCaptureValue) continue;

            // futility pruning - skip captures that won't beat alpha
            if (move.IsCapture() &&
                futilityBase > -KnownWinValue &&
                futilityBase <= alpha &&
                !position.StaticExchangeEvaluation(move, 1))
            {
                bestValue = std::max(bestValue, futilityBase);
                continue;
            }
        }

        childNode.position = position;
        if (!childNode.position.DoMove(move, childNode.nnContext))
        {
            continue;
        }

        // start prefetching child node's TT entry
        ctx.searchParam.transpositionTable.Prefetch(childNode.position);

        // don't try all check evasions
        if (node.isInCheck && move.IsQuiet())
        {
            if (numBestMoves > 0 &&
                numQuietCheckEvasion > 1 &&
                bestValue > -TablebaseWinValue) continue;

            numQuietCheckEvasion++;
        }

        moveIndex++;

        // Move Count Pruning
        // skip everything after some sane amount of moves has been tried
        // there shouldn't be many "good" captures available in a "normal" chess positions
        if (numBestMoves > 0 &&
            bestValue > -TablebaseWinValue)
        {
                 if (node.depth < -4 && moveIndex > 1) break;
            else if (node.depth < -2 && moveIndex > 2) break;
            else if (node.depth <  0 && moveIndex > 3) break;
        }

        childNode.previousMove = move;
        childNode.isInCheck = childNode.position.IsInCheck();

        childNode.alpha = -beta;
        childNode.beta = -alpha;
        const ScoreType score = -QuiescenceNegaMax(thread, childNode, ctx);
        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        if (move.IsCapture())
        {
            captureMovesTried[numCaptureMovesTried++] = move;
        }

        if (score > bestValue) // new best move found
        {
            // update PV line
            if (isPvNode)
            {
                node.pvLength = std::min<uint16_t>(1u + childNode.pvLength, MaxSearchDepth);
                node.pvLine[0] = move;
                memcpy(node.pvLine + 1, childNode.pvLine, sizeof(PackedMove) * std::min<uint16_t>(childNode.pvLength, MaxSearchDepth - 1));
            }

            // push new best move to the beginning of the list
            for (uint32_t j = TTEntry::NumMoves; j-- > 1; )
            {
                bestMoves[j] = bestMoves[j - 1];
            }
            numBestMoves = std::min(TTEntry::NumMoves, numBestMoves + 1);
            bestMoves[0] = move;
            bestValue = score;

            if (score >= beta) break;
            if (score > alpha) alpha = score;
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

    // update move orderer
    if (bestValue >= beta)
    {
        if (bestMoves[0].IsCapture())
        {
            thread.moveOrderer.UpdateCapturesHistory(node, captureMovesTried, numCaptureMovesTried, bestMoves[0], node.depth);
        }
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
    ASSERT(node.alpha < node.beta);

#ifdef ENABLE_SEARCH_TRACE
    SearchTrace trace(node);
#endif // ENABLE_SEARCH_TRACE

    // clear PV line
    node.pvLength = 0;

    // update stats
    thread.stats.nodes++;
    thread.stats.maxDepth = std::max<uint32_t>(thread.stats.maxDepth, node.height + 1);
    ctx.stats.Append(thread.stats);

    const Position& position = node.position;
    const bool isRootNode = node.height == 0; // root node is the first node in the chain (best move)
    const bool isPvNode = node.IsPV();
    const bool hasMoveFilter = node.moveFilterCount > 0u;

    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;

    // check if we can draw by repetition in losing position
    if (!isRootNode && alpha < 0 && SearchUtils::CanReachGameCycle(node))
    {
        alpha = 0;
        if (alpha >= beta)
        {
#ifdef ENABLE_SEARCH_TRACE
            trace.OnNodeExit(SearchTrace::ExitReason::GameCycle, alpha);
#endif // ENABLE_SEARCH_TRACE
            return alpha;
        }
    }

    // maximum search depth reached, enter quiescence search to find final evaluation
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
#ifdef ENABLE_SEARCH_TRACE
            trace.OnNodeExit(SearchTrace::ExitReason::Draw, 0);
#endif // ENABLE_SEARCH_TRACE
            return 0;
        }
    }

    ASSERT(node.isInCheck == position.IsInCheck(position.GetSideToMove()));

    // mate distance pruning
    if (!isRootNode)
    {
        alpha = std::max<ScoreType>(-CheckmateValue + (ScoreType)node.height, alpha);
        beta = std::min<ScoreType>(CheckmateValue - (ScoreType)node.height - 1, beta);
        if (alpha >= beta)
        {
#ifdef ENABLE_SEARCH_TRACE
            trace.OnNodeExit(SearchTrace::ExitReason::MateDistancePruning, alpha);
#endif // ENABLE_SEARCH_TRACE
            return alpha;
        }
    }

    const ScoreType oldAlpha = node.alpha;
    ScoreType bestValue = -InfValue;
    ScoreType staticEval = InvalidValue;
    ScoreType maxValue = InfValue; // max value according to tablebases
    bool tbHit = false;

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

        staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
        ASSERT(ttScore > -CheckmateValue && ttScore < CheckmateValue);

        // don't prune in PV nodes, because TT does not contain path information
        if (!isPvNode &&
            !hasMoveFilter &&
            ttEntry.depth >= node.depth &&
            position.GetHalfMoveCount() < 80)
        {
            // transposition table cutoff
            ScoreType ttCutoffValue = InvalidValue;
            if (ttEntry.bounds == TTEntry::Bounds::Exact)                           ttCutoffValue = ttScore;
            else if (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= alpha)  ttCutoffValue = alpha;
            else if (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= beta)   ttCutoffValue = beta;

            if (ttCutoffValue != InvalidValue)
            {
#ifdef ENABLE_SEARCH_TRACE
                trace.OnNodeExit(SearchTrace::ExitReason::TTCutoff, ttCutoffValue);
#endif // ENABLE_SEARCH_TRACE
                return ttCutoffValue;
            }
        }
    }

    // try probing Win-Draw-Loose endgame tables
    {
        int32_t wdl = 0;
        if (!isRootNode &&
            (node.depth >= WdlTablebaseProbeDepth || !node.previousMove.IsQuiet()) &&
            position.GetNumPieces() <= WdlTablebaseProbeMaxNumPieces &&
            (ProbeSyzygy_WDL(position, &wdl) || ProbeGaviota(position, nullptr, &wdl)))
        {
            tbHit = true;
            thread.stats.tbHits++;

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

            // clamp best score to tablebase score
            if (bounds == TTEntry::Bounds::Lower)
            {
                alpha = std::max(alpha, bestValue);
            }
            else
            {
                maxValue = tbValue;
            }

            if (!isPvNode &&
                (bounds == TTEntry::Bounds::Exact ||
                 (bounds == TTEntry::Bounds::Lower && tbValue >= beta) ||
                 (bounds == TTEntry::Bounds::Upper && tbValue <= alpha)))
            {
                if (!ttEntry.IsValid())
                {
                    ctx.searchParam.transpositionTable.Write(position, ScoreToTT(tbValue, node.height), staticEval, node.depth, bounds);
#ifdef COLLECT_SEARCH_STATS
                    ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS
                }

#ifdef ENABLE_SEARCH_TRACE
                trace.OnNodeExit(SearchTrace::ExitReason::TBHit, tbValue);
#endif // ENABLE_SEARCH_TRACE
                return tbValue;
            }

        }
    }

    // make sure PSQT score is up to date before calling Evaluate()
    RefreshPsqtScore(node);

    // evaluate position if it wasn't evaluated
    if (!node.isInCheck)
    {
        // always evaluate on PV line so the NN accumulator is up to date
        // TODO there should be a separate function to just update accumulator,
        // so it could be done if in check as well
        if (staticEval == InvalidValue || node.isPvNodeFromPrevIteration)
        {
            const ScoreType evalScore = Evaluate(position, &node);
            ASSERT(evalScore < TablebaseWinValue&& evalScore > -TablebaseWinValue);

            if (ctx.searchParam.evalProbingInterface)
            {
                ctx.searchParam.evalProbingInterface->ReportPosition(position, evalScore);
            }

            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;
        }

        ASSERT(staticEval != InvalidValue);

        // adjust static eval based on node path
        node.staticEval = AdjustEvalScore(staticEval, node);

        // try to use TT score for better evaluation estimate
        if (std::abs(ttScore) < KnownWinValue)
        {
            if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > node.staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < node.staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Exact))
            {
                node.staticEval = ttScore;
            }
        }
    }

    // check how much static evaluation improved between current position and position in previous turn
    // if we were in check in previous turn, use position prior to it
    int32_t evalImprovement = 0;
    if (!node.isInCheck)
    {
        // TODO use proper stack
        const NodeInfo* prevNodes[4] = { nullptr };
        prevNodes[0] = node.parentNode;
        prevNodes[1] = prevNodes[0] ? prevNodes[0]->parentNode : nullptr;

        if (prevNodes[1] && prevNodes[1]->staticEval != InvalidValue)
        {
            evalImprovement = node.staticEval - prevNodes[1]->staticEval;
        }
        else
        {
            prevNodes[2] = prevNodes[1] ? prevNodes[1]->parentNode : nullptr;
            prevNodes[3] = prevNodes[2] ? prevNodes[2]->parentNode : nullptr;

            if (prevNodes[3] && prevNodes[3]->staticEval != InvalidValue)
            {
                evalImprovement = node.staticEval - prevNodes[3]->staticEval;
            }
        }
    }
    const bool isImproving = evalImprovement >= -5; // leave some small margin

    if (!isPvNode && !hasMoveFilter && !node.isInCheck)
    {
        // Futility/Beta Pruning
        if (node.depth <= BetaPruningDepth &&
            node.staticEval <= KnownWinValue &&
            node.staticEval >= (beta + BetaMarginBias + BetaMarginMultiplier * (node.depth - isImproving)))
        {
#ifdef ENABLE_SEARCH_TRACE
            trace.OnNodeExit(SearchTrace::ExitReason::BetaPruning, alpha);
#endif // ENABLE_SEARCH_TRACE
            return node.staticEval;
        }

        // Alpha Pruning
        if (node.depth <= AlphaPruningDepth &&
            alpha < KnownWinValue &&
            node.staticEval > -KnownWinValue &&
            node.staticEval + AlphaMarginBias + AlphaMarginMultiplier * node.depth <= alpha)
        {
#ifdef ENABLE_SEARCH_TRACE
            trace.OnNodeExit(SearchTrace::ExitReason::AlphaPruning, alpha);
#endif // ENABLE_SEARCH_TRACE
            return node.staticEval;
        }

        // Razoring
        // prune if quiescence search on current position can't beat beta
        if (node.depth <= RazoringStartDepth &&
            beta < KnownWinValue &&
            node.staticEval + RazoringMarginBias + RazoringMarginMultiplier * node.depth < beta)
        {
            const ScoreType qScore = QuiescenceNegaMax(thread, node, ctx);
            if (qScore < beta)
            {
#ifdef ENABLE_SEARCH_TRACE
                trace.OnNodeExit(SearchTrace::ExitReason::Razoring, qScore);
#endif // ENABLE_SEARCH_TRACE
                return qScore;
            }
        }

        // Null Move Reductions
        if (node.staticEval >= beta &&
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
                    std::min(3, int32_t(node.staticEval - beta) / 256);

                NodeInfo childNode;
                childNode.parentNode = &node;
                childNode.pvIndex = node.pvIndex;
                childNode.position = position;
                childNode.alpha = -beta;
                childNode.beta = -beta + 1;
                childNode.isNullMove = true;
                childNode.height = node.height + 1;
                childNode.depth = static_cast<int16_t>(node.depth - depthReduction);
                childNode.isCutNode = !node.isCutNode;
                childNode.nnContext = thread.GetNNEvaluatorContext(childNode.height);
                childNode.nnContext->MarkAsDirty();

                childNode.position.DoNullMove();

                ScoreType nullMoveScore = -NegaMax(thread, childNode, ctx);

                if (nullMoveScore >= beta)
                {
                    if (nullMoveScore >= TablebaseWinValue)
                        nullMoveScore = beta;

                    if (std::abs(beta) < KnownWinValue && node.depth < 10)
                    {
#ifdef ENABLE_SEARCH_TRACE
                        trace.OnNodeExit(SearchTrace::ExitReason::NullMovePruning, nullMoveScore);
#endif // ENABLE_SEARCH_TRACE
                        return nullMoveScore;
                    }

                    node.depth -= NullMoveReductions_ReSearchDepthReduction;

                    if (node.depth <= 0)
                    {
                        return QuiescenceNegaMax(thread, node, ctx);
                    }
                }
            }
        }
    }

    // reduce depth if position was not found in transposition table
    if (node.depth >= 4 && !ttEntry.IsValid())
    {
        node.depth -= 1 + node.depth / 4;
    }

    NodeInfo childNode;
    childNode.parentNode = &node;
    childNode.height = node.height + 1;
    childNode.pvIndex = node.pvIndex;
    childNode.nnContext = thread.GetNNEvaluatorContext(childNode.height);
    childNode.nnContext->MarkAsDirty();

    int32_t extension = 0;

    // check extension
    if (node.isInCheck)
    {
        extension++;
    }

    const Move pvMove = thread.GetPvMove(node);
    const PackedMove ttMove = ttEntry.moves[0];

    // determine global depth reduction for quiet moves
    int32_t globalDepthReduction = 0;
    {
        // reduce non-PV nodes more
        if (!isPvNode) globalDepthReduction++;

        // reduce more if eval is dropping
        if (!isImproving) globalDepthReduction++;

        // reduce more if TT move is a capture
        if (ttMove.IsValid() && position.IsCapture(ttMove)) globalDepthReduction++;

        if (tbHit) globalDepthReduction++;

        // reduce more if entered a winning endgame
        if (node.previousMove.IsCapture() && node.staticEval >= KnownWinValue) globalDepthReduction++;
    }

    thread.moveOrderer.ClearKillerMoves(node.height + 1);

    MovePicker movePicker(position, thread.moveOrderer, ttEntry, pvMove, MOVE_GEN_MASK_ALL);

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

    Move quietMovesTried[MoveList::MaxMoves];
    Move captureMovesTried[MoveList::MaxMoves];
    uint32_t numQuietMovesTried = 0;
    uint32_t numCaptureMovesTried = 0;
    uint32_t numChecks = 0;

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
        if (!childNode.position.DoMove(move, childNode.nnContext))
        {
            continue;
        }

        // start prefetching child node's TT entry
        ctx.searchParam.transpositionTable.Prefetch(childNode.position);

        moveIndex++;
        if (move.IsQuiet()) quietMoveIndex++;

        childNode.isInCheck = childNode.position.IsInCheck();
        const bool isFirstCheckMove = childNode.isInCheck && numChecks == 0;
        numChecks += childNode.isInCheck;

        if (!node.isInCheck &&
            !childNode.isInCheck &&
            !isRootNode &&
            bestValue > -KnownWinValue &&
            position.HasNonPawnMaterial(position.GetSideToMove()))
        {
            if (move.IsQuiet() || move.IsUnderpromotion())
            {
                // Late Move Pruning
                // skip quiet moves that are far in the list
                // the higher depth is, the less aggressive pruning is
                if (node.depth < 9 &&
                    quietMoveIndex >= GetLateMovePruningTreshold(node.depth) + isImproving + isPvNode)
                {
                    continue;
                }

                // History Pruning
                // if a move score is really bad, do not consider this move at low depth
                if (quietMoveIndex > 1 &&
                    node.depth < 9 &&
                    moveScore < GetHistoryPruningTreshold(node.depth))
                {
                    continue;
                }

                // Futility Pruning
                // skip quiet move that have low chance to beat alpha
                if (quietMoveIndex > 1 &&
                    node.depth > 1 && node.depth < 9 &&
                    node.staticEval >= -KnownWinValue && node.staticEval <= KnownWinValue &&
                    node.staticEval + 32 * node.depth * node.depth < alpha)
                {
                    continue;
                }
            }

            // Static Exchange Evaluation pruning
            // skip all moves that are bad according to SEE
            // the higher depth is, the less aggressive pruning is
            if (move.IsCapture())
            {
                if (node.depth <= 4 &&
                    moveScore < MoveOrderer::GoodCaptureValue &&
                    !position.StaticExchangeEvaluation(move, -120 * node.depth)) continue;
            }
            else
            {
                if (node.depth <= 8 &&
                    !position.StaticExchangeEvaluation(move, -64 * node.depth)) continue;
            }
        }

        // report current move to UCI
        if (isRootNode && thread.isMainThread && ctx.searchParam.debugLog && node.pvIndex == 0)
        {
            const float timeElapsed = (TimePoint::GetCurrent() - ctx.searchParam.limits.startTimePoint).ToSeconds();
            if (timeElapsed > CurrentMoveReportDelay)
            {
                ReportCurrentMove(move, node.depth, moveIndex + node.pvIndex);
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
            move == ttMove &&
            node.depth >= SingularitySearchMinDepth &&
            std::abs(ttScore) < KnownWinValue &&
            ((ttEntry.bounds & TTEntry::Bounds::Lower) != TTEntry::Bounds::Invalid) &&
            ttEntry.depth >= node.depth - 2)
        {
            const ScoreType singularBeta = (ScoreType)std::max(-CheckmateValue, (int32_t)ttScore - SingularExtensionScoreMarigin - 2 * node.depth);

            NodeInfo singularChildNode = node;
            singularChildNode.isPvNodeFromPrevIteration = false;
            singularChildNode.isSingularSearch = true;
            singularChildNode.depth = node.depth / 2;
            singularChildNode.alpha = singularBeta - 1;
            singularChildNode.beta = singularBeta;
            singularChildNode.moveFilter = &move;
            singularChildNode.moveFilterCount = 1;

            const ScoreType singularScore = NegaMax(thread, singularChildNode, ctx);

            if (singularScore < singularBeta)
            {
                if (node.height < 2 * thread.rootDepth)
                {
                    moveExtension++;
                }
            }
            else if (singularScore >= beta)
            {
                // if second best move beats current beta, there most likely would be beta cutoff
                // when searching it at full depth
#ifdef ENABLE_SEARCH_TRACE
                trace.OnNodeExit(SearchTrace::ExitReason::SingularPruning, singularScore);
#endif // ENABLE_SEARCH_TRACE
                return singularScore;
            }
            else if (ttScore >= beta)
            {
                moveExtension--;
            }

            // NegaMax can overwrite NN context for child node, so we need to recreate it by doing the move again...
            childNode.position = position;
            VERIFY(childNode.position.DoMove(move, childNode.nnContext));
        }

        // avoid extending search too much (maximum 2x depth at root node)
        if (node.height < 2 * thread.rootDepth)
        {
            moveExtension = std::clamp(moveExtension, 0, MaxExtension);
        }
        else
        {
            moveExtension = 0;
        }

        childNode.previousMove = move;
        childNode.isPvNodeFromPrevIteration = node.isPvNodeFromPrevIteration && (move == pvMove);

        int32_t depthReduction = 0;

        // Late Move Reduction
        // don't reduce while in check, good captures, promotions, etc.
        if (node.depth >= LateMoveReductionStartDepth &&
            !node.isInCheck &&
            !isFirstCheckMove &&
            moveIndex > 1u &&
            (moveScore < MoveOrderer::GoodCaptureValue || numCaptureMovesTried > 4) && // allow reducing bad captures and any capture if far in the list
            move.GetPromoteTo() != Piece::Queen)
        {
            depthReduction = globalDepthReduction;

            // reduce depth gradually
            depthReduction += mMoveReductionTable[std::min(uint32_t(node.depth), LMRTableSize - 1)][std::min(moveIndex, LMRTableSize - 1)];

            // reduce good moves less
            if (moveScore < -8000) depthReduction++;
            if (moveScore > 0) depthReduction--;
            if (moveScore > 8000) depthReduction--;

            // reduce less if move gives check
            if (childNode.isInCheck) depthReduction--;

            if (node.isCutNode) depthReduction++;

            if (!thread.isMainThread && (thread.GetRandomUint() % 8 == 0)) depthReduction++;
        }

        // limit reduction, don't drop into QS
        depthReduction = std::min(depthReduction, MaxDepthReduction);
        depthReduction = std::clamp(depthReduction, 0, node.depth + moveExtension - 1);

        ScoreType score = InvalidValue;

        bool doFullDepthSearch = !(isPvNode && moveIndex == 1);

        // PVS search at reduced depth
        if (depthReduction > 0)
        {
            ASSERT(moveIndex > 1);

            childNode.depth = static_cast<int16_t>(node.depth + moveExtension - 1 - depthReduction);
            childNode.alpha = -alpha - 1;
            childNode.beta = -alpha;
            childNode.isCutNode = true;

            score = -NegaMax(thread, childNode, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            doFullDepthSearch = score > alpha;
        }

        // PVS search at full depth
        // TODO: internal aspiration window?
        if (doFullDepthSearch)
        {
            childNode.depth = static_cast<int16_t>(node.depth + moveExtension - 1);
            childNode.alpha = -alpha - 1;
            childNode.beta = -alpha;
            childNode.isCutNode = !node.isCutNode;

            score = -NegaMax(thread, childNode, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
        }

        // full search for PV nodes
        if (isPvNode)
        {
            if (moveIndex == 1 || (score > alpha && score < beta))
            {
                childNode.depth = static_cast<int16_t>(node.depth + moveExtension - 1);
                childNode.alpha = -beta;
                childNode.beta = -alpha;
                childNode.isCutNode = false;

                score = -NegaMax(thread, childNode, ctx);
            }
        }

        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        if (move.IsQuiet())
        {
            quietMovesTried[numQuietMovesTried++] = move;
        }
        else if (move.IsCapture())
        {
            captureMovesTried[numCaptureMovesTried++] = move;
        }

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
            ASSERT(moveIndex <= MoveList::MaxMoves);
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.betaCutoffHistogram[moveIndex - 1]++;
#endif // COLLECT_SEARCH_STATS

            break;
        }

        if (score > alpha)
        {
            ASSERT(isPvNode);

            // reduce remaining moves more if we managed to find new best move
            int32_t reducedDepth = node.depth - globalDepthReduction;
            if (reducedDepth > 1 && reducedDepth < 8 &&
                beta < KnownWinValue &&
                alpha > -KnownWinValue)
            {
                globalDepthReduction++;
            }

            alpha = score;
        }

        if (!isRootNode && CheckStopCondition(thread, ctx, false))
        {
            // abort search of further moves
            searchAborted = true;
            break;
        }
    }

    // no legal moves
    if (!searchAborted && moveIndex == 0u)
    {
        if (filteredSomeMove)
        {
            bestValue = -InfValue;
        }
        else
        {
            bestValue = node.isInCheck ? -CheckmateValue + (ScoreType)node.height : 0;

            // write TT entry so it will overwrite any incorrect entry coming from QSearch
            ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), bestValue, INT8_MAX, TTEntry::Bounds::Exact);
        }

#ifdef ENABLE_SEARCH_TRACE
        trace.OnNodeExit(SearchTrace::ExitReason::Regular, bestValue);
#endif // ENABLE_SEARCH_TRACE
        return bestValue;
    }

    // update move orderer
    if (bestValue >= beta)
    {
        if (bestMoves[0].IsQuiet())
        {
            thread.moveOrderer.UpdateQuietMovesHistory(node, quietMovesTried, numQuietMovesTried, bestMoves[0], node.depth);
            thread.moveOrderer.UpdateKillerMove(node, bestMoves[0]);
        }
        else if (bestMoves[0].IsCapture())
        {
            thread.moveOrderer.UpdateCapturesHistory(node, captureMovesTried, numCaptureMovesTried, bestMoves[0], node.depth);
        }
    }

#ifdef COLLECT_SEARCH_STATS
    {
        const bool isCutNode = bestValue >= beta;

        if (isCutNode)                      ctx.stats.numCutNodes++;
        else if (bestValue > oldAlpha)      ctx.stats.numPvNodes++;
        else                                ctx.stats.numAllNodes++;

        if (node.isCutNode == isCutNode)    ctx.stats.expectedCutNodesSuccess++;
        else                                ctx.stats.expectedCutNodesFailure++;
    }
#endif // COLLECT_SEARCH_STATS

    ASSERT(bestValue >= -CheckmateValue && bestValue <= CheckmateValue);

    if (isRootNode)
    {
        ASSERT(numBestMoves > 0);
        ASSERT(!isPvNode || node.pvLength > 0);
        ASSERT(!isPvNode || node.pvLine[0] == bestMoves[0]);
    }

    bestValue = std::min(bestValue, maxValue);

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

#ifdef ENABLE_SEARCH_TRACE
    trace.OnNodeExit(SearchTrace::ExitReason::Regular, bestValue, bestMoves[0]);
#endif // ENABLE_SEARCH_TRACE
    return bestValue;
}
