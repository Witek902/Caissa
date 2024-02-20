#include "Search.hpp"
#include "SearchUtils.hpp"
#include "MovePicker.hpp"
#include "Game.hpp"
#include "MoveList.hpp"
#include "Material.hpp"
#include "Evaluate.hpp"
#include "TranspositionTable.hpp"
#include "Tablebase.hpp"
#include "TimeManager.hpp"
#include "PositionHash.hpp"
#include "Score.hpp"
#include "Tuning.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <thread>
#include <math.h>

// #define VALIDATE_MOVE_PICKER

static const float PvLineReportDelay = 0.005f;
static const float CurrentMoveReportDelay = 5.0f;
static const uint32_t DefaultMaxPvLineLength = 20;
static const uint32_t MateCountStopCondition = 7;
static const int32_t WdlTablebaseProbeDepth = 5;

DEFINE_PARAM(LateMoveReductionScale_Quiets, 42, 20, 70);
DEFINE_PARAM(LateMoveReductionBias_Quiets, 50, 20, 80);
DEFINE_PARAM(LateMoveReductionScale_Captures, 39, 20, 70);
DEFINE_PARAM(LateMoveReductionBias_Captures, 58, 20, 80);

DEFINE_PARAM(ProbcutStartDepth, 5, 3, 8);
DEFINE_PARAM(ProbcutBetaOffset, 153, 80, 300);
DEFINE_PARAM(ProbcutBetaOffsetInCheck, 320, 100, 500);

DEFINE_PARAM(FutilityPruningDepth, 9, 6, 15);
DEFINE_PARAM(FutilityPruningScale, 33, 16, 64);
DEFINE_PARAM(FutilityPruningStatscoreDiv, 506, 128, 1024);

DEFINE_PARAM(SingularitySearchMinDepth, 9, 5, 20);
DEFINE_PARAM(SingularitySearchScoreTresholdMin, 180, 100, 300);
DEFINE_PARAM(SingularitySearchScoreTresholdMax, 420, 200, 600);
DEFINE_PARAM(SingularitySearchScoreStep, 27, 10, 50);

DEFINE_PARAM(NullMovePruningStartDepth, 2, 1, 10);
DEFINE_PARAM(NullMovePruning_NullMoveDepthReduction, 3, 1, 5);
DEFINE_PARAM(NullMovePruning_ReSearchDepthReduction, 5, 1, 5);

DEFINE_PARAM(LateMoveReductionStartDepth, 2, 1, 3);
DEFINE_PARAM(LateMovePruningBase, 4, 1, 10);
DEFINE_PARAM(HistoryPruningLinearFactor, 237, 100, 500);
DEFINE_PARAM(HistoryPruningQuadraticFactor, 136, 50, 200);

DEFINE_PARAM(AspirationWindowMaxSize, 498, 200, 1000);
DEFINE_PARAM(AspirationWindow, 10, 6, 20);

DEFINE_PARAM(SingularExtensionMinDepth, 5, 4, 10);
DEFINE_PARAM(SingularDoubleExtensionMarigin, 18, 10, 30);

DEFINE_PARAM(QSearchFutilityPruningOffset, 100, 50, 150);

DEFINE_PARAM(BetaPruningDepth, 6, 5, 10);
DEFINE_PARAM(BetaMarginMultiplier, 118, 80, 180);
DEFINE_PARAM(BetaMarginBias, 6, 0, 20);

DEFINE_PARAM(SSEPruningMultiplier_Captures, 125, 50, 200);
DEFINE_PARAM(SSEPruningMultiplier_NonCaptures, 56, 50, 150);

DEFINE_PARAM(RazoringStartDepth, 3, 1, 6);
DEFINE_PARAM(RazoringMarginMultiplier, 147, 100, 200);
DEFINE_PARAM(RazoringMarginBias, 19, 0, 25);

DEFINE_PARAM(ReductionStatOffset, 7538, 5000, 12000);
DEFINE_PARAM(ReductionStatDiv, 9964, 6000, 12000);

DEFINE_PARAM(EvalCorrectionScale, 501, 1, 1024);
DEFINE_PARAM(EvalCorrectionBlendFactor, 256, 8, 512);

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

INLINE static uint32_t GetLateMovePruningTreshold(uint32_t depth, bool improving)
{
    return improving ?
        LateMovePruningBase + depth * depth :
        LateMovePruningBase + depth * depth / 2;
}

INLINE static int32_t GetHistoryPruningTreshold(int32_t depth)
{
    return 0 - HistoryPruningLinearFactor * depth - HistoryPruningQuadraticFactor * depth * depth;
}

void SearchStats::Append(SearchThreadStats& threadStats, bool flush)
{
    if (threadStats.nodesTemp >= 128 || flush)
    {
        nodes += threadStats.nodesTemp;
        threadStats.nodesTemp = 0;

        quiescenceNodes += threadStats.quiescenceNodes;
        threadStats.quiescenceNodes = 0;

        tbHits += threadStats.tbHits;
        threadStats.tbHits = 0;

        AtomicMax(maxDepth, threadStats.maxDepth);
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

void Search::BuildMoveReductionTable(LMRTableType& table, float scale, float bias)
{
    // clear first row and column
    for (uint32_t i = 0; i < LMRTableSize; ++i)
    {
        table[i][0] = table[0][i] = 0;
    }

    for (uint32_t depth = 1; depth < LMRTableSize; ++depth)
    {
        for (uint32_t moveIndex = 1; moveIndex < LMRTableSize; ++moveIndex)
        {
            const int32_t reduction = int32_t(bias + scale * Log(float(depth)) * Log(float(moveIndex)));
            ASSERT(reduction <= 64);
            table[depth][moveIndex] = (uint8_t)std::clamp<int32_t>(reduction, 0, 64);
        }
    }
}

void Search::BuildMoveReductionTable()
{
    BuildMoveReductionTable(mMoveReductionTable_Quiets,
        static_cast<float>(LateMoveReductionScale_Quiets) / 100.0f,
        static_cast<float>(LateMoveReductionBias_Quiets) / 100.0f);

    BuildMoveReductionTable(mMoveReductionTable_Captures,
        static_cast<float>(LateMoveReductionScale_Captures) / 100.0f,
        static_cast<float>(LateMoveReductionBias_Captures) / 100.0f);
}

void Search::Clear()
{
    for (const ThreadDataPtr& threadData : mThreadData)
    {
        ASSERT(threadData);
        threadData->moveOrderer.Clear();
        threadData->nodeCache.Reset();
        threadData->stats = SearchThreadStats{};
        memset(threadData->matScoreCorrection, 0, sizeof(threadData->matScoreCorrection));
        memset(threadData->pawnStructureCorrection, 0, sizeof(threadData->pawnStructureCorrection));
    }
}

const MoveOrderer& Search::GetMoveOrderer() const
{
    return mThreadData.front()->moveOrderer;
}

const NodeCache& Search::GetNodeCache() const
{
    return mThreadData.front()->nodeCache;
}

bool Search::CheckStopCondition(const ThreadData& thread, const SearchContext& ctx, bool isRootNode)
{
    SearchParam& param = ctx.searchParam;

    if (param.stopSearch.load(std::memory_order_relaxed)) [[unlikely]]
    {
        return true;
    }

    if (thread.isMainThread && !param.isPonder.load(std::memory_order_acquire))
    {
        if (param.limits.maxNodes < UINT64_MAX &&
            ctx.stats.nodes > param.limits.maxNodes) [[unlikely]]
        {
            // nodes limit exceeded
            param.stopSearch = true;
            return true;
        }

        // check inner nodes periodically
        if (isRootNode || (thread.stats.nodesTotal % 512 == 0)) [[unlikely]]
        {
            if (param.limits.maxTime.IsValid() &&
                param.limits.startTimePoint.IsValid() &&
                TimePoint::GetCurrent() >= param.limits.startTimePoint + param.limits.maxTime) [[unlikely]]
            {
                // time limit exceeded
                param.stopSearch = true;
                return true;
            }
        }
    }

    return false;
}

void Search::DoSearch(const Game& game, SearchParam& param, SearchResult& outResult, SearchStats* outStats)
{
    ASSERT(!param.stopSearch);

    outResult.clear();

    if (!game.GetPosition().IsValid())
    {
        return;
    }

    // clamp number of PV lines (there can't be more than number of max moves)
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

#ifdef ENABLE_TUNING
    BuildMoveReductionTable();
#endif // ENABLE_TUNING

    SearchStats globalStats;

    // Quiescence search debugging 
    if (param.limits.maxDepth == 0)
    {
        ThreadData& thread = *mThreadData.front();

        NodeInfo& rootNode = thread.searchStack[0];
        rootNode = NodeInfo{};
        rootNode.position = game.GetPosition();
        rootNode.isInCheck = game.GetPosition().IsInCheck();
        rootNode.position.ComputeThreats(rootNode.threats);
        rootNode.isPvNodeFromPrevIteration = true;
        rootNode.alpha = -InfValue;
        rootNode.beta = InfValue;
        rootNode.nnContext.MarkAsDirty();

        SearchContext searchContext{ game, param, globalStats, param.excludedMoves };
        outResult.resize(1);
        outResult.front().score = QuiescenceNegaMax<NodeType::Root>(thread, &rootNode, searchContext);
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

            if ((threadData->depthCompleted >= bestDepth && pvLine.score > bestScore) ||
                (threadData->depthCompleted > bestDepth && !IsMate(bestScore)) ||
                (IsMate(pvLine.score) && pvLine.score > bestScore))
            {
                bestDepth = threadData->depthCompleted;
                bestScore = pvLine.score;
                bestThreadIndex = i;
            }
        }

#ifndef CONFIGURATION_FINAL
        if (param.numThreads > 1)
        {
            for (uint32_t i = 0; i < param.numThreads; ++i)
            {
                const ThreadDataPtr& threadData = mThreadData[i];
                const PvLine& pvLine = threadData->pvLines.front();
                std::cout << "info string thread " << i
                    << " completed depth " << threadData->depthCompleted
                    << " move " << pvLine.moves.front().ToString() << " score " << pvLine.score;
                if (i == bestThreadIndex) std::cout << " (selected)";
                std::cout << std::endl;
            }
        }
#endif // CONFIGURATION_FINAL

        outResult = std::move(mThreadData[bestThreadIndex]->pvLines);
    }

    if (outStats)
    {
        *outStats = globalStats;
    }

    param.stopSearch = false;
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
    const float timeInSeconds = searchTime.ToSeconds();

    // don't report PV line if very small amount of time passed and we have time limits
    if (timeInSeconds < PvLineReportDelay &&
        param.searchParam.limits.maxTime.IsValid() &&
        !param.searchParam.limits.analysisMode)
    {
        return;
    }

    std::stringstream ss{ std::ios_base::out };

    const uint64_t numNodes = param.searchContext.stats.nodes.load();

    ss << "info depth " << param.depth;
    ss << " seldepth " << (uint32_t)param.searchContext.stats.maxDepth;
    if (param.searchParam.numPvLines > 1) ss << " multipv " << (param.pvIndex + 1);

    if (pvLine.score > CheckmateValue - (int32_t)MaxSearchDepth)        ss << " score mate " << (CheckmateValue - pvLine.score + 1) / 2;
    else if (pvLine.score < -CheckmateValue + (int32_t)MaxSearchDepth)  ss << " score mate -" << (CheckmateValue + pvLine.score + 1) / 2;
    else                                                                ss << " score cp " << NormalizeEval(pvLine.score);

    if (param.searchParam.showWDL)
    {
        const uint32_t ply = 2 * param.searchContext.game.GetPosition().GetMoveCount();
        const float w = EvalToWinProbability(pvLine.score / 100.0f, ply);
        const float l = EvalToWinProbability(-pvLine.score / 100.0f, ply);
        const float d = 1.0f - w - l;
        ss << " wdl "
            << (int32_t)roundf(w * 1000.0f) << " "
            << (int32_t)roundf(d * 1000.0f) << " "
            << (int32_t)roundf(l * 1000.0f);
    }

    if (boundsType == BoundsType::LowerBound) ss << " lowerbound";
    if (boundsType == BoundsType::UpperBound) ss << " upperbound";

    ss << " nodes " << numNodes;
    if (timeInSeconds > 0.01f && numNodes > 100) ss << " nps " << (int64_t)((double)numNodes / (double)timeInSeconds);
    ss << " hashfull " << param.searchParam.transpositionTable.GetHashFull();
    if (param.searchContext.stats.tbHits) ss << " tbhits " << param.searchContext.stats.tbHits;
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
        const SearchStats& stats = param.searchContext.stats;

        {
            const float sum = float(stats.numPvNodes + stats.numAllNodes + stats.numCutNodes);
            printf("Num PV-Nodes:  %" PRIu64 " (%.2f%%)\n", stats.numPvNodes, 100.0f * float(stats.numPvNodes) / sum);
            printf("Num Cut-Nodes: %" PRIu64 " (%.2f%%)\n", stats.numCutNodes, 100.0f * float(stats.numCutNodes) / sum);
            printf("Num All-Nodes: %" PRIu64 " (%.2f%%)\n", stats.numAllNodes, 100.0f * float(stats.numAllNodes) / sum);

            printf("Expected Cut-Nodes Hits: %.2f%%\n", 100.0f * float(stats.expectedCutNodesSuccess) / float(stats.expectedCutNodesSuccess + stats.expectedCutNodesFailure));
        }

        // beta cutoffs stats
        {
            uint32_t maxMoveIndex = 0;
            double average = 0.0;
            for (uint32_t i = 0; i < MoveList::MaxMoves; ++i)
            {
                if (stats.betaCutoffHistogram[i])
                {
                    average += (double)i * (double)stats.betaCutoffHistogram[i];
                    maxMoveIndex = std::max(maxMoveIndex, i);
                }
            }
            average /= stats.totalBetaCutoffs;
            printf("Average cutoff move index: %.3f\n", average);
            printf("TT-move beta cutoffs : %" PRIu64 " (%.2f%%)\n", stats.ttMoveBetaCutoffs, 100.0f * float(stats.ttMoveBetaCutoffs) / float(stats.totalBetaCutoffs));
            printf("Winning capture cutoffs : %" PRIu64 " (%.2f%%)\n", stats.winningCaptureCutoffs, 100.0f * float(stats.winningCaptureCutoffs) / float(stats.totalBetaCutoffs));
            printf("Good capture cutoffs : %" PRIu64 " (%.2f%%)\n", stats.goodCaptureCutoffs, 100.0f * float(stats.goodCaptureCutoffs) / float(stats.totalBetaCutoffs));
            printf("Killer move beta cutoffs : %" PRIu64 " (%.2f%%)\n", stats.killerMoveBetaCutoffs, 100.0f * float(stats.killerMoveBetaCutoffs) / float(stats.totalBetaCutoffs));
            printf("Counter move beta cutoffs : %" PRIu64 " (%.2f%%)\n", stats.counterMoveBetaCutoffs, 100.0f * float(stats.counterMoveBetaCutoffs) / float(stats.totalBetaCutoffs));
            printf("Quiet cutoffs : %" PRIu64 " (%.2f%%)\n", stats.quietCutoffs, 100.0f * float(stats.quietCutoffs) / float(stats.totalBetaCutoffs));
            printf("Bad capture cutoffs : %" PRIu64 " (%.2f%%)\n", stats.badCaptureCutoffs, 100.0f * float(stats.badCaptureCutoffs) / float(stats.totalBetaCutoffs));

            for (uint32_t i = 0; i < maxMoveIndex; ++i)
            {
                const uint64_t value = stats.betaCutoffHistogram[i];
                printf("    %u : %" PRIu64 " (%.2f%%)\n", i, value, 100.0f * float(value) / float(stats.totalBetaCutoffs));
            }
        }

        {
            printf("Eval value histogram\n");
            for (uint32_t i = 0; i < SearchStats::EvalHistogramBins; ++i)
            {
                const int32_t lowEval = -SearchStats::EvalHistogramMaxValue + i * 2 * SearchStats::EvalHistogramMaxValue / SearchStats::EvalHistogramBins;
                const int32_t highEval = lowEval + 2 * SearchStats::EvalHistogramMaxValue / SearchStats::EvalHistogramBins;
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

void Search::Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, SearchParam& param, SearchStats& outStats)
{
    const bool isMainThread = threadID == 0;
    ThreadData& thread = *(mThreadData[threadID]);

    // clear per-thread data for new search
    thread.stats = SearchThreadStats{};
    thread.depthCompleted = 0;
    thread.pvLines.clear();
    thread.pvLines.resize(numPvLines);
    thread.avgScores.clear();
    thread.avgScores.resize(numPvLines, 0);
    thread.moveOrderer.NewSearch();
    thread.nodeCache.OnNewSearch();

    uint32_t mateCounter = 0;
    TimeManagerState timeManagerState;

    SearchContext searchContext{ game, param, outStats };
    searchContext.excludedRootMoves.reserve(param.excludedMoves.size() + numPvLines);

    // main iterative deepening loop
    for (uint16_t depth = 1; depth <= param.limits.maxDepth; ++depth)
    {
        SearchResult tempResult;
        tempResult.resize(numPvLines);

        searchContext.excludedRootMoves.clear();
        searchContext.excludedRootMoves = param.excludedMoves;

        thread.rootDepth = depth;

        bool abortSearch = false;

        for (uint32_t pvIndex = 0; pvIndex < numPvLines; ++pvIndex)
        {
            // use previous iteration score as starting aspiration window
            // if it's the first iteration - try score from transposition table
            ScoreType prevScore = thread.avgScores[pvIndex];
            if (depth <= 1 && pvIndex == 0)
            {
                TTEntry ttEntry;
                if (param.transpositionTable.Read(game.GetPosition(), ttEntry) && ttEntry.IsValid())
                {
                    prevScore = ScoreFromTT(ttEntry.score, 0, game.GetPosition().GetHalfMoveCount());
                }
            }

            const AspirationWindowSearchParam aspirationWindowSearchParam =
            {
                game.GetPosition(),
                param,
                depth,
                pvIndex,
                searchContext,
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
            for (const Move prevMove : searchContext.excludedRootMoves)
            {
                ASSERT(prevMove != pvLine.moves.front());
            }
#endif // CONFIGURATION_FINAL
            searchContext.excludedRootMoves.push_back(pvLine.moves.front());

            tempResult[pvIndex] = std::move(pvLine);
            thread.avgScores[pvIndex] = ScoreType(((int32_t)thread.avgScores[pvIndex] + (int32_t)tempResult[pvIndex].score) / 2);
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
            TimeManagerUpdateData data{ depth, tempResult, thread.pvLines };

            // compute fraction of nodes spent on searching best move
            if (const NodeCacheEntry* nodeCacheEntry = thread.nodeCache.GetEntry(game.GetPosition(), 0))
            {
                if (const NodeCacheEntry::MoveInfo* moveInfo = nodeCacheEntry->GetMove(primaryMove))
                {
                    data.bestMoveNodeFraction = nodeCacheEntry->nodesSum > 0 ?
                        (static_cast<double>(moveInfo->nodesSearched) / static_cast<double>(nodeCacheEntry->nodesSum)) : 0.0;
                }
            }

            UpdateTimeManager(data, searchContext.searchParam.limits, timeManagerState);
        }

        // remember PV lines so they can be used in next iteration
        thread.pvLines = std::move(tempResult);

        if (!param.stopSearch)
        {
            // search stopped due to hard time limit is not considered fully completed
            thread.depthCompleted = depth;
        }

        if (isMainThread &&
            !param.isPonder.load(std::memory_order_acquire))
        {
            // check soft time limit every depth iteration
            if (param.limits.idealTimeCurrent.IsValid() &&
                param.limits.startTimePoint.IsValid() &&
                TimePoint::GetCurrent() >= param.limits.startTimePoint + param.limits.idealTimeCurrent)
            {
                param.stopSearch = true;
                break;
            }

            // check soft node limit
            if (param.limits.maxNodesSoft < UINT64_MAX &&
                searchContext.stats.nodes > param.limits.maxNodesSoft)
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
            primaryMove.IsValid() &&
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

            NodeInfo& rootNode = thread.searchStack[0];
            rootNode = NodeInfo{};
            rootNode.position = game.GetPosition();
            rootNode.isInCheck = rootNode.position.IsInCheck();
            rootNode.position.ComputeThreats(rootNode.threats);
            rootNode.depth = singularDepth;
            rootNode.alpha = singularBeta - 1;
            rootNode.beta = singularBeta;
            rootNode.filteredMove = primaryMove;
            rootNode.nnContext.MarkAsDirty();

            ScoreType score = NegaMax<NodeType::NonPV>(thread, &rootNode, searchContext);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            if (score < singularBeta || CheckStopCondition(thread, searchContext, true))
            {
                param.stopSearch = true;
                break;
            }
        }
    }

    // make sure all threads are stopped
    param.stopSearch = true;
}

PvLine Search::AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param) const
{
    int32_t alpha = -InfValue;
    int32_t beta = InfValue;
    uint32_t depth = param.depth;
    int32_t window = AspirationWindow;

    // increase window based on score
    window += std::abs(param.previousScore) / 16;

    // start applying aspiration window at given depth
    if (param.searchParam.useAspirationWindows &&
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

    // TODO root node could be created in Search_Internal
    NodeInfo& rootNode = thread.searchStack[0];
    rootNode = NodeInfo{};
    rootNode.position = param.position;
    rootNode.isInCheck = param.position.IsInCheck();
    rootNode.position.ComputeThreats(rootNode.threats);
    rootNode.isPvNodeFromPrevIteration = true;
    rootNode.pvIndex = static_cast<uint16_t>(param.pvIndex);
    rootNode.nnContext.MarkAsDirty();

    thread.accumulatorCache.Init(g_mainNeuralNetwork.get());

    for (;;)
    {
        rootNode.depth = static_cast<int16_t>(depth);
        rootNode.alpha = ScoreType(alpha);
        rootNode.beta = ScoreType(beta);

        pvLine.score = NegaMax<NodeType::Root>(thread, &rootNode, param.searchContext);
        ASSERT(pvLine.score >= -CheckmateValue && pvLine.score <= CheckmateValue);
        SearchUtils::GetPvLine(rootNode, maxPvLine, pvLine.moves);

        // flush pending per-thread stats
        param.searchContext.stats.Append(thread.stats, true);

        BoundsType boundsType = BoundsType::Exact;

        // out of aspiration window, redo the search in wider score range
        if (pvLine.score <= alpha)
        {
            pvLine.score = ScoreType(alpha);
            beta = (alpha + beta + 1) / 2;
            alpha = std::max<int32_t>(alpha - window, -CheckmateValue);
            depth = param.depth;
            boundsType = BoundsType::UpperBound;
        }
        else if (pvLine.score >= beta)
        {
            pvLine.score = ScoreType(beta);
            beta = std::min<int32_t>(beta + window, CheckmateValue);
            boundsType = BoundsType::LowerBound;

            // reduce re-search depth
            if (depth > 1 && depth + 3 > param.depth) depth--;
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

        // increase window, fallback to full window after some threshold
        window += window / 3;
        if (window > AspirationWindowMaxSize) window = CheckmateValue;
    }

    return finalPvLine;
}

Search::ThreadData::ThreadData()
{
    randomSeed = 0x4abf372b;
}

const Move Search::ThreadData::GetPvMove(const NodeInfo& node) const
{
    if (!node.isPvNodeFromPrevIteration || pvLines.empty() || node.filteredMove.IsValid())
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

INLINE static bool OppCanWinMaterial(const Position& position, const Threats& threats)
{
    const auto& us = position.GetCurrentSide();
    return (threats.attackedByRooks & us.queens) ||
        (threats.attackedByMinors & (us.queens | us.rooks)) ||
        (threats.attackedByPawns & (us.queens | us.rooks | us.bishops | us.knights));
}

ScoreType Search::ThreadData::GetEvalCorrection(const Position& pos) const
{
    const int32_t matIndex = Murmur3(pos.GetMaterialKey().value) % MaterialCorrectionTableSize;
    const int32_t pawnIndex = pos.GetPawnsHash() % PawnStructureCorrectionTableSize;
    return (matScoreCorrection[matIndex] + pawnStructureCorrection[pawnIndex]) / EvalCorrectionScale;
}

void Search::ThreadData::UpdateEvalCorrection(const Position& pos, ScoreType evalScore, ScoreType trueScore)
{
    int32_t diff = std::clamp<int32_t>(EvalCorrectionScale * (trueScore - evalScore), -32000, 32000);
    if (pos.GetSideToMove() == Black) diff = -diff;

    // material
    {
        const int32_t index = Murmur3(pos.GetMaterialKey().value) % MaterialCorrectionTableSize;
        int16_t& matScore = matScoreCorrection[index];
        matScore = static_cast<int16_t>((matScore * (EvalCorrectionBlendFactor - 1) + diff) / EvalCorrectionBlendFactor);
    }

    // pawn structure
    {
        const int32_t index = pos.GetPawnsHash() % PawnStructureCorrectionTableSize;
        int16_t& pawnScore = pawnStructureCorrection[index];
        pawnScore = static_cast<int16_t>((pawnScore * (EvalCorrectionBlendFactor - 1) + diff) / EvalCorrectionBlendFactor);
    }
}

INLINE static int32_t GetContemptFactor(const Position& pos, const Color rootStm, const SearchParam& searchParam)
{
    int32_t contempt = searchParam.staticContempt;

    if (searchParam.dynamicContempt > 0)
        contempt += (searchParam.dynamicContempt * pos.GetNumPiecesExcludingKing()) / 32;

    if (pos.GetSideToMove() != rootStm)
        contempt = -contempt;

    return contempt;
}

ScoreType Search::AdjustEvalScore(const ThreadData& threadData, const NodeInfo& node, const Color rootStm, const SearchParam& searchParam)
{
    // TODO analyze history moves, scale down when moving same piece all the time

    int32_t adjustedScore = node.staticEval;
    
    if (std::abs(adjustedScore) < KnownWinValue)
    {
        adjustedScore += GetContemptFactor(node.position, rootStm, searchParam);

        // apply eval correction term
        const ScoreType evalCorrection = ScoreType((int32_t)threadData.GetEvalCorrection(node.position) * EvalCorrectionScale / 1024);
        adjustedScore += node.position.GetSideToMove() == White ? evalCorrection : -evalCorrection;

        // scale down when approaching 50-move draw
        adjustedScore = adjustedScore * (256 - std::max(0, (int32_t)node.position.GetHalfMoveCount())) / 256;

        if (searchParam.evalRandomization > 0)
            adjustedScore += ((uint32_t)node.position.GetHash() ^ searchParam.seed) % (2 * searchParam.evalRandomization + 1) - searchParam.evalRandomization;
    }

    return static_cast<ScoreType>(adjustedScore);
}

template<NodeType nodeType>
ScoreType Search::QuiescenceNegaMax(ThreadData& thread, NodeInfo* node, SearchContext& ctx) const
{
    ASSERT(node->height < MaxSearchDepth);
    ASSERT(!node->filteredMove.IsValid());
    ASSERT(node->isInCheck == node->position.IsInCheck());

    constexpr bool isPvNode = nodeType == NodeType::PV || nodeType == NodeType::Root;

    if constexpr (!isPvNode)
        ASSERT(node->alpha == node->beta - 1);
    else
        ASSERT(node->alpha < node->beta);

    // clear PV line
    node->pvLength = 0;

    // update stats
    thread.stats.quiescenceNodes++;
    thread.stats.OnNodeEnter(node->height + 1);
    ctx.stats.Append(thread.stats);

    // Not checking for draw by repetition in the quiescence search
    if (node->previousMove.IsCapture() && CheckInsufficientMaterial(node->position)) [[unlikely]]
        return 0;

    const Position& position = node->position;

    ScoreType alpha = node->alpha;
    ScoreType beta = node->beta;
    ScoreType bestValue = -InfValue;
    ScoreType futilityBase = -InfValue;

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        node->staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node->height, position.GetHalfMoveCount());
        ASSERT(ttScore > -CheckmateValue && ttScore < CheckmateValue);

#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

        // don't prune in PV nodes, because TT does not contain path information
        if constexpr (!isPvNode)
        {
            if (ttEntry.bounds == TTEntry::Bounds::Exact)                           return ttScore;
            else if (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= alpha)  return ttScore;
            else if (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= beta)   return ttScore;
        }
    }

    // do not consider stand pat if in check
    if (node->isInCheck)
    {
        node->staticEval = InvalidValue;
    }
    else
    {
        if (node->staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(*node, thread.accumulatorCache);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);
            node->staticEval = evalScore;

#ifdef COLLECT_SEARCH_STATS
            int32_t binIndex = (evalScore + SearchStats::EvalHistogramMaxValue) * SearchStats::EvalHistogramBins / (2 * SearchStats::EvalHistogramMaxValue);
            binIndex = std::clamp<int32_t>(binIndex, 0, SearchStats::EvalHistogramBins - 1);
            ctx.stats.evalHistogram[binIndex]++;
#endif // COLLECT_SEARCH_STATS
        }

        ASSERT(node->staticEval != InvalidValue);

        const ScoreType adjustedEvalScore = AdjustEvalScore(thread, *node, ctx.game.GetPosition().GetSideToMove(), ctx.searchParam);

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

        if (bestValue >= beta)
        {
            if (!ttEntry.IsValid())
            {
                ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node->height), node->staticEval, 0, TTEntry::Bounds::Lower);
            }
            return bestValue;
        }

        if (bestValue > alpha)
        {
            alpha = bestValue;
        }

        futilityBase = bestValue + static_cast<ScoreType>(QSearchFutilityPruningOffset);
    }

    // guard against overflowing the search stack
    if (node->height >= MaxSearchDepth - 1) [[unlikely]]
    {
        return node->isInCheck ? 0 : bestValue;
    }

    NodeInfo& childNode = *(node + 1);
    childNode.Clear();
    childNode.pvIndex = node->pvIndex;
    childNode.depth = node->depth - 1;
    childNode.height = node->height + 1;
    childNode.nnContext.MarkAsDirty();

    const Square prevSquare = node->previousMove.IsValid() ? node->previousMove.ToSquare() : Square::Invalid();

    MovePicker movePicker(position, thread.moveOrderer, nullptr, ttEntry.move, node->isInCheck);

    int32_t moveScore = 0;
    Move move;

    Move bestMove = Move::Invalid();
    int32_t moveIndex = 0;

    Move captureMovesTried[MoveList::MaxMoves];
    uint32_t numCaptureMovesTried = 0;

    while (movePicker.PickMove(*node, move, moveScore))
    {
        if (bestValue > -TablebaseWinValue && position.HasNonPawnMaterial(position.GetSideToMove()))
        {
            ASSERT(!node->isInCheck);

            // skip underpromotions
            if (move.IsUnderpromotion()) continue;

            // futility pruning - skip captures that won't beat alpha
            if (move.IsCapture() &&
                futilityBase > -KnownWinValue &&
                futilityBase <= alpha &&
                move.ToSquare() != prevSquare &&
                !position.StaticExchangeEvaluation(move, 1))
            {
                bestValue = std::max(bestValue, futilityBase);
                continue;
            }

            // skip very bad captures
            if (moveScore < MoveOrderer::GoodCaptureValue &&
                !position.StaticExchangeEvaluation(move))
                break;
        }

        // start prefetching child node's TT entry
        ctx.searchParam.transpositionTable.Prefetch(position.HashAfterMove(move));

        childNode.position = position;
        if (!childNode.position.DoMove(move, childNode.nnContext))
            continue;
        moveIndex++;

        // Move Count Pruning
        // skip everything after some sane amount of moves has been tried
        // there shouldn't be many "good" captures available in a "normal" chess positions
        if (bestValue > -TablebaseWinValue && moveIndex > 1 && node->depth < 0)
            break;

        childNode.previousMove = move;
        childNode.position.ComputeThreats(childNode.threats);
        childNode.isInCheck = childNode.threats.allThreats & childNode.position.GetCurrentSideKingSquare();
        ASSERT(childNode.isInCheck == childNode.position.IsInCheck());

        childNode.staticEval = InvalidValue;
        childNode.alpha = -beta;
        childNode.beta = -alpha;
        const ScoreType score = -QuiescenceNegaMax<nodeType>(thread, &childNode, ctx);
        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        if (move.IsCapture())
        {
            captureMovesTried[numCaptureMovesTried++] = move;
        }

        if (score > bestValue) // new best move found
        {
            bestValue = score;

            if (score > alpha)
            {
                alpha = score;
                bestMove = move;

                // update PV line
                if constexpr (isPvNode)
                {
                    node->pvLength = std::min<uint16_t>(1u + childNode.pvLength, MaxSearchDepth);
                    node->pvLine[0] = move;
                    memcpy(node->pvLine + 1, childNode.pvLine, sizeof(PackedMove) * std::min<uint16_t>(childNode.pvLength, MaxSearchDepth - 1));
                }

                if (score >= beta)
                {
                    if (bestMove.IsCapture())
                        thread.moveOrderer.UpdateCapturesHistory(*node, captureMovesTried, numCaptureMovesTried, bestMove);

                    break;
                }
            }

            if (node->isInCheck) break; // try only one check evasion
        }
    }

    // no legal moves - checkmate
    if (node->isInCheck && moveIndex == 0)
    {
        return -CheckmateValue + (ScoreType)node->height;
    }

    // store value in transposition table
    const TTEntry::Bounds bounds = bestValue >= beta ? TTEntry::Bounds::Lower : TTEntry::Bounds::Upper;
    ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node->height), node->staticEval, 0, bounds, bestMove);
#ifdef COLLECT_SEARCH_STATS
    ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS

    return bestValue;
}

template<NodeType nodeType>
ScoreType Search::NegaMax(ThreadData& thread, NodeInfo* node, SearchContext& ctx) const
{
    ASSERT(node->height < MaxSearchDepth);

    constexpr bool isRootNode = nodeType == NodeType::Root;
    constexpr bool isPvNode = nodeType == NodeType::PV || nodeType == NodeType::Root;

    if constexpr (!isPvNode)
        ASSERT(node->alpha == node->beta - 1);
    else
        ASSERT(node->alpha < node->beta);

    // clear PV line
    node->pvLength = 0;

    const Position& position = node->position;

    ScoreType alpha = node->alpha;
    ScoreType beta = node->beta;

    // check if we can draw by repetition in losing position
    if constexpr (!isPvNode)
    {
        if (alpha < 0 && SearchUtils::CanReachGameCycle(*node))
        {
            alpha = 0;
            if (alpha >= beta)
            {
                // update stats
                thread.stats.OnNodeEnter(node->height + 1);
                ctx.stats.Append(thread.stats);
                return alpha;
            }
        }
    }

    // maximum search depth reached, enter quiescence search to find final evaluation
    if (node->depth <= 0)
    {
        return QuiescenceNegaMax<nodeType>(thread, node, ctx);
    }

    ASSERT(node->isInCheck == position.IsInCheck());

    // update stats
    thread.stats.OnNodeEnter(node->height + 1);
    ctx.stats.Append(thread.stats);

    if constexpr (!isRootNode)
    {
        // Check for draw
        // Skip root node as we need some move to be reported in PV
        if (node->position.IsFiftyMoveRuleDraw() ||
            CheckInsufficientMaterial(node->position) ||
            SearchUtils::IsRepetition(*node, ctx.game, isPvNode))
        {
            return 0;
        }

        // mate distance pruning
        alpha = std::max<ScoreType>(-CheckmateValue + (ScoreType)node->height, alpha);
        beta = std::min<ScoreType>(CheckmateValue - (ScoreType)node->height - 1, beta);
        if (alpha >= beta)
            return alpha;
    }

    // clear killer moves for next ply
    thread.moveOrderer.ClearKillerMoves(node->height + 1);

    const ScoreType oldAlpha = node->alpha;
    ScoreType bestValue = -InfValue;
    ScoreType eval = InvalidValue;
    ScoreType tbMinValue = -InfValue; // min value according to tablebases
    ScoreType tbMaxValue = InfValue; // max value according to tablebases

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (!node->filteredMove.IsValid() &&
        ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

        node->staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node->height, position.GetHalfMoveCount());
        ASSERT(ttScore > -CheckmateValue && ttScore < CheckmateValue);

        // don't prune in PV nodes, because TT does not contain path information
        if constexpr (!isPvNode)
        {
            if (ttEntry.depth >= node->depth &&
                position.GetHalfMoveCount() < 80)
            {
                // transposition table cutoff
                ScoreType ttCutoffValue = InvalidValue;
                if (ttEntry.bounds == TTEntry::Bounds::Exact)                           ttCutoffValue = ttScore;
                else if (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= alpha)  ttCutoffValue = ttScore;
                else if (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= beta)   ttCutoffValue = ttScore;

                if (ttCutoffValue != InvalidValue)
                    return ttCutoffValue;
            }
            else if ((ttEntry.bounds == TTEntry::Bounds::Upper || ttEntry.bounds == TTEntry::Bounds::Exact) &&
                ttEntry.depth < node->depth && node->depth - ttEntry.depth < 5 &&
                ttScore > -KnownWinValue && alpha < KnownWinValue &&
                ttScore + 128 * (node->depth - ttEntry.depth) <= alpha)
            {
                // accept TT cutoff from shallower search if the score is way below alpha
                return alpha;
            }
        }
    }

    // try probing Win-Draw-Loose endgame tables
    if constexpr (!isRootNode)
    {
        int32_t wdl = 0;
        if (node->depth >= WdlTablebaseProbeDepth &&
            position.GetHalfMoveCount() == 0 &&
            position.GetNumPieces() <= g_syzygyProbeLimit &&
            (ProbeSyzygy_WDL(position, &wdl) || ProbeGaviota(position, nullptr, &wdl))) [[unlikely]]
        {
            thread.stats.tbHits++;

            const ScoreType tbWinScore = TablebaseWinValue - ScoreType(100 * position.GetNumPiecesExcludingKing()) - ScoreType(node->height);
            ASSERT(tbWinScore > KnownWinValue);

            // convert the WDL value to a score
            const ScoreType tbValue = wdl < 0 ? -tbWinScore : wdl > 0 ? tbWinScore : 0;
            ASSERT(tbValue > -CheckmateValue && tbValue < CheckmateValue);

            // only draws are exact, we don't know exact value for win/loss just based on WDL value
            TTEntry::Bounds bounds =
                wdl < 0 ? TTEntry::Bounds::Upper :
                wdl > 0 ? TTEntry::Bounds::Lower :
                TTEntry::Bounds::Exact;

            // clamp best score to tablebase score
            if constexpr (isPvNode)
            {
                if (bounds == TTEntry::Bounds::Lower)
                {
                    alpha = std::max(alpha, bestValue);
                    tbMinValue = tbValue;
                }
                else if (bounds == TTEntry::Bounds::Upper)
                {
                    tbMaxValue = tbValue;
                }
            }

            if ((bounds == TTEntry::Bounds::Exact ||
                (bounds == TTEntry::Bounds::Lower && tbValue >= beta) ||
                (bounds == TTEntry::Bounds::Upper && tbValue <= alpha)))
            {
                if (!ttEntry.IsValid())
                {
                    ctx.searchParam.transpositionTable.Write(position, ScoreToTT(tbValue, node->height), node->staticEval, node->depth, bounds);
#ifdef COLLECT_SEARCH_STATS
                    ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS
                }
                return tbValue;
            }
        }
    }

    // evaluate position
    if (node->isInCheck)
    {
        eval = node->staticEval = InvalidValue;

        if (!node->isCutNode)
        {
            EnsureAccumulatorUpdated(*node, thread.accumulatorCache);
        }
    }
    else
    {
        if (node->staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(*node, thread.accumulatorCache);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);
            node->staticEval = evalScore;

            ctx.searchParam.transpositionTable.Write(position, node->staticEval, node->staticEval, -1, TTEntry::Bounds::Lower);
        }
        else if (!node->isCutNode)
        {
            EnsureAccumulatorUpdated(*node, thread.accumulatorCache);
        }

        ASSERT(node->staticEval != InvalidValue);

        // adjust static eval based on node path
        eval = AdjustEvalScore(thread, *node, ctx.game.GetPosition().GetSideToMove(), ctx.searchParam);

        if (!node->filteredMove.IsValid())
        {
            // try to use TT score for better evaluation estimate
            if (std::abs(ttScore) < KnownWinValue)
            {
                if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > eval) ||
                    (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < eval) ||
                    (ttEntry.bounds == TTEntry::Bounds::Exact))
                {
                    eval = ttScore;
                }
            }
        }
    }

    // guard against overflowing the search stack
    if (node->height >= MaxSearchDepth - 1) [[unlikely]]
    {
        return node->isInCheck ? 0 : eval;
    }

    // check how much static evaluation improved between current position and position in previous turn
    // if we were in check in previous turn, use position prior to it
    bool isImproving = false;
    if (!node->isInCheck)
    {
        int32_t evalImprovement = 0;

        if (node->height > 1 && (node - 2)->staticEval != InvalidValue)
            evalImprovement = node->staticEval - (node - 2)->staticEval;
        else if (node->height > 3 && (node - 4)->staticEval != InvalidValue)
            evalImprovement = node->staticEval - (node - 4)->staticEval;

        isImproving = evalImprovement >= 0;
    }

    if constexpr (!isPvNode)
    {
        if (!node->filteredMove.IsValid() && !node->isInCheck)
        {
            // Reverse Futility Pruning
            if (node->depth <= BetaPruningDepth &&
                eval <= KnownWinValue &&
                eval >= beta + BetaMarginBias + BetaMarginMultiplier * (node->depth - (isImproving && !OppCanWinMaterial(position, node->threats))))
            {
                return (eval + beta) / 2;
            }

            // Razoring
            // prune if quiescence search on current position can't beat beta
            if (node->depth <= RazoringStartDepth &&
                beta < KnownWinValue &&
                eval + RazoringMarginBias + RazoringMarginMultiplier * node->depth < beta)
            {
                const ScoreType qScore = QuiescenceNegaMax<nodeType>(thread, node, ctx);
                if (qScore < beta)
                    return qScore;
            }

            // Null Move Pruning
            if (eval >= beta + (node->depth < 4 ? 20 : 0) &&
                node->staticEval >= beta &&
                node->depth >= NullMovePruningStartDepth &&
                position.HasNonPawnMaterial(position.GetSideToMove()))
            {
                // don't allow null move if parent or grandparent node was null move
                bool doNullMove = !node->isNullMove;
                if (node->height > 0 && (node - 1)->isNullMove) doNullMove = false;

                if (doNullMove)
                {
                    const int32_t r =
                        NullMovePruning_NullMoveDepthReduction +
                        node->depth / 3 +
                        std::min(3, int32_t(eval - beta) / 256) + isImproving;

                    NodeInfo& childNode = *(node + 1);
                    childNode.Clear();
                    childNode.pvIndex = node->pvIndex;
                    childNode.position = position;
                    childNode.alpha = -beta;
                    childNode.beta = -beta + 1;
                    childNode.isNullMove = true;
                    childNode.isCutNode = !node->isCutNode;
                    childNode.doubleExtensions = node->doubleExtensions;
                    childNode.height = node->height + 1;
                    childNode.depth = static_cast<int16_t>(node->depth - r);
                    childNode.nnContext.MarkAsDirty();

                    childNode.position.DoNullMove();
                    childNode.position.ComputeThreats(childNode.threats);

                    ScoreType nullMoveScore = -NegaMax<NodeType::NonPV>(thread, &childNode, ctx);

                    if (nullMoveScore >= beta)
                    {
                        if (nullMoveScore >= TablebaseWinValue)
                            nullMoveScore = beta;

                        if (std::abs(beta) < KnownWinValue && node->depth < 10)
                            return nullMoveScore;

                        node->depth -= static_cast<uint16_t>(NullMovePruning_ReSearchDepthReduction);

                        if (node->depth <= 0)
                        {
                            return QuiescenceNegaMax<nodeType>(thread, node, ctx);
                        }
                    }
                }
            }

            // Probcut
            const ScoreType probBeta = ScoreType(beta + ProbcutBetaOffset);
            if (node->depth >= ProbcutStartDepth &&
                abs(beta) < TablebaseWinValue &&
                !(ttEntry.IsValid() && ttEntry.depth >= node->depth - 3 && ttEntry.score < probBeta))
            {
                NodeInfo& childNode = *(node + 1);
                childNode.Clear();
                childNode.height = node->height + 1;
                childNode.pvIndex = node->pvIndex;
                childNode.doubleExtensions = node->doubleExtensions;
                childNode.nnContext.MarkAsDirty();
                childNode.alpha = -probBeta;
                childNode.beta = -probBeta + 1;
                childNode.isCutNode = !node->isCutNode;

                const ScoreType seeThreshold = probBeta - node->staticEval;
                MovePicker movePicker(position, thread.moveOrderer, nullptr,
                    (ttEntry.move.IsValid() && position.IsCapture(ttEntry.move)) ? ttEntry.move : PackedMove::Invalid(), false);

                int32_t moveScore = 0;
                Move move;
                while (movePicker.PickMove(*node, move, moveScore))
                {
                    if (moveScore < MoveOrderer::GoodCaptureValue && seeThreshold >= 0) continue;
                    if (!position.StaticExchangeEvaluation(move, seeThreshold)) continue;

                    // start prefetching child node's TT entry
                    ctx.searchParam.transpositionTable.Prefetch(position.HashAfterMove(move));

                    childNode.position = position;
                    if (!childNode.position.DoMove(move, childNode.nnContext))
                        continue;

                    childNode.depth = 0;
                    childNode.previousMove = move;
                    childNode.position.ComputeThreats(childNode.threats);
                    childNode.isInCheck = childNode.threats.allThreats & childNode.position.GetCurrentSideKingSquare();
                    ASSERT(childNode.isInCheck == childNode.position.IsInCheck());

                    // quick verification search
                    ScoreType score = -QuiescenceNegaMax<NodeType::NonPV>(thread, &childNode, ctx);
                    ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

                    // verification search
                    if (score >= probBeta)
                    {
                        childNode.depth = node->depth - 4;
                        score = -NegaMax<NodeType::NonPV>(thread, &childNode, ctx);
                    }

                    // probcut failed
                    if (score >= probBeta)
                    {
                        ctx.searchParam.transpositionTable.Write(position, ScoreToTT(score, node->height), node->staticEval, node->depth - 3, TTEntry::Bounds::Lower, move);
                        return score;
                    }
                }
            }
        }
    }

    // reduce depth if position was not found in transposition table
    if (node->depth >= 3 + 4 * node->isCutNode && !ttEntry.IsValid())
    {
        node->depth--;
    }

    const Move pvMove = thread.GetPvMove(*node);
    const PackedMove ttMove = ttEntry.move.IsValid() ? ttEntry.move : pvMove;
    const bool ttCapture = ttMove.IsValid() && (position.IsCapture(ttMove) || ttMove.GetPromoteTo() != Piece::None);

    // in-check probcut (idea from Stockfish)
    if constexpr (!isPvNode)
    {
        const ScoreType probCutBeta = ScoreType(beta + ProbcutBetaOffsetInCheck);
        if (ttCapture && node->isInCheck &&
            ((ttEntry.bounds & TTEntry::Bounds::Lower) != TTEntry::Bounds::Invalid) &&
            ttEntry.depth >= node->depth - 4 && ttScore >= probCutBeta &&
            std::abs(ttScore) < KnownWinValue && std::abs(node->beta) < KnownWinValue)
            return probCutBeta;
    }

    NodeInfo& childNode = *(node + 1);
    childNode.Clear();
    childNode.height = node->height + 1;
    childNode.pvIndex = node->pvIndex;
    childNode.doubleExtensions = node->doubleExtensions;
    childNode.nnContext.MarkAsDirty();

    int32_t extension = 0;

    // check extension
    if (node->isInCheck)
    {
        extension++;
    }

    thread.moveOrderer.InitContinuationHistoryPointers(*node);

    NodeCacheEntry* nodeCacheEntry = nullptr;
    if (node->height < 3)
    {
        nodeCacheEntry = thread.nodeCache.GetEntry(position, node->height);
    }

    MovePicker movePicker(position, thread.moveOrderer, nodeCacheEntry, ttMove, true);

    int32_t moveScore = 0;
    Move move;

    Move bestMove = Move::Invalid();

    uint32_t moveIndex = 0;
    uint32_t quietMoveIndex = 0;
    bool searchAborted = false;
    bool filteredSomeMove = false;

    constexpr uint32_t maxMovesTried = 32;
    Move quietMovesTried[maxMovesTried];
    Move captureMovesTried[maxMovesTried];
    uint32_t numQuietMovesTried = 0;
    uint32_t numCaptureMovesTried = 0;

#ifdef VALIDATE_MOVE_PICKER
    uint32_t numGeneratedMoves = 0;
    Move generatedSoFar[MoveList::MaxMoves];
    int32_t generatedSoFarScores[MoveList::MaxMoves];
#endif // VALIDATE_MOVE_PICKER

    while (movePicker.PickMove(*node, move, moveScore))
    {
        // start prefetching child node's TT entry
        ctx.searchParam.transpositionTable.Prefetch(position.HashAfterMove(move));

#ifdef VALIDATE_MOVE_PICKER
        for (uint32_t i = 0; i < numGeneratedMoves; ++i) ASSERT(generatedSoFar[i] != move);
        generatedSoFarScores[numGeneratedMoves] = moveScore;
        generatedSoFar[numGeneratedMoves++] = move;
#endif // VALIDATE_MOVE_PICKER

        // apply move filter (multi-PV search, singularity search, etc.)
        if (move == node->filteredMove)
        {
            filteredSomeMove = true;
            continue;
        }
        else if constexpr (isRootNode)
        {
            if (ctx.excludedRootMoves.end() != std::find(ctx.excludedRootMoves.begin(), ctx.excludedRootMoves.end(), move))
            {
                filteredSomeMove = true;
                continue;
            }
        }

        int32_t moveStatScore = 0;

        if (move.IsQuiet())
        {
            // compute move stat score using some of history counters
            const uint32_t piece = (uint32_t)move.GetPiece() - 1;
            const uint32_t to = move.ToSquare().Index();
            moveStatScore = (int32_t)thread.moveOrderer.GetHistoryScore(*node, move);
            if (const auto* h = node->continuationHistories[0]) moveStatScore += (*h)[piece][to];
            if (const auto* h = node->continuationHistories[1]) moveStatScore += (*h)[piece][to];
            if (const auto* h = node->continuationHistories[3]) moveStatScore += (*h)[piece][to];

            quietMoveIndex++;
        }

        const bool doPruning = isPvNode ? ctx.searchParam.allowPruningInPvNodes : true;

        if (doPruning &&
            bestValue > -KnownWinValue &&
            position.HasNonPawnMaterial(position.GetSideToMove()))
        {
            if (move.IsQuiet() || move.IsUnderpromotion())
            {
                // Late Move Pruning
                // skip quiet moves that are far in the list
                // the higher depth is, the less aggressive pruning is
                if (quietMoveIndex >= GetLateMovePruningTreshold(node->depth + 2 * isPvNode, isImproving))
                {
                    // if we're in quiets stage, skip everything
                    if (movePicker.GetStage() == MovePicker::Stage::PickQuiets) break;

                    continue;
                }

                // History Pruning
                // if a move score is really bad, do not consider this move at low depth
                if (quietMoveIndex > 1 &&
                    node->depth < 9 &&
                    moveStatScore < GetHistoryPruningTreshold(node->depth))
                {
                    continue;
                }

                // Futility Pruning
                // skip quiet move that have low chance to beat alpha
                if (!node->isInCheck &&
                    node->depth < FutilityPruningDepth &&
                    node->staticEval + FutilityPruningScale * node->depth * node->depth + moveStatScore / FutilityPruningStatscoreDiv < alpha)
                {
                    movePicker.SkipQuiets();
                    if (quietMoveIndex > 1) continue;
                }
            }

            // Static Exchange Evaluation pruning - skip all moves that are bad according to SEE
            // the higher depth is, the less aggressive pruning is
            if (move.ToSquare().GetBitboard() & node->threats.allThreats)
            {
                if (move.IsCapture())
                {
                    if (node->depth <= 4 &&
                        moveScore < MoveOrderer::GoodCaptureValue &&
                        !position.StaticExchangeEvaluation(move, -SSEPruningMultiplier_Captures * node->depth)) continue;
                }
                else
                {
                    if (node->depth <= 8 &&
                        !position.StaticExchangeEvaluation(move, -SSEPruningMultiplier_NonCaptures * node->depth)) continue;
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
        if constexpr (!isRootNode)
        {
            if (!node->filteredMove.IsValid() &&
                move == ttMove &&
                node->depth >= SingularExtensionMinDepth &&
                std::abs(ttScore) < KnownWinValue &&
                ((ttEntry.bounds & TTEntry::Bounds::Lower) != TTEntry::Bounds::Invalid) &&
                ttEntry.depth >= node->depth - 3)
            {
                const ScoreType singularBeta = (ScoreType)std::max(-CheckmateValue, (int32_t)ttScore - node->depth);

                const bool originalIsPvNodeFromPrevIteration = node->isPvNodeFromPrevIteration;
                const int16_t originalDepth = node->depth;
                const ScoreType originalAlpha = node->alpha;
                const ScoreType originalBeta = node->beta;

                node->isPvNodeFromPrevIteration = false;
                node->depth = node->depth / 2 - 1;
                node->alpha = singularBeta - 1;
                node->beta = singularBeta;
                node->filteredMove = move;
                const ScoreType singularScore = NegaMax<NodeType::NonPV>(thread, node, ctx);

                // restore node state
                node->isPvNodeFromPrevIteration = originalIsPvNodeFromPrevIteration;
                node->depth = originalDepth;
                node->alpha = originalAlpha;
                node->beta = originalBeta;
                node->filteredMove = PackedMove::Invalid();

                if (singularScore < singularBeta)
                {
                    if (node->height < 2 * thread.rootDepth)
                    {
                        moveExtension = 1;
                        // double extension if singular score is way below beta
                        if constexpr (!isPvNode)
                            if (node->doubleExtensions <= 6 && singularScore < singularBeta - SingularDoubleExtensionMarigin)
                                moveExtension = 2;
                    }
                }
                // if second best move beats current beta, there most likely would be beta cutoff
                // when searching it at full depth
                else if (singularBeta >= beta)
                    return singularBeta;
                else if (ttScore >= beta)
                    moveExtension = -2 - !isPvNode;
                else if (node->isCutNode)
                    moveExtension = -1;
                else if (ttScore <= alpha)
                    moveExtension = -1;
            }
        }

        // do the move
        childNode.position = position;
        if (!childNode.position.DoMove(move, childNode.nnContext))
            continue;
        moveIndex++;

        // report current move to UCI
        if constexpr (isRootNode)
        {
            if (thread.isMainThread && ctx.searchParam.debugLog && node->pvIndex == 0)
            {
                const float timeElapsed = (TimePoint::GetCurrent() - ctx.searchParam.limits.startTimePoint).ToSeconds();
                if (timeElapsed > CurrentMoveReportDelay)
                {
                    ReportCurrentMove(move, node->depth, moveIndex + node->pvIndex);
                }
            }
        }

        childNode.staticEval = InvalidValue;
        childNode.position.ComputeThreats(childNode.threats);
        childNode.isInCheck = childNode.threats.allThreats & childNode.position.GetCurrentSideKingSquare();
        childNode.previousMove = move;
        childNode.moveStatScore = moveStatScore;
        childNode.isPvNodeFromPrevIteration = node->isPvNodeFromPrevIteration && (move == pvMove);
        childNode.doubleExtensions = node->doubleExtensions + (moveExtension >= 2);

        const uint64_t nodesSearchedBefore = thread.stats.nodesTotal;

        // Late Move Reductions
        int32_t r = 0;
        if (node->depth >= LateMoveReductionStartDepth &&
            moveIndex > 1 &&
            (!isPvNode || move.IsQuiet()))
        {
            if (move.IsQuiet())
            {
                r = GetQuietsDepthReduction(node->depth, moveIndex);

                // reduce non-PV nodes more
                if constexpr (!isPvNode) r++;

                // reduce more if TT move is capture
                if (ttCapture) r++;

                // reduce good moves less
                if (moveScore >= MoveOrderer::CounterMoveBonus) r -= 2;

                // reduce less based on move stat score
                r -= DivFloor<int32_t>(moveStatScore + ReductionStatOffset, ReductionStatDiv);

                if (node->isCutNode) r += 2;
            }
            else
            {
                r = GetCapturesDepthReduction(node->depth, moveIndex);

                // reduce winning captures less
                if (moveScore > MoveOrderer::WinningCaptureValue) r--;

                // reduce bad captures more
                if (moveScore < MoveOrderer::GoodCaptureValue) r++;

                if (node->isCutNode) r++;
            }

            // reduce more if eval is not improving
            if (!isImproving) r++;

            // reduce less if move is a check
            if (childNode.isInCheck) r--;
        }

        int32_t newDepth = node->depth + moveExtension - 1;

        // limit reduction, don't drop into QS
        r = std::clamp(r, 0, newDepth);

        ScoreType score = InvalidValue;

        bool doFullDepthSearch = !(isPvNode && moveIndex == 1);

        // PVS search at reduced depth
        if (r > 0)
        {
            ASSERT(moveIndex > 1);

            const int32_t lmrDepth = newDepth - r;
            childNode.depth = static_cast<int16_t>(lmrDepth);
            childNode.alpha = -alpha - 1;
            childNode.beta = -alpha;
            childNode.isCutNode = true;

            score = -NegaMax<NodeType::NonPV>(thread, &childNode, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            if (score > alpha)
            {
                newDepth += (score > bestValue + 80) && (node->height < 2 * thread.rootDepth); // prevent search explosions
                newDepth -= (score < bestValue + newDepth);
                doFullDepthSearch = newDepth > lmrDepth;
            }
            else
            {
                doFullDepthSearch = false;
            }
        }

        // PVS search at full depth
        if (doFullDepthSearch) [[unlikely]]
        {
            childNode.depth = static_cast<int16_t>(newDepth);
            childNode.alpha = -alpha - 1;
            childNode.beta = -alpha;
            childNode.isCutNode = !node->isCutNode;

            score = -NegaMax<NodeType::NonPV>(thread, &childNode, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
        }

        // full search for PV nodes
        if constexpr (isPvNode)
        {
            if (moveIndex == 1 ||
                (score > alpha && (isRootNode || score < beta)))
            {
                childNode.depth = static_cast<int16_t>(newDepth);
                childNode.alpha = -beta;
                childNode.beta = -alpha;
                childNode.isCutNode = false;

                score = -NegaMax<NodeType::PV>(thread, &childNode, ctx);
            }
        }

        // update node cache after searching a move
        if (nodeCacheEntry) [[unlikely]]
        {
            ASSERT(thread.stats.nodesTotal > nodesSearchedBefore);
            const uint64_t nodesSearched = thread.stats.nodesTotal - nodesSearchedBefore;
            nodeCacheEntry->AddMoveStats(move, nodesSearched);
        }

        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        if (move.IsQuiet() && numQuietMovesTried < maxMovesTried)
        {
            quietMovesTried[numQuietMovesTried++] = move;
        }
        else if (move.IsCapture() && numCaptureMovesTried < maxMovesTried)
        {
            captureMovesTried[numCaptureMovesTried++] = move;
        }

        if (score > bestValue) // new best move found
        {
            bestValue = score;

            // make sure we have any best move in root node
            if constexpr (isRootNode) bestMove = move;

            // update PV line
            if constexpr (isPvNode)
            {
                if (!node->filteredMove.IsValid()) // don't overwrite PV in singular search
                {
                    node->pvLength = std::min<uint16_t>(1u + childNode.pvLength, MaxSearchDepth);
                    node->pvLine[0] = move;
                    memcpy(node->pvLine + 1, childNode.pvLine, sizeof(PackedMove) * std::min<uint16_t>(childNode.pvLength, MaxSearchDepth - 1));
                }
            }
        }

        if (score > alpha)
        {
            alpha = score;
            bestMove = move;

            if (score >= beta)
            {
                ASSERT(moveIndex > 0);
                ASSERT(moveIndex <= MoveList::MaxMoves);

#ifdef COLLECT_SEARCH_STATS
                ctx.stats.totalBetaCutoffs++;
                ctx.stats.betaCutoffHistogram[moveIndex - 1]++;
                if (moveScore == MoveOrderer::TTMoveValue) ctx.stats.ttMoveBetaCutoffs++;
                else if (moveScore == MoveOrderer::KillerMoveBonus) ctx.stats.killerMoveBetaCutoffs++;
                else if (moveScore == MoveOrderer::CounterMoveBonus) ctx.stats.counterMoveBetaCutoffs++;
                else if (move.IsCapture() && moveScore >= MoveOrderer::WinningCaptureValue) ctx.stats.winningCaptureCutoffs++;
                else if (move.IsCapture() && moveScore >= MoveOrderer::GoodCaptureValue) ctx.stats.goodCaptureCutoffs++;
                else if (move.IsCapture() && moveScore < MoveOrderer::GoodCaptureValue) ctx.stats.badCaptureCutoffs++;
                else if (move.IsQuiet()) ctx.stats.quietCutoffs++;
#endif // COLLECT_SEARCH_STATS

                break;
            }

            // reduce remaining moves more if we managed to find new best move
            if (node->depth > 2) node->depth--;
        }

        if constexpr (!isRootNode)
        {
            if (CheckStopCondition(thread, ctx, false))
            {
                // abort search of further moves
                searchAborted = true;
                break;
            }
        }
    }

    // no legal moves
    if (!searchAborted && moveIndex == 0u)
    {
        if (filteredSomeMove)
            bestValue = -InfValue;
        else
            bestValue = node->isInCheck ? -CheckmateValue + (ScoreType)node->height : 0;

        return bestValue;
    }

    // update move orderer
    if (bestValue >= beta)
    {
        if (bestMove.IsQuiet())
        {
            thread.moveOrderer.UpdateQuietMovesHistory(*node, quietMovesTried, numQuietMovesTried, bestMove, std::min(bestValue - beta, 256));
            thread.moveOrderer.UpdateKillerMove(node->height, bestMove);
        }
        thread.moveOrderer.UpdateCapturesHistory(*node, captureMovesTried, numCaptureMovesTried, bestMove);
    }

#ifdef COLLECT_SEARCH_STATS
    {
        const bool isCutNode = bestValue >= beta;

        if (isCutNode)                      ctx.stats.numCutNodes++;
        else if (bestValue > oldAlpha)      ctx.stats.numPvNodes++;
        else                                ctx.stats.numAllNodes++;

        if (node->isCutNode == isCutNode)   ctx.stats.expectedCutNodesSuccess++;
        else                                ctx.stats.expectedCutNodesFailure++;
    }
#endif // COLLECT_SEARCH_STATS

    ASSERT(bestValue >= -CheckmateValue && bestValue <= CheckmateValue);

    if constexpr (isRootNode)
    {
        ASSERT(bestMove.IsValid());
        ASSERT(!isPvNode || node->pvLength > 0);
        ASSERT(!isPvNode || node->pvLine[0] == bestMove);
    }

    // clamp score to TB bounds
    if constexpr (isPvNode)
        bestValue = std::clamp(bestValue, tbMinValue, tbMaxValue);

    // update transposition table
    // don't write if:
    // - time is exceeded as evaluation may be inaccurate
    // - some move was skipped due to filtering, because 'bestMove' may not be "the best" for the current position
    if (!filteredSomeMove && !CheckStopCondition(thread, ctx, false))
    {
        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::Lower :
            bestValue > oldAlpha ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::Upper;

        // only PV nodes can have exact score
        if constexpr (!isPvNode)
            ASSERT(bounds != TTEntry::Bounds::Exact);

        ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node->height), node->staticEval, node->depth, bounds, bestMove);

#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS

        // if we beat alpha, adjust material score
        if (node->depth >= 1 &&
            !node->isInCheck &&
            bestMove.IsQuiet() &&
            (bounds == TTEntry::Bounds::Exact ||
             (bounds == TTEntry::Bounds::Lower && bestValue >= node->staticEval) ||
             (bounds == TTEntry::Bounds::Upper && bestValue <= node->staticEval)))
        {
            thread.UpdateEvalCorrection(node->position, node->staticEval, bestValue);
        }
    }

    return bestValue;
}
