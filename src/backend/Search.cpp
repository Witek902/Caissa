#include "Search.hpp"
#include "MovePicker.hpp"
#include "Game.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"
#include "TranspositionTable.hpp"
#include "Tablebase.hpp"

#ifdef USE_TABLE_BASES
    #include "tablebase/tbprobe.h"
#endif // USE_TABLE_BASES

#include <iostream>
#include <cstring>
#include <string>
#include <thread>
#include <math.h>

static const float CurrentMoveReportDelay = 10.0f;

static const uint8_t SingularitySearchPvIndex = UINT8_MAX;
static const uint32_t SingularitySearchMinDepth = 7;
static const int32_t SingularitySearchScoreTresholdMin = 200;
static const int32_t SingularitySearchScoreTresholdMax = 500;
static const int32_t SingularitySearchScoreStep = 50;

static const uint32_t DefaultMaxPvLineLength = 20;
static const uint32_t MateCountStopCondition = 5;

static const int32_t TablebaseProbeDepth = 0;

static const int32_t NullMoveReductionsStartDepth = 2;
static const int32_t NullMoveReductions_NullMoveDepthReduction = 4;
static const int32_t NullMoveReductions_ReSearchDepthReduction = 4;

static const int32_t LateMoveReductionStartDepth = 3;

static const int32_t AspirationWindowMax = 50;
static const int32_t AspirationWindowMin = 15;
static const int32_t AspirationWindowStep = 5;

static const int32_t BetaPruningDepth = 8;
static const int32_t BetaMarginMultiplier = 200;
static const int32_t BetaMarginBias = 10;

static const int32_t QSearchSeeMargin = -50;
static const uint32_t MaxQSearchMoves = 20;

static const int32_t LateMovePruningStartDepth = 10;

INLINE static uint32_t GetLateMovePrunningTreshold(uint32_t depth)
{
    return 5 + depth * depth;
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
            const int32_t reduction = int32_t(0.5f + 0.5f * logf(float(depth + 1)) * logf(float(moveIndex + 1)));

            ASSERT(reduction <= 64);
            mMoveReductionTable[depth][moveIndex] = (uint8_t)std::min<int32_t>(reduction, UINT8_MAX);
        }
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

NO_INLINE bool Search::CheckStopCondition(const SearchContext& ctx, bool isRootNode) const
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
        if (isRootNode || (ctx.stats.nodes % 64 == 0))
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

void Search::GetPvLine(const ThreadData& thread, const Position& pos, const TranspositionTable& tt, uint32_t maxLength, std::vector<Move>& outLine)
{
    outLine.clear();

    uint32_t pvLength = thread.pvLengths[0];

    if (pvLength > 0)
    {
        uint32_t i;

        // reconstruct PV line using PV array
        Position iteratedPosition = pos;
        for (i = 0; i < pvLength; ++i)
        {
            const Move move = iteratedPosition.MoveFromPacked(thread.pvArray[0][i]);

            if (!move.IsValid()) break;
            if (!iteratedPosition.DoMove(move)) break;

            outLine.push_back(move);
        }

        // reconstruct PV line using transposition table
        for (; i < maxLength; ++i)
        {
            if (iteratedPosition.GetNumLegalMoves() == 0) break;

            TTEntry ttEntry;
            if (!tt.Read(iteratedPosition, ttEntry)) break;

            const Move move = iteratedPosition.MoveFromPacked(ttEntry.moves[0]);

            // Note: move in transpostion table may be invalid due to hash collision
            if (!move.IsValid()) break;
            if (!iteratedPosition.DoMove(move)) break;

            outLine.push_back(move);
        }

        ASSERT(!outLine.empty());
    }
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

    mThreadData.resize(param.numThreads);
    mThreadData[0].isMainThread = true;

    if (param.numThreads > 1)
    {
        std::vector<std::thread> threads;
        threads.reserve(param.numThreads);

        for (uint32_t i = param.numThreads; i-- > 0; )
        {
            threads.emplace_back([this, i, numPvLines, &game, &param, &outResult]() INLINE_LAMBDA
            {
                Search_Internal(i, numPvLines, game, param, outResult);
            });
        }

        for (uint32_t threadID = 0; threadID < param.numThreads; ++threadID)
        {
            threads[threadID].join();
        }
    }
    else
    {
        Search_Internal(0, numPvLines, game, param, outResult);
    }
}

void Search::ReportPV(const AspirationWindowSearchParam& param, const PvLine& pvLine, BoundsType boundsType, const TimePoint& searchTime) const
{
    std::cout << "info";
    std::cout << " depth " << param.depth;
    std::cout << " seldepth " << (uint32_t)param.searchContext.stats.maxDepth;
    if (param.searchParam.numPvLines > 1)
    {
        std::cout << " multipv " << (param.pvIndex + 1);
    }

    if (pvLine.score > CheckmateValue - MaxSearchDepth)
    {
        std::cout << " score mate " << (CheckmateValue - pvLine.score + 1) / 2;
    }
    else if (pvLine.score < -CheckmateValue + MaxSearchDepth)
    {
        std::cout << " score mate -" << (CheckmateValue + pvLine.score + 1) / 2;
    }
    else
    {
        std::cout << " score cp " << pvLine.score;
    }

    if (boundsType == BoundsType::LowerBound)
    {
        std::cout << " lowerbound";
    }
    if (boundsType == BoundsType::UpperBound)
    {
        std::cout << " upperbound";
    }

    const float timeInSeconds = searchTime.ToSeconds();
    const uint64_t numNodes = param.searchContext.stats.nodes.load();

    std::cout << " nodes " << numNodes;

    if (timeInSeconds > 0.01f && numNodes > 100)
    {
        std::cout << " nps " << (int64_t)((double)numNodes / (double)timeInSeconds);
    }

#ifdef COLLECT_SEARCH_STATS
    if (param.searchContext.stats.tbHits)
    {
        std::cout << " tbhit " << param.searchContext.stats.tbHits;
    }
#endif // COLLECT_SEARCH_STATS

    std::cout << " time " << static_cast<int64_t>(0.5f + 1000.0f * timeInSeconds);

    std::cout << " pv ";
    {
        Position tempPosition = param.position;
        for (size_t i = 0; i < pvLine.moves.size(); ++i)
        {
            const Move move = pvLine.moves[i];
            ASSERT(move.IsValid());
            std::cout << tempPosition.MoveToString(move, param.searchParam.moveNotation);
            if (i + 1 < pvLine.moves.size()) std::cout << ' ';
            tempPosition.DoMove(move);
        }
    }

    std::cout << std::endl;

#ifdef COLLECT_SEARCH_STATS
    if (param.searchParam.verboseStats)
    {
        std::cout << "Beta cutoff histogram\n";
        uint32_t maxMoveIndex = 0;
        uint64_t sum = 0;
        for (uint32_t i = 0; i < MoveList::MaxMoves; ++i)
        {
            if (param.searchContext.stats.betaCutoffHistogram[i])
            {
                sum += param.searchContext.stats.betaCutoffHistogram[i];
                maxMoveIndex = std::max(maxMoveIndex, i);
            }
        }
        for (uint32_t i = 0; i <= maxMoveIndex; ++i)
        {
            const uint64_t value = param.searchContext.stats.betaCutoffHistogram[i];
            printf("    %u : %" PRIu64 " (%.2f%%)\n", i, value, 100.0f * float(value) / float(sum));
        }
    }
#endif // COLLECT_SEARCH_STATS
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
    return score > CheckmateValue - MaxSearchDepth || score < -CheckmateValue + MaxSearchDepth;
}

void Search::Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, const SearchParam& param, SearchResult& outResult)
{
    const bool isMainThread = threadID == 0;
    ThreadData& thread = mThreadData[threadID];

    std::vector<Move> pvMovesSoFar;
    pvMovesSoFar.reserve(param.excludedMoves.size() + numPvLines);

    outResult.resize(numPvLines);

    thread.moveOrderer.Clear();
    thread.prevPvLines.clear();
    thread.prevPvLines.resize(numPvLines);

    uint32_t mateCounter = 0;

    SearchContext searchContext{ game, param, SearchStats{} };

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
                        prevScore = ttEntry.score;
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
                !pvMovesSoFar.empty() ? (uint8_t)pvMovesSoFar.size() : 0u,
                prevScore,
                threadID,
            };

            PvLine pvLine = AspirationWindowSearch(thread, aspirationWindowSearchParam);

            // stop search only at depth 2 and more
            if (depth > 1 && CheckStopCondition(searchContext, true))
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

        // rememeber PV lines so they can be used in next iteration
        thread.prevPvLines = std::move(tempResult);

        // check soft time limit every depth iteration
        if (isMainThread &&
            !param.isPonder &&
            param.limits.maxTimeSoft.IsValid() &&
            TimePoint::GetCurrent() >= param.limits.maxTimeSoft)
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
            std::abs(primaryMoveScore) < KnownWinValue &&
            param.limits.rootSingularityTime.IsValid() &&
            TimePoint::GetCurrent() >= param.limits.rootSingularityTime)
        {
            const int32_t scoreTreshold = std::max<int32_t>(SingularitySearchScoreTresholdMin, SingularitySearchScoreTresholdMax - SingularitySearchScoreStep * (depth - SingularitySearchMinDepth));

            const uint32_t singularDepth = depth / 2;
            const ScoreType singularBeta = primaryMoveScore - (ScoreType)scoreTreshold;

            NodeInfo rootNode;
            rootNode.position = game.GetPosition();
            rootNode.isPvNode = false;
            rootNode.depth = singularDepth;
            rootNode.pvIndex = SingularitySearchPvIndex;
            rootNode.alpha = singularBeta - 1;
            rootNode.beta = singularBeta;
            rootNode.moveFilter = &primaryMove;
            rootNode.moveFilterCount = 1;

            ScoreType score = NegaMax(thread, rootNode, searchContext);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            if (score < singularBeta || CheckStopCondition(searchContext, true))
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

    // start applying aspiration window at given depth
    if (param.previousScore != InvalidValue &&
        !CheckStopCondition(param.searchContext, true))
    {
        alpha = std::max<int32_t>(param.previousScore - aspirationWindow, -InfValue);
        beta = std::min<int32_t>(param.previousScore + aspirationWindow, InfValue);
    }

    PvLine pvLine; // working copy
    PvLine finalPvLine;

    for (;;)
    {
        memset(thread.pvLengths, 0, sizeof(ThreadData::pvLengths));

        NodeInfo rootNode;
        rootNode.position = param.position;
        rootNode.isPvNode = true;
        rootNode.depth = param.depth;
        rootNode.pvIndex = param.pvIndex;
        rootNode.alpha = ScoreType(alpha);
        rootNode.beta = ScoreType(beta);
        rootNode.moveFilter = param.moveFilter;
        rootNode.moveFilterCount = param.moveFilterCount;

        pvLine.score = NegaMax(thread, rootNode, param.searchContext);
        ASSERT(pvLine.score >= -CheckmateValue && pvLine.score <= CheckmateValue);

        // limit PV line length so the output is not flooded
        {
            uint32_t maxPvLength = param.depth;
            if (!param.searchParam.limits.analysisMode) maxPvLength = std::min(maxPvLength, DefaultMaxPvLineLength);
            GetPvLine(thread, param.position, param.searchParam.transpositionTable, maxPvLength, pvLine.moves);
        }

        BoundsType boundsType = BoundsType::Exact;
        
        // out of aspiration window, redo the search in wider score range
        if (pvLine.score <= alpha)
        {
            //beta = alpha + 1;
            //beta = std::min<int32_t>(alpha + 1, CheckmateValue);
            pvLine.score = ScoreType(alpha);
            alpha -= aspirationWindow;
            alpha = std::max<int32_t>(alpha, -CheckmateValue);
            aspirationWindow *= 4;
            boundsType = BoundsType::UpperBound;
        }
        else if (pvLine.score >= beta)
        {
            //alpha = beta - 1;
            //alpha = std::max<int32_t>(beta - 1, -CheckmateValue);
            pvLine.score = ScoreType(beta);
            beta += aspirationWindow;
            beta = std::min<int32_t>(beta, CheckmateValue);
            aspirationWindow *= 4;
            boundsType = BoundsType::LowerBound;
        }

        const bool stopSearch = param.depth > 1 && CheckStopCondition(param.searchContext, true);
        const bool isMainThread = param.threadID == 0;

        // don't report line if search was aborted, because the result comes from incomplete search
        if (!stopSearch)
        {
            ASSERT(!pvLine.moves.empty());
            ASSERT(pvLine.moves.front().IsValid());

            if (isMainThread && param.searchParam.debugLog)
            {
                const TimePoint searchTime = TimePoint::GetCurrent() - param.searchParam.limits.startTimePoint;
                ReportPV(param, pvLine, boundsType, searchTime);
            }

            finalPvLine = std::move(pvLine);
        }

        // stop the search when exact score is found
        if (boundsType == BoundsType::Exact)
        {
            break;
        }
    }

    return finalPvLine;
}

static INLINE ScoreType ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}

const Move Search::ThreadData::FindPvMove(const NodeInfo& node, MoveList& moves) const
{
    if (!node.isPvNode || prevPvLines.empty() || node.pvIndex == SingularitySearchPvIndex)
    {
        return Move::Invalid();
    }

    const std::vector<Move>& pvLine = prevPvLines[node.pvIndex].moves;
    if (node.height >= pvLine.size())
    {
        return Move::Invalid();
    }

    if (node.height >= pvLine.size())
    {
        return Move::Invalid();
    }

    const Move pvMove = pvLine[node.height];
    ASSERT(pvMove.IsValid());

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        if (pvMove.IsValid() && moves[i].move == pvMove)
        {
            moves[i].score = MoveOrderer::PVMoveValue;
            return pvMove;
        }
    }

    // no PV move found?
    return pvMove;
}

void Search::ThreadData::UpdatePvArray(uint32_t depth, const Move move)
{
    if (depth + 1 < MaxSearchDepth)
    {
        const uint8_t childPvLength = pvLengths[depth + 1];
        pvArray[depth][depth] = move;
        for (uint32_t j = depth + 1; j < childPvLength; ++j)
        {
            pvArray[depth][j] = pvArray[depth + 1][j];
        }
        pvLengths[depth] = childPvLength;
    }
}

bool Search::IsRepetition(const NodeInfo& node, const Game& game) const
{
    const NodeInfo* prevNode = &node;

    for (uint32_t ply = 1; ; ++ply)
    {
        // don't need to check more moves if reached pawn push or capture,
        // because these moves are irreversible
        if (prevNode->previousMove.IsValid())
        {
            if (prevNode->previousMove.GetPiece() == Piece::Pawn || prevNode->previousMove.IsCapture())
            {
                return false;
            }
        }

        prevNode = prevNode->parentNode;

        // reached end of the stack
        if (!prevNode)
        {
            break;
        }

        // only check every second previous node, because side to move must be the same
        if (ply % 2 == 0)
        {
            ASSERT(prevNode->position.GetSideToMove() == node.position.GetSideToMove());

            if (prevNode->position.GetHash() == node.position.GetHash())
            {
                if (prevNode->position == node.position)
                {
                    return true;
                }
            }
        }
    }

    return game.GetRepetitionCount(node.position) >= 2;
}

bool Search::IsDraw(const NodeInfo& node, const Game& game) const
{
    if (node.position.GetHalfMoveCount() >= 100)
    {
        return true;
    }

    if (CheckInsufficientMaterial(node.position))
    {
        return true;
    }

    if (IsRepetition(node, game))
    {
        return true;
    }

    return false;
}

ScoreType Search::QuiescenceNegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const
{
    ASSERT(node.depth <= 0);
    ASSERT(node.alpha <= node.beta);
    ASSERT(node.isPvNode || node.alpha == node.beta - 1);
    ASSERT(node.moveFilterCount == 0);
    ASSERT(node.height > 0);

    // clean PV line
    if (node.height < MaxSearchDepth)
    {
        thread.pvLengths[node.height] = (uint8_t)node.height;
    }

    // update stats
    ctx.stats.quiescenceNodes++;
    ctx.stats.maxDepth = std::max<uint32_t>(ctx.stats.maxDepth, node.height);

    if (IsDraw(node, ctx.game))
    {
        return 0;
    }

    const Position& position = node.position;

    const bool isPvNode = node.isPvNode;

    ScoreType alpha = node.alpha;
    ScoreType oldAlpha = alpha;
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

        if (!isPvNode && ttEntry.depth >= node.depth)
        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

            if ((ttEntry.bounds == TTEntry::Bounds::Exact) ||
                (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= node.beta) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < node.beta))
            {
                return ttScore;
            }
        }
    }

    const bool isInCheck = position.IsInCheck(position.GetSideToMove());

    // do not consider stand pat if in check
    if (!isInCheck)
    {
        if (staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(position);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);
            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;
        }

        // try to use TT score for better evaluation estimate
        if (ttScore != InvalidValue)
        {
            if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < staticEval))
            {
                staticEval = ttScore;
            }
        }

        bestValue = staticEval;

        if (bestValue >= beta)
        {
            ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, node.depth, TTEntry::Bounds::Lower);
            return bestValue;
        }

        if (isPvNode && bestValue > alpha)
        {
            alpha = bestValue;
        }
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.pvIndex = node.pvIndex;
    childNodeParam.isPvNode = node.isPvNode;
    childNodeParam.depth = node.depth - 1;
    childNodeParam.height = node.height + 1;

    uint32_t moveGenFlags = 0;
    if (!isInCheck)
    {
        moveGenFlags |= MOVE_GEN_ONLY_TACTICAL;
        moveGenFlags |= MOVE_GEN_ONLY_QUEEN_PROMOTIONS;
    }

    MovePicker movePicker(position, thread.moveOrderer, ttEntry, moveGenFlags);

    int32_t moveScore = 0;
    Move move;

    Move bestMoves[TTEntry::NumMoves];
    uint32_t numBestMoves = 0;
    uint32_t moveIndex = 0;

    while (movePicker.PickMove(node, move, moveScore))
    {
        ASSERT(move.IsValid());

        // skip losing captures
        if (!isInCheck && move.IsCapture() && bestValue > -KnownWinValue && moveScore < MoveOrderer::GoodCaptureValue)
        {
            if (!position.StaticExchangeEvaluation(move, QSearchSeeMargin))
            {
                continue;
            }
        }

        childNodeParam.position = position;
        if (!childNodeParam.position.DoMove(move))
        {
            continue;
        }

        ctx.searchParam.transpositionTable.Prefetch(childNodeParam.position);

        moveIndex++;

        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;
        ScoreType score = -QuiescenceNegaMax(thread, childNodeParam, ctx);

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

            if (isPvNode)
            {
                thread.UpdatePvArray(node.height, move);
            }

            if (score > alpha) // update lower bound
            {
                if (isPvNode && score < beta) // keep alpha < beta
                {
                    alpha = score;
                }
                else
                {
                    ASSERT(score >= beta);
                    ASSERT(alpha < beta);
                    break;
                }
            }
        }

        // skip everything after some sane amount of moves has been tried
        // there shouldn't be more than 20 captures available in a "normal" chess positions
        if (!isInCheck && bestValue > -KnownWinValue && moveIndex >= MaxQSearchMoves)
        {
            break;
        }
    }

    // no legal moves - checkmate
    if (isInCheck && moveIndex == 0u)
    {
        return -CheckmateValue + (ScoreType)node.height;
    }

    // store value in transposition table
    if (!CheckStopCondition(ctx, false))
    {
        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::Lower :
            (isPvNode && bestValue > oldAlpha) ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::Upper;

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


static ScoreType PruneByMateDistance(const NodeInfo& node, ScoreType& alpha, ScoreType& beta)
{
    ScoreType matingValue = CheckmateValue - (ScoreType)node.height;
    if (matingValue < beta)
    {
        beta = matingValue;
        if (alpha >= matingValue)
        {
            return matingValue;
        }
    }

    matingValue = -CheckmateValue + (ScoreType)node.height;
    if (matingValue > alpha)
    {
        alpha = matingValue;
        if (beta <= matingValue)
        {
            return matingValue;
        }
    }

    return 0;
}

ScoreType Search::NegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx) const
{
    ASSERT(node.alpha <= node.beta);
    ASSERT(node.isPvNode || node.alpha == node.beta - 1);

    // clean PV line
    if (node.height < MaxSearchDepth)
    {
        thread.pvLengths[node.height] = (uint8_t)node.height;
    }

    // update stats
    ctx.stats.nodes++;
    ctx.stats.maxDepth = std::max<uint32_t>(ctx.stats.maxDepth, node.height);

    const bool isRootNode = node.height == 0; // root node is the first node in the chain (best move)
    const bool isPvNode = node.isPvNode;
    const bool hasMoveFilter = node.moveFilterCount > 0u;

    // maximum search depth reached, enter quisence search to find final evaluation
    if (node.depth <= 0)
    {
        return QuiescenceNegaMax(thread, node, ctx);
    }

    // Check for draw
    // Skip root node as we need some move to be reported
    if (!isRootNode && IsDraw(node, ctx.game))
    {
        return 0;
    }

    const Position& position = node.position;

    // TODO use proper stack
    const NodeInfo* prevNodes[4] = { nullptr };
    prevNodes[0] = node.parentNode;
    prevNodes[1] = prevNodes[0] ? prevNodes[0]->parentNode : nullptr;
    prevNodes[2] = prevNodes[1] ? prevNodes[1]->parentNode : nullptr;
    prevNodes[3] = prevNodes[2] ? prevNodes[2]->parentNode : nullptr;
    
    const bool isInCheck = position.IsInCheck(position.GetSideToMove());

    const ScoreType oldAlpha = node.alpha;
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

        // don't prune in PV nodes, because TT does not contain path information
        if (ttEntry.depth >= node.depth &&
            (node.depth == 0 || !isPvNode) &&
            !hasMoveFilter &&
            position.GetHalfMoveCount() < 90)
        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

            if (ttEntry.bounds == TTEntry::Bounds::Exact)
            {
                return ttScore;
            }
            else if (ttEntry.bounds == TTEntry::Bounds::Upper)
            {
                if (ttScore <= alpha) return alpha;
                if (ttScore < beta) beta = ttScore;
            }
            else if (ttEntry.bounds == TTEntry::Bounds::Lower)
            {
                if (ttScore >= beta) return beta;
                if (ttScore > alpha) alpha = ttScore;
            }
        }
    }

    // mate distance pruning
    if (!isRootNode)
    {
        const ScoreType mateDistanceScore = PruneByMateDistance(node, alpha, beta);
        if (mateDistanceScore != 0)
        {
            return mateDistanceScore;
        }
    }

    // try probing Win-Draw-Loose endgame tables
    {
        int32_t wdl = 0;
        if (!isRootNode &&
            node.depth >= TablebaseProbeDepth &&
            ProbeTablebase_WDL(position, &wdl))
        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.tbHits++;
#endif // COLLECT_SEARCH_STATS

            // convert the WDL value to a score
            const ScoreType tbValue =
                wdl < 0 ? -(TablebaseWinValue - (ScoreType)node.height) :
                wdl > 0 ? (TablebaseWinValue - (ScoreType)node.height) : 0;
            ASSERT(tbValue > -CheckmateValue && tbValue < CheckmateValue);

            // only draws are exact, we don't know exact value for win/loss just based on WDL value
            const TTEntry::Bounds bounds =
                wdl < 0 ? TTEntry::Bounds::Upper :
                wdl > 0 ? TTEntry::Bounds::Lower :
                TTEntry::Bounds::Exact;

            if (bounds == TTEntry::Bounds::Exact
                || (bounds == TTEntry::Bounds::Lower && tbValue >= beta)
                || (bounds == TTEntry::Bounds::Upper && tbValue <= alpha))
            {
                ctx.searchParam.transpositionTable.Write(position, ScoreToTT(tbValue, node.height), staticEval, node.depth, bounds);

#ifdef COLLECT_SEARCH_STATS
                ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS

                return tbValue;
            }
        }
    }

    // evaluate position if it wasn't evaluated
    bool wasPositionEvaluated = true;
    if (!isInCheck)
    {
        if (staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(position);
            ASSERT(evalScore < TablebaseWinValue&& evalScore > -TablebaseWinValue);
            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;
            wasPositionEvaluated = false;
        }

        // try to use TT score for better evaluation estimate
        if (ttScore != InvalidValue)
        {
            if ((ttEntry.bounds == TTEntry::Bounds::Lower && ttScore > staticEval) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore < staticEval))
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
    const bool isImproving = evalImprovement > 0;

    // Futility/Beta Pruning
    if (!isPvNode &&
        !isInCheck &&
        node.depth <= BetaPruningDepth &&
        staticEval <= TablebaseWinValue &&
        !ctx.searchParam.limits.mateSearch)
    {
        const int32_t betaMargin = BetaMarginBias + BetaMarginMultiplier * (node.depth - isImproving);

        if (staticEval - betaMargin >= beta)
        {
            return (ScoreType)std::max<int32_t>(-TablebaseWinValue, staticEval);
        }
    }

    // Null Move Reductions
    if (!isRootNode &&
        !isPvNode &&
        !isInCheck &&
        !hasMoveFilter &&
        staticEval >= beta &&
        node.depth >= NullMoveReductionsStartDepth &&
        !ctx.searchParam.limits.mateSearch &&
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

            NodeInfo childNodeParam;
            childNodeParam.parentNode = &node;
            childNodeParam.pvIndex = node.pvIndex;
            childNodeParam.position = position;
            childNodeParam.alpha = -beta;
            childNodeParam.beta = -beta + 1;
            childNodeParam.isNullMove = true;
            childNodeParam.height = node.height + 1;
            childNodeParam.depth = node.depth - depthReduction;

            childNodeParam.position.DoNullMove();

            ScoreType nullMoveScore = -NegaMax(thread, childNodeParam, ctx);

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

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.height = node.height + 1;
    childNodeParam.pvIndex = node.pvIndex;

    int32_t extension = 0;

    // check extension
    if (isInCheck)
    {
        extension++;
    }

    MovePicker movePicker(position, thread.moveOrderer, ttEntry);

    // randomize move order for root node on secondary threads
    if (isRootNode && !thread.isMainThread)
    {
        movePicker.Shuffle();
    }

    int32_t moveScore = 0;
    Move move;

    Move bestMoves[TTEntry::NumMoves] = { Move::Invalid(), Move::Invalid(), Move::Invalid() };
    uint32_t numBestMoves = 0;

    uint32_t moveIndex = 0;
    uint32_t numReducedMoves = 0;
    bool searchAborted = false;
    bool filteredSomeMove = false;

    Move quietMovesTried[MoveList::MaxMoves];
    uint32_t numQuietMovesTried = 0;

    while (movePicker.PickMove(node, move, moveScore))
    {
        ASSERT(move.IsValid());

        // apply node filter (multi-PV search, singularity search, etc.)
        if (!node.ShouldCheckMove(move))
        {
            filteredSomeMove = true;
            continue;
        }

        childNodeParam.position = position;
        if (!childNodeParam.position.DoMove(move))
        {
            continue;
        }

        // start prefetching child node's TT entry
        ctx.searchParam.transpositionTable.Prefetch(childNodeParam.position);

        moveIndex++;

        // report current move to UCI
        if (isRootNode && thread.isMainThread && ctx.searchParam.debugLog && node.pvIndex != SingularitySearchPvIndex)
        {
            const float timeElapsed = (TimePoint::GetCurrent() - ctx.searchParam.limits.startTimePoint).ToSeconds();
            if (timeElapsed > CurrentMoveReportDelay)
            {
                ReportCurrentMove(move, node.depth, moveIndex + node.pvIndex);
            }
        }

        if (move.IsQuiet())
        {
            quietMovesTried[numQuietMovesTried++] = move;
        }

        // Static Exchange Evaluation pruning
        // skip all moves that are bad according to SEE
        // the higher depth is, the less agressing pruning is
        if (!isInCheck &&
            bestValue > -KnownWinValue &&
            node.depth <= 9)
        {
            const int32_t pruningTreshold = 16 * node.depth * node.depth;
            if (!position.StaticExchangeEvaluation(move, -pruningTreshold))
            {
                continue;
            }
        }

        // Late Move Pruning
        // skip quiet moves that are far in the list
        // the higher depth is, the less agressing pruning is
        if (move.IsQuiet() &&
            moveScore < 0 &&
            !isInCheck &&
            bestValue > -KnownWinValue &&
            node.depth <= LateMovePruningStartDepth &&
            moveIndex >= GetLateMovePrunningTreshold(node.depth))
        {
            continue;
        }

        int32_t moveExtension = extension;

        // avoid extending search too much (maximum 2x depth at root node)
        if (node.height < 2 * thread.rootDepth)
        {
            // promotion extension
            if (move.GetPromoteTo() == Piece::Queen)
            {
                moveExtension++;
            }

            // extend if there's only one legal move
            if (movePicker.IsOnlyOneLegalMove())
            {
                moveExtension++;
            }

            // Singular move extension
            if (moveExtension <= 1 &&
                !isRootNode &&
                !hasMoveFilter &&
                move == ttEntry.moves[0] &&
                node.depth >= SingularitySearchMinDepth &&
                std::abs(ttScore) < KnownWinValue &&
                ((ttEntry.bounds & TTEntry::Bounds::Lower) != TTEntry::Bounds::Invalid) &&
                ttEntry.depth >= node.depth - 3)
            {
                const uint32_t singularDepth = (node.depth - 1) / 2;
                const ScoreType singularBeta = (ScoreType)std::max(-CheckmateValue, (int32_t)ttScore - 8 * node.depth);

                NodeInfo singularChildNode = node;
                singularChildNode.isPvNode = false;
                singularChildNode.depth = singularDepth;
                singularChildNode.pvIndex = SingularitySearchPvIndex;
                singularChildNode.alpha = singularBeta - 1;
                singularChildNode.beta = singularBeta;
                singularChildNode.moveFilter = &move;
                singularChildNode.moveFilterCount = 1;

                const ScoreType singularScore = NegaMax(thread, singularChildNode, ctx);

                if (singularScore < singularBeta)
                {
                    moveExtension++;
                }
                //else if (singularBeta >= node.beta)
                //{
                //    return singularBeta;
                //}
            }
        }

        // avoid search explosion
        moveExtension = std::clamp(moveExtension, 0, 2);

        childNodeParam.previousMove = move;

        int32_t depthReduction = 0;

        // Late Move Reduction
        // don't reduce PV moves, while in check, good captures, promotions, etc.
        if (moveIndex > 1 &&
            !isRootNode &&
            node.depth >= LateMoveReductionStartDepth &&
            bestValue > -CheckmateValue &&
            !isInCheck &&
            !ctx.searchParam.limits.mateSearch &&
            move.IsQuiet())
        {
            // reduce depth gradually
            depthReduction = mMoveReductionTable[node.depth][std::min(moveIndex, MaxReducedMoves - 1)];

            if (!node.isPvNode) depthReduction += 1;

            // reduce good moves less
            if (moveScore > 0 && depthReduction > 0) depthReduction--;

            // don't drop into QS
            depthReduction = std::min(depthReduction, node.depth + moveExtension - 1);

            numReducedMoves++;
        }

        ScoreType score = InvalidValue;

        bool doFullDepthSearch = !(isPvNode && moveIndex == 1);

        // PVS search at reduced depth
        if (depthReduction > 0)
        {
            childNodeParam.depth = node.depth + moveExtension - 1 - (int32_t)depthReduction;
            childNodeParam.alpha = -alpha - 1;
            childNodeParam.beta = -alpha;
            childNodeParam.isPvNode = false;

            score = -NegaMax(thread, childNodeParam, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            doFullDepthSearch = score > alpha;
        }

        // PVS search at full depth
        // TODO: internal aspiration window?
        if (doFullDepthSearch)
        {
            childNodeParam.depth = node.depth + moveExtension - 1;
            childNodeParam.alpha = -alpha - 1;
            childNodeParam.beta = -alpha;
            childNodeParam.isPvNode = false;

            score = -NegaMax(thread, childNodeParam, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
        }

        // full search for PV nodes
        if (isPvNode)
        {
            if (moveIndex == 1 || score > alpha)
            {
                childNodeParam.depth = node.depth + moveExtension - 1;
                childNodeParam.alpha = -beta;
                childNodeParam.beta = -alpha;
                childNodeParam.isPvNode = true;

                score = -NegaMax(thread, childNodeParam, ctx);
                ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
            }
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

            if (isPvNode)
            {
                thread.UpdatePvArray(node.height, move);
            }

            if (score > alpha) // update lower bound
            {
                if (isPvNode && score < beta) // keep alpha < beta
                {
                    alpha = score;
                }
                else
                {
                    ASSERT(moveIndex > 0);
                    ASSERT(score >= beta);
                    ASSERT(alpha < beta);

#ifdef COLLECT_SEARCH_STATS
                    ctx.stats.betaCutoffHistogram[moveIndex - 1]++;
#endif // COLLECT_SEARCH_STATS

                    break;
                }
            }
        }

        if (!isRootNode && CheckStopCondition(ctx, false))
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
        bestValue = isInCheck ? -CheckmateValue + (ScoreType)node.height : 0;
        return bestValue;
    }

    ASSERT(alpha < beta);
    ASSERT(bestValue >= -CheckmateValue && bestValue <= CheckmateValue);
    
    if (isPvNode && isRootNode)
    {
        ASSERT(numBestMoves > 0);
        ASSERT(thread.pvLengths[0] > 0);
    }

    // update transposition table
    // don't write if:
    // - time is exceeded as evaluation may be inaccurate
    // - some move was skipped due to filtering, because 'bestMove' may not be "the best" for the current position
    if (!filteredSomeMove && !CheckStopCondition(ctx, false))
    {
        ASSERT(numBestMoves > 0);

        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::Lower :
            bestValue > oldAlpha ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::Upper;

        // only PV nodes can have exact score
        if (!isPvNode)
        {
            ASSERT(bounds != TTEntry::Bounds::Exact);
        }

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
