#include "Search.hpp"
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

static const uint8_t SingularitySearchPvIndex = UINT8_MAX;
static const uint32_t SingularitySearchMinDepth = 7;
static const int32_t SingularitySearchScoreTreshold = 400;

static const bool UsePVS = true;

static const uint32_t DefaultMaxPvLineLength = 20;
static const uint32_t MateCountStopCondition = 5;

static const int32_t TablebaseProbeDepth = 0;

static const int32_t NullMoveReductionsStartDepth = 2;
static const int32_t NullMoveReductions_NullMoveDepthReduction = 4;
static const int32_t NullMoveReductions_ReSearchDepthReduction = 4;

static const int32_t LateMoveReductionStartDepth = 3;

static const int32_t AspirationWindowSearchStartDepth = 2;
static const int32_t AspirationWindowMax = 60;
static const int32_t AspirationWindowMin = 10;
static const int32_t AspirationWindowStep = 5;

static const int32_t BetaPruningDepth = 8;
static const int32_t BetaMarginMultiplier = 200;
static const int32_t BetaMarginBias = 10;

static const int32_t QSearchSeeMargin = 120;

int64_t SearchParam::GetElapsedTime() const
{
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint).count();
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
            const int32_t reduction = int32_t(0.5f + logf(float(depth + 1)) * logf(float(moveIndex + 1)) / 2.0f);

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

bool Search::CheckStopCondition(const SearchContext& ctx) const
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
            return true;
        }

        if (ctx.searchParam.limits.maxTime < UINT32_MAX &&
            ctx.searchParam.GetElapsedTime() >= ctx.searchParam.limits.maxTime)
        {
            // time limit exceeded
            return true;
        }
    }

    return false;
}

std::vector<Move> Search::GetPvLine(const ThreadData& thread, const Position& pos, const TranspositionTable& tt, uint32_t maxLength)
{
    std::vector<Move> moves;

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

            moves.push_back(move);
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

            moves.push_back(move);
        }

        ASSERT(!moves.empty());
    }

    return moves;
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
        if (param.limits.maxTime < UINT32_MAX && numLegalMoves == 1)
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

void Search::ReportPV(const AspirationWindowSearchParam& param, const PvLine& pvLine, BoundsType boundsType, const std::chrono::high_resolution_clock::duration searchTime) const
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

    std::cout << " time " << std::chrono::duration_cast<std::chrono::milliseconds>(searchTime).count();

    std::cout << " nodes " << param.searchContext.stats.nodes;
    std::cout << " nps " << (int32_t)(1.0e9 * (double)param.searchContext.stats.nodes / (double)std::chrono::duration_cast<std::chrono::nanoseconds>(searchTime).count());

#ifdef COLLECT_SEARCH_STATS
    if (param.searchContext.stats.tbHits)
    {
        std::cout << " tbhit " << param.searchContext.stats.tbHits;
    }
#endif // COLLECT_SEARCH_STATS

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

    // main iterative deepening loop
    for (uint32_t depth = 1; depth <= param.limits.maxDepth; ++depth)
    {
        SearchResult tempResult;
        tempResult.resize(numPvLines);

        pvMovesSoFar.clear();
        pvMovesSoFar = param.excludedMoves;

        bool finishSearchAtDepth = false;

        for (uint32_t pvIndex = 0; pvIndex < numPvLines; ++pvIndex)
        {
            PvLine& prevPvLine = thread.prevPvLines[pvIndex];

            SearchContext searchContext{ game, param, SearchStats{} };

            const AspirationWindowSearchParam aspirationWindowSearchParam =
            {
                game.GetPosition(),
                param,
                depth,
                (uint8_t)pvIndex,
                searchContext,
                !pvMovesSoFar.empty() ? pvMovesSoFar.data() : nullptr,
                !pvMovesSoFar.empty() ? (uint8_t)pvMovesSoFar.size() : 0u,
                prevPvLine.score,
                threadID,
            };

            PvLine pvLine = AspirationWindowSearch(thread, aspirationWindowSearchParam);
            ASSERT(pvLine.score > -CheckmateValue && pvLine.score < CheckmateValue);
            ASSERT(!pvLine.moves.empty());

            if (IsMate(pvLine.score))
            {
                mateCounter++;
            }
            else
            {
                mateCounter = 0;
            }

            // store for multi-PV filtering in next iteration
            for (const Move prevMove : pvMovesSoFar)
            {
                ASSERT(prevMove != pvLine.moves.front());
            }
            pvMovesSoFar.push_back(pvLine.moves.front());

            // stop search only at depth 2 and more
            if (depth > 1 && CheckStopCondition(searchContext))
            {
                finishSearchAtDepth = true;
            }
            else
            {
                if (isMainThread)
                {
                    ASSERT(!pvLine.moves.empty());
                    outResult[pvIndex] = pvLine;
                }
            }

            tempResult[pvIndex] = std::move(pvLine);
        }

        if (finishSearchAtDepth)
        {
            if (isMainThread)
            {
                ASSERT(outResult[0].moves.size() > 0);

                // stop other threads
                StopSearch();
            }
            break;
        }

        // check for singular root move
        if (isMainThread &&
            numPvLines == 1 &&
            depth >= SingularitySearchMinDepth &&
            std::abs(tempResult[0].score) < KnownWinValue &&
            param.limits.rootSingularityTime < UINT32_MAX &&
            param.GetElapsedTime() >= param.limits.rootSingularityTime)
        {
            SearchContext searchContext{ game, param, SearchStats{} };

            const uint32_t singularDepth = depth / 2;
            const ScoreType singularBeta = tempResult[0].score - SingularitySearchScoreTreshold;

            NodeInfo rootNode;
            rootNode.position = game.GetPosition();
            rootNode.isPvNode = false;
            rootNode.depth = singularDepth;
            rootNode.height = 0;
            rootNode.pvIndex = SingularitySearchPvIndex;
            rootNode.alpha = singularBeta - 1;
            rootNode.beta = singularBeta;
            rootNode.moveFilter = &tempResult[0].moves[0];
            rootNode.moveFilterCount = 1;

            ScoreType score = NegaMax(thread, rootNode, searchContext);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            if (score < singularBeta)
            {
                StopSearch();
                break;
            }
        }

        // rememeber PV lines so they can be used in next iteration
        thread.prevPvLines = std::move(tempResult);

        // check soft time limit every depth iteration
        if (isMainThread &&
            !param.isPonder &&
            param.limits.maxTimeSoft < UINT32_MAX &&
            param.GetElapsedTime() >= param.limits.maxTimeSoft)
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
    }
}

PvLine Search::AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param) const
{
    int32_t alpha = -InfValue;
    int32_t beta = InfValue;

    // decrease aspiration window with increasing depth
    int32_t aspirationWindow = AspirationWindowMax - (param.depth - AspirationWindowSearchStartDepth) * AspirationWindowStep;
    aspirationWindow = std::max<int32_t>(AspirationWindowMin, aspirationWindow);
    ASSERT(aspirationWindow > 0);

    // start applying aspiration window at given depth
    if (param.depth >= AspirationWindowSearchStartDepth &&
        !CheckStopCondition(param.searchContext))
    {
        alpha = std::max<int32_t>(param.previousScore - aspirationWindow, -InfValue);
        beta = std::min<int32_t>(param.previousScore + aspirationWindow, InfValue);
    }

    PvLine pvLine;

    for (;;)
    {
        //std::cout << "aspiration window: " << alpha << "..." << beta << "\n";

        const auto startTime = std::chrono::high_resolution_clock::now();

        memset(thread.pvArray, 0, sizeof(ThreadData::pvArray));
        memset(thread.pvLengths, 0, sizeof(ThreadData::pvLengths));

        NodeInfo rootNode;
        rootNode.position = param.position;
        rootNode.isPvNode = true;
        rootNode.depth = param.depth;
        rootNode.height = 0;
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
            pvLine.moves = GetPvLine(thread, param.position, param.searchParam.transpositionTable, maxPvLength);
        }

        const auto endTime = std::chrono::high_resolution_clock::now();
        const std::chrono::high_resolution_clock::duration searchTime = endTime - startTime;

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

        const bool stopSearch = param.depth > 1 && CheckStopCondition(param.searchContext);
        const bool isMainThread = param.threadID == 0;

        // don't report line if search was aborted, because the result comes from incomplete search
        if (isMainThread && !stopSearch && param.pvIndex != SingularitySearchPvIndex)
        {
            ASSERT(!pvLine.moves.empty());

            if (param.searchParam.debugLog)
            {
                ReportPV(param, pvLine, boundsType, searchTime);
            }
        }

        // stop the search when exact score is found
        if (boundsType == BoundsType::Exact)
        {
            break;
        }
    }

    return pvLine;
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
    //ASSERT(false);
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

    ScoreType staticEval = InvalidValue;

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
        ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

        if (ttEntry.depth >= node.depth)
        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

            if ((ttEntry.bounds == TTEntry::Bounds::Exact) ||
                (ttEntry.bounds == TTEntry::Bounds::Lower && ttScore >= node.beta) ||
                (ttEntry.bounds == TTEntry::Bounds::Upper && ttScore <= node.alpha))
            {
                return ttScore;
            }
        }
    }

    const bool isInCheck = position.IsInCheck(position.GetSideToMove());

    ScoreType alpha = node.alpha;
    ScoreType oldAlpha = alpha;
    ScoreType beta = node.beta;
    ScoreType bestValue = -InfValue;

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

    MoveList moves;
    position.GenerateMoveList(moves, moveGenFlags);

    //const int32_t sseTreshold = std::max(0, alpha - staticEval) - QSearchSeeMargin;

    // resolve move scoring
    // the idea here is to defer scoring if we have a TT/PV move
    // most likely we'll get beta cutoff on it so we won't need to score any other move
    uint32_t numScoredMoves = moves.numMoves;
    if (moves.numMoves > 1u)
    {
        numScoredMoves = 0;

        if (moves.AssignTTScores(ttEntry).IsValid())
        {
            numScoredMoves++;
        }
    }

    Move bestMoves[TTEntry::NumMoves];
    uint32_t numBestMoves = 0;
    uint32_t moveIndex = 0;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        if (i == numScoredMoves)
        {
            // we reached a point where moves are not scored anymore, so score them now
            thread.moveOrderer.ScoreMoves(node, moves);
        }

        int32_t moveScore = 0;
        const Move move = moves.PickBestMove(i, moveScore);

        // skip losing captures
        //if (!isInCheck && move.IsCapture() && bestValue > -KnownWinValue)
        //{
        //    if (!position.StaticExchangeEvaluation(move, sseTreshold))
        //    {
        //        continue;
        //    }
        //}

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
            for (uint32_t j = 1; j <= numBestMoves && j < TTEntry::NumMoves; ++j)
            {
                bestMoves[j] = bestMoves[j - 1];
            }
            numBestMoves = std::min(TTEntry::NumMoves, numBestMoves + 1);
            bestMoves[0] = move;
            bestValue = score;

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
    }

    // no legal moves - checkmate
    if (isInCheck && moveIndex == 0u)
    {
        return -CheckmateValue + (ScoreType)node.height;
    }

    // store value in transposition table
    if (!CheckStopCondition(ctx))
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

    // Check for draw
    // Skip root node as we need some move to be reported
    if (!isRootNode && IsDraw(node, ctx.game))
    {
        return 0;
    }

    const Position& position = node.position;

    // maximum search depth reached, enter quisence search to find final evaluation
    if (node.depth <= 0)
    {
        return QuiescenceNegaMax(thread, node, ctx);
    }

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
    ScoreType maxValue = CheckmateValue; // max score limited by tablebase
    ScoreType staticEval = InvalidValue;

    // transposition table lookup
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        staticEval = ttEntry.staticEval;

        ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
        ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

        // don't prune in PV nodes, because TT does not contain path information
        if (ttEntry.depth >= node.depth &&
            (node.depth == 0 || !isPvNode) &&
            !hasMoveFilter)
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

    // mate distance prunning
    if (!isRootNode)
    {
        const ScoreType mateDistanceScore = PruneByMateDistance(node, alpha, beta);
        if (mateDistanceScore != 0)
        {
            return mateDistanceScore;
        }
    }

#ifdef USE_TABLE_BASES
    // probe endgame tables
    if (!isRootNode && HasTablebases())
    {
        if (position.GetNumPieces() <= TB_LARGEST &&
            node.depth >= TablebaseProbeDepth &&
            position.GetBlacksCastlingRights() == 0 && position.GetWhitesCastlingRights() == 0)
        {
            // TODO skip if too many pieces, obvious wins, etc.
            const uint32_t probeResult = tb_probe_wdl(
                position.Whites().Occupied(),
                position.Blacks().Occupied(),
                position.Whites().king | position.Blacks().king,
                position.Whites().queens | position.Blacks().queens,
                position.Whites().rooks | position.Blacks().rooks,
                position.Whites().bishops | position.Blacks().bishops,
                position.Whites().knights | position.Blacks().knights,
                position.Whites().pawns | position.Blacks().pawns,
                position.GetHalfMoveCount(),
                0, // TODO castling rights
                position.GetEnPassantSquare().mIndex,
                position.GetSideToMove() == Color::White);

            if (probeResult != TB_RESULT_FAILED)
            {
#ifdef COLLECT_SEARCH_STATS
                ctx.stats.tbHits++;
#endif // COLLECT_SEARCH_STATS

                // convert the WDL value to a score
                const ScoreType tbValue =
                    probeResult == TB_LOSS ? -(TablebaseWinValue - (ScoreType)node.height) :
                    probeResult == TB_WIN  ?  (TablebaseWinValue - (ScoreType)node.height) : 0;
                ASSERT(tbValue > -CheckmateValue && tbValue < CheckmateValue);

                // only draws are exact, we don't know exact value for win/loss just based on WDL value
                const TTEntry::Bounds bounds =
                    probeResult == TB_LOSS ? TTEntry::Bounds::Upper :
                    probeResult == TB_WIN  ? TTEntry::Bounds::Lower :
                    TTEntry::Bounds::Exact;

                if (    bounds == TTEntry::Bounds::Exact
                    || (bounds == TTEntry::Bounds::Lower && tbValue >= beta)
                    || (bounds == TTEntry::Bounds::Upper && tbValue <= alpha))
                {
                    ctx.searchParam.transpositionTable.Write(
                        position,
                        ScoreToTT(tbValue, node.height), staticEval,
                        bounds == TTEntry::Bounds::Exact ? INT8_MAX : node.depth,
                        bounds);

#ifdef COLLECT_SEARCH_STATS
                    ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS

                    return tbValue;
                }

                //if (isPvNode)
                //{
                //    if (bounds == TTEntry::Bounds::Lower)
                //    {
                //        bestValue = tbValue;
                //        alpha = std::max(alpha, tbValue);
                //    }
                //    else
                //    {
                //        maxValue = tbValue;
                //    }
                //}
            }
        }
    }
#endif // USE_TABLE_BASES

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
            if (!wasPositionEvaluated)
            {
                ctx.searchParam.transpositionTable.Write(position, staticEval, staticEval, INT8_MIN, TTEntry::Bounds::Exact);
            }
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

    MoveList moves;
    position.GenerateMoveList(moves);

    // resolve move scoring
    // the idea here is to defer scoring if we have a TT/PV move
    // most likely we'll get beta cutoff on it so we won't need to score any other move
    uint32_t numScoredMoves = moves.numMoves;
    if (moves.numMoves > 1u)
    {
        const Move pvMove = thread.FindPvMove(node, moves);
        const Move ttMove = moves.AssignTTScores(ttEntry);

        numScoredMoves = 0;

        if (pvMove.IsValid()) numScoredMoves++;
        if (ttMove.IsValid()) numScoredMoves++;
    }

#ifndef CONFIGURATION_FINAL
    if (isRootNode)
    {
        if (ctx.searchParam.printMoves)
        {
            moves.Print();
        }
    }
#endif // CONFIGURATION_FINAL

    // randomize move order for root node on secondary threads
    const bool shuffleMoves = isRootNode && !thread.isMainThread;
    if (shuffleMoves)
    {
        moves.Shuffle();
    }

    Move bestMoves[TTEntry::NumMoves] = { Move::Invalid() };
    uint32_t numBestMoves = 0;

    uint32_t moveIndex = 0;
    uint32_t numReducedMoves = 0;
    bool searchAborted = false;
    bool filteredSomeMove = false;

    Move quietMovesTried[256];
    uint32_t numQuietMovesTried = 0;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        if (i == numScoredMoves && !shuffleMoves)
        {
            // We reached a point where moves are not scored anymore, so score them now
            // Randomize move order a bit for non-main threads
            thread.moveOrderer.ScoreMoves(node, moves, thread.isMainThread ? 0 : 0b11);
        }

        int32_t moveScore = 0;
        const Move move = moves.PickBestMove(i, moveScore);
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

        ctx.searchParam.transpositionTable.Prefetch(childNodeParam.position);

        moveIndex++;

        if (move.IsQuiet()) quietMovesTried[numQuietMovesTried++] = move;

        int32_t moveExtension = extension;

        // promotion extension
        if (move.GetPromoteTo() == Piece::Queen)
        {
            moveExtension++;
        }

        // extend if there's only one legal move
        if (moveIndex == 1 && i + 1 == moves.Size())
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
        }

        // avoid search explosion
        moveExtension = std::min(moveExtension, 2);

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
            if (moveScore >= MoveOrderer::KillerMoveBonus && depthReduction > 0) depthReduction--;

            // don't drop into QS
            depthReduction = std::min(depthReduction, node.depth + moveExtension - 1);

            numReducedMoves++;
        }

        ScoreType score = InvalidValue;

        if (UsePVS)
        {
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
        }
        else
        {
            // search at reduced depth
            if (depthReduction > 0)
            {
                childNodeParam.depth = node.depth + moveExtension - 1 - (int32_t)depthReduction;
                childNodeParam.alpha = -beta;
                childNodeParam.beta = -alpha;
                childNodeParam.isPvNode = true;

                score = -NegaMax(thread, childNodeParam, ctx);
                ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
            }

            // full depth re-search
            if (depthReduction <= 0 || score > alpha)
            {
                childNodeParam.depth = node.depth + moveExtension - 1;
                childNodeParam.alpha = -beta;
                childNodeParam.beta = -alpha;
                childNodeParam.isPvNode = true;

                score = -NegaMax(thread, childNodeParam, ctx);
                ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
            }
        }

        if (isRootNode)
        {
            if (ctx.searchParam.printMoves)
            {
                std::cout << move.ToString() << " eval=" << score << " alpha=" << alpha << " beta=" << beta;
                if (score > alpha) std::cout << " !!!";
                std::cout << std::endl;
            }
        }

        if (score > bestValue) // new best move found
        {
            // push new best move to the beginning of the list
            for (uint32_t j = 1; j <= numBestMoves && j < TTEntry::NumMoves; ++j)
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

        if (!isRootNode && CheckStopCondition(ctx))
        {
            // abort search of further moves
            searchAborted = true;
            break;
        }
    }

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

    // limit by TB
    bestValue = std::min(bestValue, maxValue);

    // update transposition table
    // don't write if:
    // - time is exceeded as evaluation may be inaccurate
    // - some move was skipped due to filtering, because 'bestMove' may not be "the best" for the current position
    if (!filteredSomeMove && !CheckStopCondition(ctx))
    {
        ASSERT(numBestMoves > 0);

        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::Lower :
            bestValue > oldAlpha ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::Upper;

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
