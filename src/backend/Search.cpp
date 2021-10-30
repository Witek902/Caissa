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

static const bool UsePVS = true;

static const int32_t TablebaseProbeDepth = 0;

static const int32_t NullMovePrunningStartDepth = 3;
static const int32_t NullMovePrunningDepthReduction = 3;

static const bool UseLateMoveReduction = true;
static const int32_t LateMoveReductionStartDepth = 2;
static const int32_t LateMoveReductionRate = 8;

static const int32_t LateMovePrunningStartDepth = 3;

static const int32_t AspirationWindowSearchStartDepth = 2;
static const int32_t AspirationWindowMax = 60;
static const int32_t AspirationWindowMin = 10;
static const int32_t AspirationWindowStep = 5;

static const int32_t BetaPruningDepth = 6;
static const int32_t BetaMarginMultiplier = 80;
static const int32_t BetaMarginBias = 10;

static const int32_t AlphaPruningDepth = 4;
static const int32_t AlphaMarginMultiplier = 600;
static const int32_t AlphaMarginBias = 150;

static const int32_t QSearchDeltaMargin = 80;
static const int32_t QSearchSeeMargin = 120;

int64_t SearchParam::GetElapsedTime() const
{
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint).count();
}

Search::Search()
{
    mThreadData.resize(1);
}

Search::~Search()
{
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

            const Move move = iteratedPosition.MoveFromPacked(ttEntry.move);

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

    if (param.numThreads > 1)
    {
        std::vector<std::thread> threads;
        threads.reserve(param.numThreads);

        for (uint32_t i = 0; i < param.numThreads; ++i)
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
        mThreadData.resize(param.numThreads);
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
    std::cout << " time " << std::chrono::duration_cast<std::chrono::milliseconds>(searchTime).count();

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

    std::cout << " nodes " << param.searchContext.stats.nodes;
    std::cout << " nps " << (int32_t)(1.0e9f * (float)param.searchContext.stats.nodes / (float)std::chrono::duration_cast<std::chrono::nanoseconds>(searchTime).count());

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

    std::cout << std::endl << std::flush;

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

void Search::Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, const SearchParam& param, SearchResult& outResult)
{
    const bool isMainThread = threadID == 0;
    ThreadData& thread = mThreadData[threadID];

    std::vector<Move> pvMovesSoFar;

    outResult.resize(numPvLines);

    thread.moveOrderer.Clear();
    thread.prevPvLines.clear();
    thread.prevPvLines.resize(numPvLines);

    // main iterative deepening loop
    for (uint32_t depth = 1; depth <= param.limits.maxDepth; ++depth)
    {
        pvMovesSoFar.clear();

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
                pvIndex,
                searchContext,
                pvIndex > 0u ? pvMovesSoFar.data() : nullptr,
                pvIndex > 0u ? (uint32_t)pvMovesSoFar.size() : 0u,
                prevPvLine.score,
                threadID,
            };

            const PvLine pvLine = AspirationWindowSearch(thread, aspirationWindowSearchParam, outResult);
            ASSERT(pvLine.score > -CheckmateValue && pvLine.score < CheckmateValue);
            ASSERT(!pvLine.moves.empty());

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
        }

        if (finishSearchAtDepth)
        {
            ASSERT(outResult[0].moves.size() > 0);
            break;
        }

        // rememeber PV lines so they can be used in next iteration
        thread.prevPvLines = outResult;

        // check soft time limit every depth iteration
        if (!param.isPonder && param.limits.maxTimeSoft < UINT32_MAX && param.GetElapsedTime() >= param.limits.maxTimeSoft)
        {
            break;
        }
    }
}

PvLine Search::AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param, SearchResult& outResult) const
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
        rootNode.position = &param.position;
        rootNode.isPvNode = true;
        rootNode.depth = param.depth;
        rootNode.height = 0;
        rootNode.pvIndex = (uint8_t)param.pvIndex;
        rootNode.alpha = ScoreType(alpha);
        rootNode.beta = ScoreType(beta);
        rootNode.rootMoves = param.searchParam.rootMoves.data();
        rootNode.rootMovesCount = (uint32_t)param.searchParam.rootMoves.size();
        rootNode.moveFilter = param.moveFilter;
        rootNode.moveFilterCount = param.moveFilterCount;

        pvLine.score = NegaMax(thread, rootNode, param.searchContext);
        ASSERT(pvLine.score >= -CheckmateValue && pvLine.score <= CheckmateValue);

        pvLine.moves = GetPvLine(thread, param.position, param.searchParam.transpositionTable, param.depth);

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
        if (isMainThread && !stopSearch && !pvLine.moves.empty())
        {
            outResult[param.pvIndex] = pvLine;

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
    if (!node.isPvNode || prevPvLines.empty())
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

const Move Search::FindTTMove(const PackedMove& ttMove, MoveList& moves) const
{
    if (ttMove.IsValid())
    {
        for (uint32_t i = 0; i < moves.numMoves; ++i)
        {
            if (moves[i].move == ttMove)
            {
                moves[i].score = MoveOrderer::TTMoveValue;
                return moves[i].move;
            }
        }
    }

    return Move::Invalid();
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

    // TODO optimize this
    // no need to lookup game history if there's zeroing move in search tree

    uint32_t count = game.GetRepetitionCount(*node.position);

    if (count >= 2)
    {
        return true;
    }

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
            ASSERT(prevNode->position);
            ASSERT(prevNode->position->GetSideToMove() == node.position->GetSideToMove());

            if (prevNode->position->GetHash() == node.position->GetHash())
            {
                if (*prevNode->position == *node.position)
                {
                    if (++count >= 2)
                    {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

bool Search::IsDraw(const NodeInfo& node, const Game& game) const
{
    if (node.position->GetHalfMoveCount() >= 100)
    {
        return true;
    }

    if (CheckInsufficientMaterial(*node.position))
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

    const Position& position = *node.position;

    const bool isRootNode = node.height == 0; // root node is the first node in the chain (best move)
    const bool isPvNode = node.isPvNode;

    ScoreType staticEval = InvalidValue;

    // transposition table lookup
    PackedMove ttMove;
    {
        TTEntry ttEntry;
        if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
        {
            ttMove = ttEntry.move;
            staticEval = ttEntry.staticEval;

            if (ttEntry.depth >= node.depth && !isRootNode && !isPvNode)
            {
#ifdef COLLECT_SEARCH_STATS
                ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

                ScoreType ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
                ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

                if ((ttEntry.bounds == TTEntry::Bounds::Exact) ||
                    (ttEntry.bounds == TTEntry::Bounds::LowerBound && ttScore >= node.beta) ||
                    (ttEntry.bounds == TTEntry::Bounds::UpperBound && ttScore <= node.alpha))
                {
                    return ttScore;
                }
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

        bestValue = staticEval;

        if (bestValue >= beta)
        {
            ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, 0, TTEntry::Bounds::LowerBound);
            return bestValue;
        }

        if (isPvNode && bestValue > alpha)
        {
            alpha = bestValue;
        }

        // Delta Pruning - skip everything if there's no possible move that will beat alpha
        //const int32_t delta = QSearchDeltaMargin + position.BestPossibleMoveValue();
        //if (staticEval + delta < alpha)
        //{
        //    return alpha;
        //}
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.pvIndex = node.pvIndex;
    childNodeParam.isPvNode = node.isPvNode;
    childNodeParam.depth = 0;
    childNodeParam.height = node.height + 1;

    uint32_t moveGenFlags = 0;
    if (!isInCheck)
    {
        moveGenFlags |= MOVE_GEN_ONLY_TACTICAL;
    }

    MoveList moves;
    position.GenerateMoveList(moves, moveGenFlags);

    const int32_t sseTreshold = std::max(0, alpha - staticEval) - QSearchSeeMargin;

    // resolve move scoring
    // the idea here is to defer scoring if we have a TT/PV move
    // most likely we'll get beta cutoff on it so we won't need to score any other move
    uint32_t numScoredMoves = moves.numMoves;
    if (moves.numMoves > 1u)
    {
        numScoredMoves = 0;

        if (FindTTMove(ttMove, moves).IsValid())
        {
            numScoredMoves++;
        }
    }

    Move bestMove = Move::Invalid();
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

        // Delta Pruning - skip losing captures
        //if (!isInCheck && move.IsCapture() && bestValue > -CheckmateValue)
        //{
        //    if (!position.StaticExchangeEvaluation(move, sseTreshold))
        //    {
        //        continue;
        //    }
        //}

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        ctx.searchParam.transpositionTable.Prefetch(childPosition);

        moveIndex++;

        childNodeParam.position = &childPosition;
        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;
        ScoreType score = -QuiescenceNegaMax(thread, childNodeParam, ctx);

        if (score > bestValue) // new best move found
        {
            bestValue = score;
            bestMove = move;

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
            bestValue >= beta ? TTEntry::Bounds::LowerBound :
            (isPvNode && bestValue > oldAlpha) ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::UpperBound;

        ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, 0, bounds, bestMove);

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
    const bool hasMoveFilter = node.moveFilter && node.moveFilterCount > 0u;

    // Check for draw
    // Skip root node as we need some move to be reported
    if (!isRootNode && IsDraw(node, ctx.game))
    {
        return 0;
    }

    const Position& position = *node.position;

    // maximum search depth reached, enter quisence search to find final evaluation
    if (node.depth <= 0)
    {
        if (ctx.searchParam.limits.mateSearch)
        {
            const ScoreType evalScore = Evaluate(position);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);
            return ColorMultiplier(position.GetSideToMove()) * evalScore;
        }
        return QuiescenceNegaMax(thread, node, ctx);
    }
    
    const bool isInCheck = position.IsInCheck(position.GetSideToMove());

    const ScoreType oldAlpha = node.alpha;
    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;
    ScoreType bestValue = -InfValue;
    ScoreType maxValue = CheckmateValue; // max score limited by tablebase
    ScoreType staticEval = InvalidValue;

    // transposition table lookup
    PackedMove ttMove;
    TTEntry ttEntry;
    ScoreType ttScore = InvalidValue;
    if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
    {
        ttMove = ttEntry.move;
        staticEval = ttEntry.staticEval;

        // don't early exit in root node, because we may have better quality score (higher depth) discovered in one of the child nodes
        if (!isRootNode && ttEntry.depth >= node.depth)
        {
#ifdef COLLECT_SEARCH_STATS
            ctx.stats.ttHits++;
#endif // COLLECT_SEARCH_STATS

            ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
            ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

            if (ttEntry.bounds == TTEntry::Bounds::Exact)
            {
                return ttScore;
            }
            else if (ttEntry.bounds == TTEntry::Bounds::UpperBound)
            {
                if (ttScore <= alpha) return alpha;
                if (ttScore < beta) beta = ttScore;
            }
            else if (ttEntry.bounds == TTEntry::Bounds::LowerBound)
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
                    probeResult == TB_LOSS ? TTEntry::Bounds::UpperBound :
                    probeResult == TB_WIN  ? TTEntry::Bounds::LowerBound :
                    TTEntry::Bounds::Exact;

                if (    bounds == TTEntry::Bounds::Exact
                    || (bounds == TTEntry::Bounds::LowerBound && tbValue >= beta)
                    || (bounds == TTEntry::Bounds::UpperBound && tbValue <= alpha))
                {
                    ctx.searchParam.transpositionTable.Write(
                        position,
                        ScoreToTT(tbValue, node.height), staticEval,
                        bounds == TTEntry::Bounds::Exact ? UINT8_MAX : (uint8_t)node.depth,
                        bounds);

#ifdef COLLECT_SEARCH_STATS
                    ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS

                    return tbValue;
                }

                if (isPvNode)
                {
                    if (bounds == TTEntry::Bounds::LowerBound)
                    {
                        bestValue = tbValue;
                        alpha = std::max(alpha, tbValue);
                    }
                    else
                    {
                        maxValue = tbValue;
                    }
                }
            }
        }
    }
#endif // USE_TABLE_BASES

    // Null Move Prunning
    if (!isPvNode &&
        !isInCheck &&
        !ctx.searchParam.limits.mateSearch &&
        node.depth >= NullMovePrunningStartDepth &&
        ttScore >= beta &&
        !ttMove.IsValid() &&
        !position.IsPawnsOnly())
    {
        // don't allow null move if parent or grandparent node was null move
        bool doNullMove = !node.isNullMove;
        if (node.parentNode && node.parentNode->isNullMove)
        {
            doNullMove = false;
        }

        if (doNullMove)
        {
            Position childPosition = position;
            childPosition.DoNullMove();

            NodeInfo childNodeParam;
            childNodeParam.parentNode = &node;
            childNodeParam.pvIndex = node.pvIndex;
            childNodeParam.position = &childPosition;
            childNodeParam.alpha = -beta;
            childNodeParam.beta = -beta + 1;
            childNodeParam.isNullMove = true;
            childNodeParam.height = node.height + 1;
            childNodeParam.depth = node.depth - NullMovePrunningDepthReduction;

            const int32_t nullMoveScore = -NegaMax(thread, childNodeParam, ctx);

            if (nullMoveScore >= beta)
            {
                return beta;
            }
        }
    }

    // Futility Pruning
    if (!isPvNode &&
        !isInCheck &&
        !ctx.searchParam.limits.mateSearch &&
        alpha < 1000)
    {
        if (staticEval == InvalidValue)
        {
            const ScoreType evalScore = Evaluate(position);
            ASSERT(evalScore < TablebaseWinValue && evalScore > -TablebaseWinValue);
            staticEval = ColorMultiplier(position.GetSideToMove()) * evalScore;
        }

        const int32_t alphaMargin = position.BestPossibleMoveValue() + AlphaMarginBias + AlphaMarginMultiplier * node.depth;
        const int32_t betaMargin = BetaMarginBias + BetaMarginMultiplier * node.depth;

        // Alpha Pruning
        if (node.depth <= AlphaPruningDepth && (staticEval + alphaMargin <= alpha))
        {
            return (ScoreType)std::min<int32_t>(TablebaseWinValue, staticEval);
        }

        // Beta Pruning
        if (node.depth <= BetaPruningDepth && (staticEval - betaMargin >= beta))
        {
            return (ScoreType)std::max<int32_t>(-TablebaseWinValue, staticEval);
        }
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.height = node.height + 1;
    childNodeParam.pvIndex = node.pvIndex;

    uint16_t extension = 0;

    // check extension
    if (isInCheck)
    {
        extension++;
    }

    MoveList moves;
    position.GenerateMoveList(moves);

    if (isRootNode)
    {
        // apply node filter (used for multi-PV search for 2nd, 3rd, etc. moves)
        if (hasMoveFilter)
        {
            for (uint32_t i = 0; i < node.moveFilterCount; ++i)
            {
                const Move& move = node.moveFilter[i];
                moves.RemoveMove(move);
            }
        }

        // TODO
        // apply node filter (used for "searchmoves" UCI command)
        //if (!node.rootMoves.empty())
        //{
            //for (const Move& move : node.rootMoves)
            //{
            //    if (!moves.HasMove(move))
            //    {
            //        moves.RemoveMove(move);
            //    }
            //}
        //}
    }

    // resolve move scoring
    // the idea here is to defer scoring if we have a TT/PV move
    // most likely we'll get beta cutoff on it so we won't need to score any other move
    uint32_t numScoredMoves = moves.numMoves;
    if (moves.numMoves > 1u)
    {
        const Move pvMove = thread.FindPvMove(node, moves);
        const Move fullTTMove = FindTTMove(ttMove, moves);

        numScoredMoves = 0;

        if (pvMove.IsValid()) numScoredMoves++;
        if (fullTTMove.IsValid()) numScoredMoves++;
    }

    if (isRootNode)
    {
        if (ctx.searchParam.printMoves)
        {
            moves.Print();
        }
    }

    /*
    if (ctx.searchParam.verboseStats)
    {
        thread.moveOrderer.ScoreMoves(node, moves);
        numScoredMoves = moves.numMoves;

        std::cout << position.ToFEN() << std::endl;
        moves.Print();
    }
    */

    Move bestMove = Move::Invalid();
    uint32_t moveIndex = 0;
    uint32_t numReducedMoves = 0;
    bool searchAborted = false;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        if (i == numScoredMoves)
        {
            // we reached a point where moves are not scored anymore, so score them now
            thread.moveOrderer.ScoreMoves(node, moves);
        }

        int32_t moveScore = 0;
        const Move move = moves.PickBestMove(i, moveScore);
        ASSERT(move.IsValid());

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        ctx.searchParam.transpositionTable.Prefetch(childPosition);

        moveIndex++;

        int32_t moveExtension = extension;

        // promotion extension
        if (move.GetPromoteTo() != Piece::None)
        {
            moveExtension++;
        }

        // extend if there's only one legal move
        if (moveIndex == 1 && i + 1 == moves.Size())
        {
            moveExtension++;
        }

        childNodeParam.position = &childPosition;
        childNodeParam.depth = node.depth + moveExtension - 1;
        childNodeParam.previousMove = move;

        int32_t depthReduction = 0;

        // Late Move Reduction
        // don't reduce PV moves, while in check, good captures, promotions, etc.
        if (moveIndex > 2 &&
            !isRootNode &&
            node.depth >= LateMoveReductionStartDepth &&
            bestValue > -CheckmateValue &&
            !isInCheck &&
            !ctx.searchParam.limits.mateSearch &&
            move.IsQuiet())
        {
            // reduce depth gradually
            // TODO cache this
            depthReduction = int32_t(0.5f + 0.25f * sqrtf(float(node.depth - LateMoveReductionStartDepth + 1)) + 0.25f * sqrtf(float(numReducedMoves)));

            depthReduction += node.isPvNode ? 0 : 1;

            // reduce good moves less
            if (moveScore >= MoveOrderer::GoodCaptureValue) depthReduction--;

            // don't drop into QS
            depthReduction = std::max(depthReduction, 1);
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
            bestValue = score;
            bestMove = move;
            
            if (score > alpha) // update lower bound
            {
                if (isPvNode)
                {
                    thread.UpdatePvArray(node.height, move);
                }

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
        thread.moveOrderer.OnBetaCutoff(node, bestMove);
    }

    const bool canWriteTT = !(isRootNode && node.pvIndex > 0) && !CheckStopCondition(ctx);

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
    // don't write anything new to TT if time is exceeded as evaluation may be inaccurate
    // skip root nodes when searching secondary PV lines, as they don't contain best moves
    if (canWriteTT)
    {
        const TTEntry::Bounds bounds =
            bestValue >= beta ? TTEntry::Bounds::LowerBound :
            bestValue > oldAlpha ? TTEntry::Bounds::Exact :
            TTEntry::Bounds::UpperBound;

        if (!isPvNode)
        {
            ASSERT(bounds != TTEntry::Bounds::Exact);
        }

        ctx.searchParam.transpositionTable.Write(position, ScoreToTT(bestValue, node.height), staticEval, (uint8_t)node.depth, bounds, bestMove);

#ifdef COLLECT_SEARCH_STATS
        ctx.stats.ttWrites++;
#endif // COLLECT_SEARCH_STATS
    }

    return bestValue;
}
