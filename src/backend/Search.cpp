#include "Search.hpp"
#include "Game.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"
#include "TranspositionTable.hpp"

#ifdef USE_TABLE_BASES
    #include "tablebase/tbprobe.h"
#endif // USE_TABLE_BASES

#include <iostream>
#include <cstring>
#include <string>
#include <thread>
#include <math.h>

static const bool UseTranspositionTableInQSearch = true;
static const bool UsePVS = true;

static const int32_t NullMovePrunningStartDepth = 3;
static const int32_t NullMovePrunningDepthReduction = 3;

static const bool UseLateMoveReduction = true;
static const int32_t LateMoveReductionStartDepth = 2;
static const int32_t LateMoveReductionRate = 8;

static const int32_t LateMovePrunningStartDepth = 3;

static const int32_t AspirationWindowSearchStartDepth = 2;
static const int32_t AspirationWindowMax = 60;
static const int32_t AspirationWindowMin = 20;
static const int32_t AspirationWindowStep = 5;

static const int32_t BetaPruningDepth = 6;
static const int32_t BetaMarginMultiplier = 80;
static const int32_t BetaMarginBias = 30;

static const int32_t AlphaPruningDepth = 4;
static const int32_t AlphaMarginMultiplier = 150;
static const int32_t AlphaMarginBias = 1000;

// value_to_tt() adjusts a mate or TB score from "plies to mate from the root" to
// "plies to mate from the current position". Standard scores are unchanged.
// The function is called before storing a value in the transposition table.

// convert from score that is relative to root to an TT score (absolute, position dependent)
static ScoreType ScoreToTT(ScoreType v, int32_t height)
{
    ASSERT(v >= -CheckmateValue && v <= CheckmateValue);
    ASSERT(height < MaxSearchDepth);

    return ScoreType(
        v >= ( TablebaseWinValue - MaxSearchDepth) ? v + height :
        v <= (-TablebaseWinValue + MaxSearchDepth) ? v - height :
        v);
}


// convert TT score (absolute, position dependent) to search node score (relative to root)
ScoreType ScoreFromTT(ScoreType v, int32_t height, int32_t fiftyMoveRuleCount)
{
    ASSERT(height < MaxSearchDepth);

    // based on Stockfish

    if (v >= TablebaseWinValue - MaxSearchDepth)  // TB win or better
    {
        if ((v >= TablebaseWinValue - MaxSearchDepth) && (CheckmateValue - v > 99 - fiftyMoveRuleCount))
        {
            // do not return a potentially false mate score
            return CheckmateValue - MaxSearchDepth - 1;
        }
        return ScoreType(v - height);
    }

    if (v <= -TablebaseWinValue + MaxSearchDepth) // TB loss or worse
    {
        if ((v <= TablebaseWinValue - MaxSearchDepth) && (CheckmateValue + v > 99 - fiftyMoveRuleCount))
        {
            // do not return a potentially false mate score
            return CheckmateValue - MaxSearchDepth + 1;
        }
        return ScoreType(v + height);
    }

    return v;
}

#ifdef USE_TABLE_BASES
static Piece TranslatePieceType(uint32_t tbPromotes)
{
    switch (tbPromotes)
    {
    case TB_PROMOTES_QUEEN:     return Piece::Queen;
    case TB_PROMOTES_ROOK:      return Piece::Rook;
    case TB_PROMOTES_BISHOP:    return Piece::Bishop;
    case TB_PROMOTES_KNIGHT:    return Piece::Knight;
    }
    return Piece::None;
}
#endif // USE_TABLE_BASES

int64_t SearchParam::GetElapsedTime() const
{
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint).count();
}

Search::Search()
{
}

Search::~Search()
{
}

void Search::StopSearch()
{
    mStopSearch = true;
}

bool Search::CheckStopCondition(const SearchContext& ctx) const
{
    if (mStopSearch)
    {
        return true;
    }

    if (!ctx.searchParam.isPonder)
    {
        if (ctx.searchParam.limits.maxNodes < UINT64_MAX && ctx.stats.nodes >= ctx.searchParam.limits.maxNodes)
        {
            // nodes limit exceeded
            return true;
        }

        if (ctx.searchParam.limits.maxTime < UINT32_MAX && ctx.searchParam.GetElapsedTime() >= ctx.searchParam.limits.maxTime)
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

    if (param.limits.maxDepth == 0)
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

    if (param.limits.maxTime < UINT32_MAX && numLegalMoves == 1)
    {
        // if we have time limit and there's only a single legal move, return it immediately without evaluation
        outResult.front().moves.push_back(legalMoves.front());
        outResult.front().score = 0;
        return;
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

void Search::Search_Internal(const uint32_t threadID, const uint32_t numPvLines, const Game& game, const SearchParam& param, SearchResult& outResult)
{
    const bool isMainThread = threadID == 0;
    ThreadData& thread = mThreadData[threadID];

    std::vector<Move> pvMovesSoFar;

    SearchResult tempResult;
    tempResult.resize(numPvLines);

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

            const auto startTime = std::chrono::high_resolution_clock::now();

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
                prevPvLine.score
            };

            const ScoreType score = AspirationWindowSearch(thread, aspirationWindowSearchParam);
            ASSERT(score > -CheckmateValue && score < CheckmateValue);

            std::vector<Move> pvLine = GetPvLine(thread, game.GetPosition(), param.transpositionTable, depth);
            ASSERT(!pvLine.empty());

            // store for multi-PV filtering in next iteration
            for (const Move prevMove : pvMovesSoFar)
            {
                ASSERT(prevMove != pvLine.front());
            }
            pvMovesSoFar.push_back(pvLine.front());

            // write PV line into result struct
            tempResult[pvIndex].score = score;
            tempResult[pvIndex].moves = pvLine;

            const auto endTime = std::chrono::high_resolution_clock::now();

            // stop search only at depth 2 and more
            if (depth > 1 && CheckStopCondition(searchContext))
            {
                finishSearchAtDepth = true;
            }
            else if (param.debugLog && isMainThread)
            {
                std::cout << "info";
                std::cout << " depth " << (uint32_t)depth;
                std::cout << " seldepth " << (uint32_t)searchContext.stats.maxDepth;
                if (param.numPvLines > 1)
                {
                    std::cout << " multipv " << (pvIndex + 1);
                }
                std::cout << " time " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

                if (score > CheckmateValue - MaxSearchDepth)
                {
                    std::cout << " score mate " << (CheckmateValue - score + 1) / 2;
                }
                else if (score < -CheckmateValue + MaxSearchDepth)
                {
                    std::cout << " score mate -" << (CheckmateValue + score + 1) / 2;
                }
                else
                {
                    std::cout << " score cp " << score;
                }

                std::cout << " nodes " << searchContext.stats.nodes;
                //std::cout << " qnodes " << searchContext.stats.quiescenceNodes;
                //std::cout << " tthit " << searchContext.stats.ttHits;
                //std::cout << " ttwrite " << searchContext.stats.ttWrites;

                if (searchContext.stats.tbHits)
                {
                    std::cout << " tbhit " << searchContext.stats.tbHits;
                }

                std::cout << " pv ";
                {
                    for (size_t i = 0; i < pvLine.size(); ++i)
                    {
                        const Move move = pvLine[i];
                        ASSERT(move.IsValid());
                        std::cout << move.ToString();
                        if (i + 1 < pvLine.size()) std::cout << ' ';
                    }
                }

                std::cout << std::endl;

                if (param.verboseStats)
                {
                    std::cout << "Beta cutoff histogram\n";
                    uint32_t maxMoveIndex = 0;
                    uint64_t sum = 0;
                    for (uint32_t i = 0; i < MoveList::MaxMoves; ++i)
                    {
                        if (searchContext.stats.betaCutoffHistogram[i])
                        {
                            sum += searchContext.stats.betaCutoffHistogram[i];
                            maxMoveIndex = std::max(maxMoveIndex, i);
                        }
                    }
                    for (uint32_t i = 0; i <= maxMoveIndex; ++i)
                    {
                        const uint64_t value = searchContext.stats.betaCutoffHistogram[i];
                        printf("    %u : %" PRIu64 " (%.2f%%)\n", i, value, 100.0f * float(value) / float(sum));
                    }
                }
            }
        }

        if (finishSearchAtDepth)
        {
            break;
        }

        // rememeber PV lines so they can be used in next iteration
        thread.prevPvLines = tempResult;

        // check soft time limit every depth iteration
        if (!param.isPonder && param.limits.maxTimeSoft < UINT32_MAX && param.GetElapsedTime() >= param.limits.maxTimeSoft)
        {
            break;
        }
    }

    if (isMainThread)
    {
        outResult = tempResult;
    }
}

ScoreType Search::AspirationWindowSearch(ThreadData& thread, const AspirationWindowSearchParam& param)
{
    int32_t alpha = -InfValue;
    int32_t beta = InfValue;

    // decrease aspiration window with increasing depth
    int32_t aspirationWindow = AspirationWindowMax - (param.depth - AspirationWindowSearchStartDepth) * AspirationWindowStep;
    aspirationWindow = std::max<int32_t>(AspirationWindowMin, aspirationWindow);
    ASSERT(aspirationWindow > 0);

    // start applying aspiration window at given depth
    if (param.depth >= AspirationWindowSearchStartDepth && !CheckStopCondition(param.searchContext))
    {
        alpha = std::max<int32_t>(param.previousScore - aspirationWindow, -InfValue);
        beta = std::min<int32_t>(param.previousScore + aspirationWindow, InfValue);
    }

    for (;;)
    {
        // std::cout << "aspiration window: " << alpha << "..." << beta << "\n";

        memset(thread.pvArray, 0, sizeof(ThreadData::pvArray));
        memset(thread.pvLengths, 0, sizeof(ThreadData::pvLengths));

        NodeInfo rootNode;
        rootNode.position = &param.position;
        rootNode.isPvNode = true;
        rootNode.isTbNode = true; // traverse endgame table for initial node
        rootNode.depth = param.depth;
        rootNode.height = 0;
        rootNode.pvIndex = (uint8_t)param.pvIndex;
        rootNode.alpha = ScoreType(alpha);
        rootNode.beta = ScoreType(beta);
        rootNode.color = param.position.GetSideToMove();
        rootNode.rootMoves = param.searchParam.rootMoves.data();
        rootNode.rootMovesCount = (uint32_t)param.searchParam.rootMoves.size();
        rootNode.moveFilter = param.moveFilter;
        rootNode.moveFilterCount = param.moveFilterCount;

        ScoreType score = NegaMax(thread, rootNode, param.searchContext);

        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        // out of aspiration window, redo the search in wider score range
        if (score <= alpha)
        {
            //beta = alpha + 1;
            alpha -= aspirationWindow;
            alpha = std::max<int32_t>(alpha, -CheckmateValue);
            aspirationWindow *= 4;
            continue;
        }
        if (score >= beta)
        {
            //alpha = beta - 1;
            beta += aspirationWindow;
            beta = std::min<int32_t>(beta, CheckmateValue);
            aspirationWindow *= 4;
            continue;
        }

        return score;
    }
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

    for (uint32_t i = 1; ; ++i)
    {
        prevNode = prevNode->parentNode;

        // reached end of the stack
        if (!prevNode)
        {
            break;
        }

        // only check every second previous node, because side to move must be the same
        if (i % 2 == 0)
        {
            ASSERT(prevNode->position);
            if (prevNode->position->GetHash() == node.position->GetHash())
            {
                if (*prevNode->position == *node.position)
                {
                    return true;
                }
            }
        }

        // don't need to check more moves if reached pawn push or capture,
        // because these moves are irreversible
        if (prevNode->previousMove.IsValid())
        {
            if (prevNode->previousMove.GetPiece() == Piece::Pawn || prevNode->previousMove.IsCapture())
            {
                break;
            }
        }
    }

    return game.GetRepetitionCount(*node.position) > 0;
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

ScoreType Search::QuiescenceNegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx)
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
    if (UseTranspositionTableInQSearch)
    {
        TTEntry ttEntry;
        if (ctx.searchParam.transpositionTable.Read(position, ttEntry))
        {
            ttMove = ttEntry.move;
            staticEval = ttEntry.staticEval;

            if (ttEntry.depth >= node.depth && !isRootNode)
            {
                ctx.stats.ttHits++;

                ScoreType ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
                ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

                if ((ttEntry.flag == TTEntry::Flag_Exact) ||
                    (ttEntry.flag == TTEntry::Flag_LowerBound && ttScore >= node.beta) ||
                    (ttEntry.flag == TTEntry::Flag_UpperBound && ttScore <= node.alpha))
                {
                    return ttScore;
                }
            }
        }
    }

    const bool isInCheck = position.IsInCheck(node.color);

    ScoreType alpha = node.alpha;
    ScoreType oldAlpha = alpha;
    ScoreType beta = node.beta;
    ScoreType bestValue = -InfValue;

    // do not consider stand pat if in check
    if (!isInCheck)
    {
        if (staticEval == InvalidValue)
        {
            staticEval = ColorMultiplier(node.color) * Evaluate(position);
        }

        alpha = std::max(staticEval, node.alpha);
        bestValue = staticEval;

        if (alpha >= beta)
        {
            return staticEval;
        }
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.isPvNode = node.isPvNode;
    childNodeParam.depth = node.depth - 1;
    childNodeParam.height = node.height + 1;
    childNodeParam.color = GetOppositeColor(node.color);

    uint32_t moveGenFlags = 0;
    if (!isInCheck)
    {
        moveGenFlags |= MOVE_GEN_ONLY_TACTICAL;
    }

    MoveList moves;
    position.GenerateMoveList(moves, moveGenFlags);

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

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        if (UseTranspositionTableInQSearch)
        {
            ctx.searchParam.transpositionTable.Prefetch(childPosition);
        }

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
    if (UseTranspositionTableInQSearch && !CheckStopCondition(ctx))
    {
        TTEntry entry;
        entry.score = ScoreToTT(bestValue, node.height);
        entry.staticEval = staticEval;
        entry.move = bestMove;
        entry.depth = 0;
        entry.flag =
            bestValue >= beta ? TTEntry::Flag_LowerBound :
            bestValue > oldAlpha ? TTEntry::Flag_Exact :
            TTEntry::Flag_UpperBound;

        ctx.searchParam.transpositionTable.Write(position, entry);

        ctx.stats.ttWrites++;
    }

    return bestValue;
}


ScoreType Search::PruneByMateDistance(const NodeInfo& node, ScoreType alpha, ScoreType beta)
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

INLINE static bool HasTablebases()
{
#ifdef USE_TABLE_BASES
    return TB_LARGEST > 0u;
#else
    return false;
#endif
}

ScoreType Search::NegaMax(ThreadData& thread, NodeInfo& node, SearchContext& ctx)
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

    // maximum search depth reached, enter quisence search to find final evaluation
    if (node.depth <= 0)
    {
        return QuiescenceNegaMax(thread, node, ctx);
    }

    const Position& position = *node.position;
    const bool isInCheck = position.IsInCheck(node.color);

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

        if (!isRootNode && !node.isTbNode && ttEntry.depth >= node.depth)
        {
            ctx.stats.ttHits++;

            ttScore = ScoreFromTT(ttEntry.score, node.height, position.GetHalfMoveCount());
            ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

            if (ttEntry.flag == TTEntry::Flag_Exact)
            {
                return ttScore;
            }
            else if (ttEntry.flag == TTEntry::Flag_UpperBound)
            {
                if (ttScore <= alpha) return alpha;
                if (ttScore < beta) beta = ttScore;
            }
            else if (ttEntry.flag == TTEntry::Flag_LowerBound)
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
        const uint32_t pieceCount = (position.Whites().Occupied() | position.Blacks().Occupied()).Count();

        if (pieceCount <= TB_LARGEST)
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
                ctx.stats.tbHits++;

                // convert the WDL value to a score
                const ScoreType tbValue =
                    probeResult == TB_LOSS ? -(TablebaseWinValue - (ScoreType)node.height) :
                    probeResult == TB_WIN  ?  (TablebaseWinValue - (ScoreType)node.height) : 0;

                // only draws are exact, we don't know exact value for win/loss just based on WDL value
                const TTEntry::Flags bounds =
                    probeResult == TB_LOSS ? TTEntry::Flag_UpperBound :
                    probeResult == TB_WIN  ? TTEntry::Flag_LowerBound :
                    TTEntry::Flag_Exact;

                if (    bounds == TTEntry::Flag_Exact
                    || (bounds == TTEntry::Flag_LowerBound && tbValue >= beta)
                    || (bounds == TTEntry::Flag_UpperBound && tbValue <= alpha))
                {
                    TTEntry entry;
                    entry.score = ScoreToTT(tbValue, node.height);
                    entry.depth = bounds == TTEntry::Flag_Exact ? UINT8_MAX : (uint8_t)node.depth;
                    entry.flag = bounds;

                    ctx.searchParam.transpositionTable.Write(position, entry);

                    ctx.stats.ttWrites++;

                    return tbValue;
                }

                if (isPvNode)
                {
                    if (bounds == TTEntry::Flag_LowerBound)
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
    if (!isPvNode && !node.isTbNode && !isInCheck && node.depth >= NullMovePrunningStartDepth && ttScore >= beta && !ttMove.IsValid())
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
            childNodeParam.color = GetOppositeColor(node.color);
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
    if (!isPvNode && !node.isTbNode && !isInCheck)
    {
        if (staticEval == InvalidValue)
        {
            staticEval = ColorMultiplier(node.color) * Evaluate(position);
        }

        const int32_t alphaMargin = AlphaMarginBias + AlphaMarginMultiplier * node.depth;
        const int32_t betaMargin = BetaMarginBias + BetaMarginMultiplier * node.depth;

        // Alpha Pruning
        if (node.depth <= AlphaPruningDepth && (staticEval + alphaMargin <= alpha))
        {
            return (ScoreType)std::min<int32_t>(TablebaseWinValue, staticEval + alphaMargin);
        }

        // Beta Pruning
        if (node.depth <= BetaPruningDepth && (staticEval - betaMargin >= beta))
        {
            return (ScoreType)std::max<int32_t>(-TablebaseWinValue, staticEval - betaMargin);
        }
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.height = node.height + 1;
    childNodeParam.color = GetOppositeColor(node.color);
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

    Move tbMove = Move::Invalid();
#ifdef USE_TABLE_BASES
    /*
    if ((isPvNode || node.isTbNode) && !hasMoveFilter && HasTablebases())
    {
        const uint32_t probeResult = tb_probe_root(
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
            position.GetSideToMove() == Color::White,
            nullptr);

        if (probeResult != TB_RESULT_FAILED)
        {
            // find move that matches tablebase probe result
            for (uint32_t i = 0; i < moves.Size(); ++i)
            {
                MoveList::MoveEntry& moveEntry = moves[i];

                if (moveEntry.move.FromSquare() == TB_GET_FROM(probeResult) &&
                    moveEntry.move.ToSquare() == TB_GET_TO(probeResult) &&
                    moveEntry.move.GetPromoteTo() == TranslatePieceType(TB_GET_PROMOTES(probeResult)))
                {
                    tbMove = moveEntry.move;
                    break;
                }
            }

            if (tbMove.IsValid())
            {
                moves.Clear();
                moves.Push(tbMove);
            }
        }
    }
    */
#endif // USE_TABLE_BASES

    Move bestMove = Move::Invalid();
    uint32_t moveIndex = 0;
    int32_t numReducedMoves = 0;

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

        // perform TB walk for child node if this node has moves filtered, so we get full line in multi-PV mode
        const bool performTablebaseWalk = HasTablebases() && (tbMove == move || hasMoveFilter);

        // promotion extension
        if (move.GetPromoteTo() != Piece::None)
        {
            moveExtension++;
        }

        // endgame tablebase walk extension
        if (performTablebaseWalk)
        {
            moveExtension++;
        }

        // extend if there's only one legal move
        if (moveIndex == 1 && i + 1 == moves.Size())
        {
            moveExtension++;
        }

        childNodeParam.position = &childPosition;
        childNodeParam.isTbNode = performTablebaseWalk;
        childNodeParam.depth = node.depth + moveExtension - 1;
        childNodeParam.previousMove = move;

        int32_t depthReduction = 0;

        // Late Move Reduction
        if (node.depth >= LateMoveReductionStartDepth)
        {
            // don't reduce PV moves, while in check, captures, promotions, etc.
            if (move.IsQuiet() && !isInCheck && moveIndex > 1u)
            {
                // reduce depth gradually
                //depthReduction = std::min(1, std::max(1, numReducedMoves / LateMoveReductionRate));
                depthReduction = std::max(1, int(0.5f * std::sqrt((float)(node.depth + 1 - LateMoveReductionStartDepth) * (float)numReducedMoves)));
                numReducedMoves++;

                // Late Move Prunning
                if (node.depth >= LateMovePrunningStartDepth && depthReduction > childNodeParam.depth)
                {
                    continue;
                }
            }
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
                    ctx.stats.betaCutoffHistogram[moveIndex - 1]++;
                    ASSERT(score >= beta);
                    ASSERT(alpha < beta);
                    break;
                }
            }
        }

        if (!isRootNode && CheckStopCondition(ctx))
        {
            // abort search of further moves
            break;
        }
    }

    if (bestValue >= beta)
    {
        thread.moveOrderer.OnBetaCutoff(node, bestMove);
    }

    // no legal moves
    if (moveIndex == 0u)
    {
        TTEntry entry;

        if (isInCheck) // checkmate
        {
            bestValue = -CheckmateValue + (ScoreType)node.height;

            // checkmate score depends on depth and may not be exact
            entry.depth = (uint8_t)node.depth;
            entry.flag =
                bestValue >= beta ? TTEntry::Flag_LowerBound :
                bestValue > oldAlpha ? TTEntry::Flag_Exact :
                TTEntry::Flag_UpperBound;
        }
        else // stalemate
        {
            bestValue = entry.score = 0;

            // stalemate score is always exact so we can even extend TT entry depth to infinity
            entry.depth = UINT8_MAX;
            entry.flag = TTEntry::Flag_Exact;
        }
        entry.score = ScoreToTT(bestValue, node.height);

        ctx.searchParam.transpositionTable.Write(position, entry);

        ctx.stats.ttWrites++;

        return bestValue;
    }

    ASSERT(alpha < beta);
    ASSERT(bestValue >= -CheckmateValue && bestValue <= CheckmateValue);

    // limit by TB
    bestValue = std::min(bestValue, maxValue);

    // update transposition table
    // don't write anything new to TT if time is exceeded as evaluation may be inaccurate
    // skip root nodes when searching secondary PV lines, as they don't contain best moves
    if (!CheckStopCondition(ctx) && !(isRootNode && node.pvIndex > 0))
    {
        TTEntry::Flags flag = TTEntry::Flag_Exact;

        // move from tablebases is always best
        if (bestMove != tbMove)
        {
            flag =
                bestValue >= beta ? TTEntry::Flag_LowerBound :
                bestValue <= oldAlpha ? TTEntry::Flag_UpperBound :
                TTEntry::Flag_Exact;
        }

        TTEntry entry;
        entry.score = ScoreToTT(bestValue, node.height);
        entry.staticEval = staticEval;
        entry.move = bestMove;
        entry.depth = (uint8_t)node.depth;
        entry.flag = flag;

        ctx.searchParam.transpositionTable.Write(position, entry);

        ctx.stats.ttWrites++;
    }

    return bestValue;
}