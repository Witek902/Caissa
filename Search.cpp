#include "Search.hpp"
#include "Game.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"

#include "tablebase/tbprobe.h"

#include <iostream>
#include <string>

static const uint32_t NullMovePrunningStartDepth = 3;
static const int32_t NullMovePrunningDepthReduction = 3;

static const uint32_t LateMoveReductionStartDepth = 3;
static const uint32_t LateMoveReductionRate = 8;

static const uint32_t LateMovePrunningStartDepth = 3;

static const uint32_t AspirationWindowSearchStartDepth = 4;
static const int32_t AspirationWindowMax = 120;
static const int32_t AspirationWindowMin = 15;
static const int32_t AspirationWindowStep = 5;

static const uint32_t BetaPruningDepth = 6;
static const int32_t BetaMarginMultiplier = 80;
static const int32_t BetaMarginBias = 30;

static const uint32_t AlphaPruningDepth = 4;
static const int32_t AlphaMarginMultiplier = 150;
static const int32_t AlphaMarginBias = 1000;

static const Piece TranslatePieceType(uint32_t tbPromotes)
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

int64_t SearchParam::GetElapsedTime() const
{
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint).count();
}

Search::Search()
{
    mTranspositionTable.Resize(1024 * 1024);
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
    
    if (ctx.searchParam.limits.maxNodes < UINT64_MAX && ctx.nodes >= ctx.searchParam.limits.maxNodes)
    {
        // nodes limit exceeded
        return true;
    }

    if (ctx.searchParam.limits.maxTime < UINT32_MAX && ctx.searchParam.GetElapsedTime() >= ctx.searchParam.limits.maxTime)
    {
        // time limit exceeded
        return true;
    }

    return false;
}

std::vector<Move> Search::GetPvLine(const Position& pos, uint32_t maxLength) const
{
    std::vector<Move> moves;

    uint32_t pvLength = pvLengths[0];

    if (pvLength > 0)
    {
        uint32_t i;

        // reconstruct PV line using PV array
        Position iteratedPosition = pos;
        for (i = 0; i < pvLength; ++i)
        {
            const Move move = iteratedPosition.MoveFromPacked(pvArray[0][i]);

            if (!move.IsValid()) break;
            if (!iteratedPosition.DoMove(move)) break;

            moves.push_back(move);
        }

        // reconstruct PV line using transposition table
        for (; i < maxLength; ++i)
        {
            if (iteratedPosition.GetNumLegalMoves() == 0) break;

            const TranspositionTableEntry* ttEntry = mTranspositionTable.Read(iteratedPosition);
            if (!ttEntry) break;

            const Move move = iteratedPosition.MoveFromPacked(ttEntry->move);

            // Note: move in transpostion table may be invalid due to hash collision
            if (!move.IsValid()) break;
            if (!iteratedPosition.DoMove(move)) break;

            moves.push_back(move);
        }

        ASSERT(!moves.empty());
    }

    return moves;
}

void Search::DoSearch(const Game& game, const SearchParam& param, SearchResult& result)
{
    std::vector<Move> pvMovesSoFar;

    mStopSearch = false;
    mPrevPvLines.clear();

    // clamp number of PV lines (there can't be more than number of max moves)
    static_assert(MoveList::MaxMoves <= UINT8_MAX, "Max move count must fit uint8");
    std::vector<Move> legalMoves;
    const uint32_t numLegalMoves = game.GetPosition().GetNumLegalMoves(&legalMoves);
    const uint32_t numPvLines = std::min(param.numPvLines, numLegalMoves);

    result.clear();
    result.resize(numPvLines);

    if (numPvLines == 0u)
    {
        // early exit in case of no legal moves
        return;
    }

    if (param.limits.maxTime < UINT32_MAX && numLegalMoves == 1)
    {
        // if we have time limit and there's only a single legal move, return it immediately without evaluation
        result.front().moves.push_back(legalMoves.front());
        result.front().score = 0;
        return;
    }

    memset(searchHistory, 0, sizeof(searchHistory));
    memset(killerMoves, 0, sizeof(killerMoves));

    for (uint32_t depth = 1; depth <= param.limits.maxDepth; ++depth)
    {
        pvMovesSoFar.clear();

        bool finishSearchAtDepth = false;

        for (uint32_t pvIndex = 0; pvIndex < numPvLines; ++pvIndex)
        {
            PvLine& outPvLine = result[pvIndex];

            auto startTime = std::chrono::high_resolution_clock::now();

            SearchContext searchContext{ game, param };

            AspirationWindowSearchParam aspirationWindowSearchParam =
            {
                game.GetPosition(),
                param,
                result,
                depth,
                pvIndex,
                searchContext,
                pvIndex > 0u ? pvMovesSoFar : std::span<const Move>(),
                outPvLine.score
            };

            const int32_t score = AspirationWindowSearch(aspirationWindowSearchParam);
            ASSERT(score > -CheckmateValue && score < CheckmateValue);

            // write PV line into result struct
            outPvLine.score = score;
            outPvLine.moves = GetPvLine(game.GetPosition(), depth);
            ASSERT(!outPvLine.moves.empty());

            // store for multi-PV filtering in next iteration
            pvMovesSoFar.push_back(outPvLine.moves.front());

            auto endTime = std::chrono::high_resolution_clock::now();

            // stop search only at depth 2 and more
            if (depth > 1 && CheckStopCondition(searchContext))
            {
                finishSearchAtDepth = true;
            }
            else if (param.debugLog)
            {
                std::cout << "info";
                std::cout << " depth " << (uint32_t)depth;
                std::cout << " seldepth " << (uint32_t)searchContext.maxDepth;
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

                std::cout << " nodes " << searchContext.nodes;

                if (searchContext.tbHits)
                {
                    std::cout << " tbhit " << searchContext.tbHits;
                }

                std::cout << " pv ";
                {
                    for (size_t i = 0; i < outPvLine.moves.size(); ++i)
                    {
                        const Move move = outPvLine.moves[i];
                        ASSERT(move.IsValid());
                        std::cout << move.ToString();
                        if (i + 1 < outPvLine.moves.size()) std::cout << ' ';
                    }
                }

                std::cout << std::endl;
            }
        }

        if (finishSearchAtDepth)
        {
            // restore result from previous depth, the current result is not reliable
            ASSERT(!mPrevPvLines.empty());
            result = mPrevPvLines;

            break;
        }

        mPrevPvLines = result;
    }
}

int32_t Search::AspirationWindowSearch(const AspirationWindowSearchParam& param)
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
        alpha = std::max(param.previousScore - aspirationWindow, -InfValue);
        beta = std::min(param.previousScore + aspirationWindow, InfValue);
    }

    for (;;)
    {
        if (param.searchParam.printMoves)
        {
            std::cout << "aspiration window: " << alpha << "..." << beta << "\n";
        }

        memset(pvArray, 0, sizeof(pvArray));
        memset(pvLengths, 0, sizeof(pvLengths));

        NodeInfo rootNode;
        rootNode.position = &param.position;
        rootNode.depth = 0u;
        rootNode.isPvNode = true;
        rootNode.isTbNode = true; // traverse endgame table for initial node
        rootNode.maxDepthFractional = param.depth << MaxDepthShift;
        rootNode.pvIndex = (uint8_t)param.pvIndex;
        rootNode.alpha = alpha;
        rootNode.beta = beta;
        rootNode.color = param.position.GetSideToMove();
        rootNode.rootMoves = param.searchParam.rootMoves;
        rootNode.moveFilter = param.moveFilter;

        ScoreType score = NegaMax(rootNode, param.searchContext);

        ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

        // out of aspiration window, redo the search in wider score range
        if (score <= alpha || score >= beta)
        {
            alpha -= aspirationWindow;
            beta += aspirationWindow;
            aspirationWindow *= 4;
            continue;
        }

        return score;
    }
}

static INLINE int32_t ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}

const Move Search::FindPvMove(const NodeInfo& node, MoveList& moves) const
{
    if (!node.isPvNode || mPrevPvLines.empty())
    {
        return Move::Invalid();
    }

    const std::vector<Move>& pvLine = mPrevPvLines[node.pvIndex].moves;
    if (node.depth >= pvLine.size())
    {
        return Move::Invalid();
    }

    if (node.depth >= pvLine.size())
    {
        return Move::Invalid();
    }

    const Move pvMove = pvLine[node.depth];
    ASSERT(pvMove.IsValid());

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        if (pvMove.IsValid() && moves[i].move == pvMove)
        {
            moves[i].score = INT32_MAX;
            return pvMove;
        }
    }

    // no PV move found?
    //ASSERT(false);
    return pvMove;
}

void Search::FindHistoryMoves(Color color, MoveList& moves) const
{
    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        const Move move = moves[i].move;
        ASSERT(move.IsValid());

        const uint32_t score = searchHistory[(uint32_t)color][move.fromSquare.Index()][move.toSquare.Index()];
        const int64_t finalScore = (int64_t)moves[i].score + score;
        moves[i].score = (int32_t)std::min<uint64_t>(finalScore, INT32_MAX);
    }
}

void Search::FindTTMove(const PackedMove& ttMove, MoveList& moves) const
{
    if (ttMove.IsValid())
    {
        for (uint32_t i = 0; i < moves.numMoves; ++i)
        {
            if (moves[i].move == ttMove)
            {
                moves[i].score = INT32_MAX - 1;
                break;
            }
        }
    }
}

void Search::FindKillerMoves(uint32_t depth, MoveList& moves) const
{
    if (depth < MaxSearchDepth)
    {
        for (uint32_t i = 0; i < moves.numMoves; ++i)
        {
            for (uint32_t j = 0; j < NumKillerMoves; ++j)
            {
                if (moves[i].move == killerMoves[depth][j])
                {
                    moves[i].score += 100000 - j;
                }
            }
        }
    }
}

void Search::UpdatePvArray(uint32_t depth, const Move move)
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

void Search::UpdateSearchHistory(const NodeInfo& node, const Move move)
{
    if (move.isCapture)
    {
        return;
    }

    uint32_t& historyCounter = searchHistory[(uint32_t)node.color][move.fromSquare.Index()][move.toSquare.Index()];

    const uint64_t historyBonus = node.MaxDepth() - node.depth;

    const uint64_t newValue = std::min<uint64_t>(UINT32_MAX, (uint64_t)historyCounter + 1u + historyBonus * historyBonus);
    historyCounter = (uint32_t)newValue;
}

void Search::RegisterKillerMove(const NodeInfo& node, const Move move)
{
    if (move.isCapture)
    {
        return;
    }

    if (node.depth < MaxSearchDepth)
    {
        for (uint32_t j = NumKillerMoves; j-- > 1u; )
        {
            killerMoves[node.depth][j] = killerMoves[node.depth][j - 1];
        }
        killerMoves[node.depth][0] = move;
    }
}

bool Search::IsRepetition(const NodeInfo& node, const Game& game) const
{
    for(const NodeInfo* prevNode = &node;;)
    {
        // only check every second previous node, because side to move must be the same
        if (prevNode->parentNode)
        {
            prevNode = prevNode->parentNode->parentNode;
        }
        else
        {
            prevNode = nullptr;
        }

        // reached end of the stack
        if (!prevNode)
        {
            break;
        }

        ASSERT(prevNode->position);
        if (prevNode->position->GetHash() == node.position->GetHash())
        {
            if (*prevNode->position == *node.position)
            {
                return true;
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

Search::ScoreType Search::QuiescenceNegaMax(const NodeInfo& node, SearchContext& ctx)
{
    // clean PV line
    if (node.depth < MaxSearchDepth)
    {
        pvLengths[node.depth] = (uint8_t)node.depth;
    }

    // update stats
    ctx.nodes++;
    ctx.quiescenceNodes++;
    ctx.maxDepth = std::max<uint32_t>(ctx.maxDepth, node.depth);

    if (IsDraw(node, ctx.game))
    {
        return 0;
    }

    const Position& position = *node.position;

    const uint32_t inversedDepth = node.MaxDepth() - node.depth;
    const bool isRootNode = node.depth == 0; // root node is the first node in the chain (best move)
    const bool isPvNode = node.alpha != node.beta - 1;

    // transposition table lookup
    PackedMove ttMove;
    ScoreType ttScore = InvalidValue;
    if (const TranspositionTableEntry* ttEntry = mTranspositionTable.Read(position))
    {
        ttMove = ttEntry->move;

        if (ttEntry->depth >= inversedDepth && !isPvNode && !isRootNode)
        {
            ctx.ttHits++;

            ttScore = ttEntry->score;
            ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

            if ((ttEntry->flag == TranspositionTableEntry::Flag_Exact) ||
                (ttEntry->flag == TranspositionTableEntry::Flag_LowerBound && ttEntry->score >= node.beta) ||
                (ttEntry->flag == TranspositionTableEntry::Flag_UpperBound && ttEntry->score <= node.alpha))
            {
                return ttScore;
            }
        }
    }

    const ScoreType staticEval = ColorMultiplier(node.color) * Evaluate(position);

    ScoreType alpha = std::max(staticEval, node.alpha);
    ScoreType oldAlpha = alpha;
    ScoreType beta = node.beta;

    if (alpha >= node.beta)
    {
        return staticEval;
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.depth = node.depth + 1;
    childNodeParam.maxDepthFractional = 0;
    childNodeParam.color = GetOppositeColor(node.color);

    uint32_t moveGenFlags = 0;
    if (!position.IsInCheck(node.color))
    {
        moveGenFlags |= MOVE_GEN_ONLY_TACTICAL;
    }

    MoveList moves;
    position.GenerateMoveList(moves, moveGenFlags);

    if (moves.numMoves > 1u)
    {
        FindTTMove(ttMove, moves);
    }

    Move bestMove = Move::Invalid();
    int32_t bestValue = staticEval;
    uint32_t moveIndex = 0;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        int32_t moveScore = 0;
        const Move move = moves.PickBestMove(i, moveScore);

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        moveIndex++;

        childNodeParam.position = &childPosition;
        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;
        ScoreType score = -QuiescenceNegaMax(childNodeParam, ctx);

        if (score > bestValue)
        {
            bestValue = score;
            bestMove = move;

            if (score > alpha)
            {
                alpha = score;

                if (score >= beta)
                {
                    // for move ordering stats
                    ctx.fh++;
                    if (moveIndex == 1u) ctx.fhf++;
                    break;
                }
            }
        }
    }

    // store value in transposition table
    {
        TranspositionTableEntry entry;
        entry.positionHash = position.GetHash();
        entry.score = bestValue;
        entry.move = bestMove;
        entry.isQuiescent = true;
        entry.flag =
            bestValue >= beta ? TranspositionTableEntry::Flag_LowerBound :
            bestValue > oldAlpha ? TranspositionTableEntry::Flag_Exact :
            TranspositionTableEntry::Flag_UpperBound;

        mTranspositionTable.Write(entry);
    }

    return bestValue;
}


int32_t Search::PruneByMateDistance(const NodeInfo& node, int32_t alpha, int32_t beta)
{
    int32_t matingValue = CheckmateValue - node.depth;
    if (matingValue < beta)
    {
        beta = matingValue;
        if (alpha >= matingValue)
        {
            return matingValue;
        }
    }

    matingValue = -CheckmateValue + node.depth;
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
    return TB_LARGEST > 0u;
}

Search::ScoreType Search::NegaMax(const NodeInfo& node, SearchContext& ctx)
{
    ASSERT(node.alpha <= node.beta);

    // clean PV line
    if (node.depth < MaxSearchDepth)
    {
        pvLengths[node.depth] = (uint8_t)node.depth;
    }

    // update stats
    ctx.nodes++;
    ctx.maxDepth = std::max<uint32_t>(ctx.maxDepth, node.depth);

    const bool isRootNode = node.depth == 0; // root node is the first node in the chain (best move)
    const bool isPvNode = node.alpha != node.beta - 1;

    // Check for draw
    // Skip root node as we need some move to be reported
    if (!isRootNode && IsDraw(node, ctx.game))
    {
        return 0;
    }

    const Position& position = *node.position;
    const bool isInCheck = position.IsInCheck(node.color);
    const uint32_t inversedDepth = node.MaxDepth() - node.depth;

    const ScoreType oldAlpha = node.alpha;
    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;

    // transposition table lookup
    PackedMove ttMove;
    ScoreType ttScore = InvalidValue;
    if (const TranspositionTableEntry* ttEntry = mTranspositionTable.Read(position))
    {
        ttMove = ttEntry->move;

        if (!isPvNode && !isRootNode && !node.isTbNode &&
            !ttEntry->isQuiescent && ttEntry->depth >= inversedDepth)
        {
            ctx.ttHits++;

            ttScore = ttEntry->score;
            ASSERT(ttScore >= -CheckmateValue && ttScore <= CheckmateValue);

            if ((ttEntry->flag == TranspositionTableEntry::Flag_Exact) ||
                (ttEntry->flag == TranspositionTableEntry::Flag_LowerBound && ttEntry->score >= beta) ||
                (ttEntry->flag == TranspositionTableEntry::Flag_UpperBound && ttEntry->score <= alpha))
            {
                return ttScore;
            }
        }
    }

    // mate distance prunning
    if (!isRootNode)
    {
        int32_t mateDistanceScore = PruneByMateDistance(node, alpha, beta);
        if (mateDistanceScore != 0)
        {
            return mateDistanceScore;
        }
    }

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
                ctx.tbHits++;

                // convert the WDL value to a score
                int32_t tbValue =
                    probeResult == TB_LOSS ? -(TablebaseWinValue - (int32_t)node.depth) :
                    probeResult == TB_WIN  ?  (TablebaseWinValue - (int32_t)node.depth) : 0;

                // only draws are exact, we don't know exact value for win/loss just based on WDL value
                const TranspositionTableEntry::Flags bounds =
                    probeResult == TB_LOSS ? TranspositionTableEntry::Flag_UpperBound :
                    probeResult == TB_WIN  ? TranspositionTableEntry::Flag_LowerBound :
                    TranspositionTableEntry::Flag_Exact;

                if (    bounds == TranspositionTableEntry::Flag_Exact
                    || (bounds == TranspositionTableEntry::Flag_LowerBound && tbValue >= beta)
                    || (bounds == TranspositionTableEntry::Flag_UpperBound && tbValue <= alpha))
                {
                    //const TranspositionTableEntry entry{ position.GetHash(), tbValue, Move::Invalid(), (uint8_t)inversedDepth, bounds };
                    //mTranspositionTable.Write(entry);

                    return tbValue;
                }
            }
        }
    }


    // maximum search depth reached, enter quisence search to find final evaluation
    if (node.depth >= node.MaxDepth())
    {
        return QuiescenceNegaMax(node, ctx);
    }

    // Futility Pruning
    if (!isPvNode && !node.isTbNode && !isInCheck)
    {
        // determine static evaluation of the board
        int32_t staticEvaluation = ttScore;
        if (staticEvaluation == InvalidValue)
        {
            staticEvaluation = ColorMultiplier(node.color) * Evaluate(position);
        }

        const int32_t alphaMargin = AlphaMarginBias + AlphaMarginMultiplier * inversedDepth;
        const int32_t betaMargin = BetaMarginBias + BetaMarginMultiplier * inversedDepth;

        // Alpha Pruning
        if (inversedDepth <= AlphaPruningDepth && (staticEvaluation + alphaMargin <= alpha))
        {
            return staticEvaluation + alphaMargin;
        }

        // Beta Pruning
        if (inversedDepth <= BetaPruningDepth && (staticEvaluation - betaMargin >= beta))
        {
            return staticEvaluation - betaMargin;
        }
    }

    // Null Move Prunning
    if (!isPvNode && !node.isTbNode && !isInCheck && inversedDepth >= NullMovePrunningStartDepth && ttScore >= beta && !ttMove.IsValid())
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
            childNodeParam.depth = node.depth + 1;
            childNodeParam.color = GetOppositeColor(node.color);
            childNodeParam.pvIndex = node.pvIndex;
            childNodeParam.position = &childPosition;
            childNodeParam.alpha = -beta;
            childNodeParam.beta = -beta + 1;
            childNodeParam.isNullMove = true;
            childNodeParam.maxDepthFractional = std::max(0, (int32_t)node.maxDepthFractional - (NullMovePrunningDepthReduction << MaxDepthShift));

            const int32_t nullMoveScore = -NegaMax(childNodeParam, ctx);

            if (nullMoveScore >= beta)
            {
                return beta;
            }
        }
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.depth = node.depth + 1;
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
        if (!node.moveFilter.empty())
        {
            for (const Move& move : node.moveFilter)
            {
                moves.RemoveMove(move);
            }
        }

        // apply node filter (used for "searchmoves" UCI command)
        if (!node.rootMoves.empty())
        {
            // TODO
            //for (const Move& move : node.rootMoves)
            //{
            //    if (!moves.HasMove(move))
            //    {
            //        moves.RemoveMove(move);
            //    }
            //}
        }
    }

    ctx.pseudoMovesPerNode += moves.numMoves;

    const Move pvMove = FindPvMove(node, moves);

    if (moves.numMoves > 1u)
    {
        FindHistoryMoves(node.color, moves);
        FindKillerMoves(node.depth, moves);
        FindTTMove(ttMove, moves);
    }

    if (isRootNode)
    {
        if (ctx.searchParam.printMoves)
        {
            moves.Print();
        }
    }

    Move tbMove = Move::Invalid();
    if ((isPvNode || node.isTbNode) && HasTablebases())
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

                if (moveEntry.move.fromSquare == TB_GET_FROM(probeResult) &&
                    moveEntry.move.toSquare == TB_GET_TO(probeResult) &&
                    moveEntry.move.promoteTo == TranslatePieceType(TB_GET_PROMOTES(probeResult)))
                {
                    tbMove = moveEntry.move;
                    break;
                }
            }

            if (tbMove.IsValid())
            {
                moves.Clear();
                moves.PushMove(tbMove, 0);
            }
        }
    }

    Move bestMove = Move::Invalid();
    int32_t bestValue = INT32_MIN;
    uint32_t moveIndex = 0;
    uint32_t numReducedMoves = 0;

    // count (pseudo) quiet moves
    uint32_t totalQuietMoves = 0;
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves[i].move;
        if (move.IsQuiet())
        {
            totalQuietMoves++;
        }
    }

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        int32_t moveScore = 0;
        const Move move = moves.PickBestMove(i, moveScore);
        ASSERT(move.IsValid());

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        mTranspositionTable.Prefetch(childPosition);

        moveIndex++;

        int32_t moveExtensionFractional = extension << MaxDepthShift;

        // recapture extension
        if (move.isCapture && node.previousMove.isCapture && move.toSquare == node.previousMove.toSquare)
        {
            //moveExtensionFractional += 1 << (MaxDepthShift - 1);
        }

        // perform TB walk for child node if this node has moves filtered, so we get full line in multi-PV mode
        const bool performTablebaseWalk = HasTablebases() && (tbMove == move || !node.moveFilter.empty());

        // endgame tablebase walk extension
        if (performTablebaseWalk)
        {
            moveExtensionFractional += 1 << MaxDepthShift;
        }

        childNodeParam.position = &childPosition;
        childNodeParam.isPvNode = pvMove == move;
        childNodeParam.isTbNode = performTablebaseWalk;
        childNodeParam.maxDepthFractional = node.maxDepthFractional + moveExtensionFractional;
        childNodeParam.previousMove = move;

        uint32_t depthReductionFractional = 0;

        // Late Move Reduction
        // don't reduce PV moves, while in check, captures, promotions, etc.
        if (move.IsQuiet() && !isInCheck && totalQuietMoves > 0 && moveIndex > 1u && inversedDepth >= LateMoveReductionStartDepth)
        {
            // reduce depth gradually
            //depthReductionFractional = (numReducedMoves << MaxDepthShift) / LateMoveReductionRate;
            depthReductionFractional = std::max<int32_t>(1, (numReducedMoves << MaxDepthShift) / LateMoveReductionRate);
            numReducedMoves++;

            // Late Move Prunning
            if (inversedDepth >= LateMovePrunningStartDepth && depthReductionFractional > childNodeParam.maxDepthFractional)
            {
                continue;
            }
        }

        ScoreType score = InvalidValue;

        bool doFullDepthSearch = !(isPvNode && moveIndex == 1);

        // PVS search at reduced depth
        if (depthReductionFractional)
        {
            ASSERT(childNodeParam.maxDepthFractional >= depthReductionFractional);
            childNodeParam.maxDepthFractional -= depthReductionFractional;
            childNodeParam.alpha = -alpha - 1;
            childNodeParam.beta = -alpha;

            score = -NegaMax(childNodeParam, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);

            doFullDepthSearch = score > alpha;
        }

        // PVS search at full depth
        // TODO: internal aspiration window?
        if (doFullDepthSearch)
        {
            childNodeParam.maxDepthFractional = node.maxDepthFractional + moveExtensionFractional;
            childNodeParam.alpha = -alpha - 1;
            childNodeParam.beta = -alpha;

            score = -NegaMax(childNodeParam, ctx);
            ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
        }

        // full search for PV nodes
        if (isPvNode)
        {
            if (moveIndex == 1 || score > alpha)
            {
                childNodeParam.maxDepthFractional = node.maxDepthFractional + moveExtensionFractional;
                childNodeParam.alpha = -beta;
                childNodeParam.beta = -alpha;

                score = -NegaMax(childNodeParam, ctx);
                ASSERT(score >= -CheckmateValue && score <= CheckmateValue);
            }
        }

        if (score > bestValue) // new best move found
        {
            bestValue = score;
            bestMove = move;

            if (score > alpha) // update lower bound
            {
                alpha = score;

                UpdatePvArray(node.depth, move);
                UpdateSearchHistory(node, move);

                if (score >= beta) // beta cutoff
                {
                    // for move ordering stats
                    ctx.fh++;
                    if (moveIndex == 1u) ctx.fhf++;

                    RegisterKillerMove(node, move);

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

    // no legal moves
    if (moveIndex == 0u)
    {
        if (isInCheck) // checkmate
        {
            return -CheckmateValue + node.depth;
        }
        else // stalemate
        {
            return 0;
        }
    }

    ASSERT(bestValue >= -CheckmateValue && bestValue <= CheckmateValue);

    // update transposition table
    // don't write anything new to TT if time is exceeded as evaluation may be inaccurate
    // skip root nodes when searching secondary PV lines, as they don't contain best moves
    if (!CheckStopCondition(ctx) && !(isRootNode && node.pvIndex > 0))
    {
        TranspositionTableEntry::Flags flag = TranspositionTableEntry::Flag_Exact;

        // move from tablebases is always best
        if (bestMove != tbMove)
        {
            flag =
                bestValue >= beta ? TranspositionTableEntry::Flag_LowerBound :
                bestValue > oldAlpha ? TranspositionTableEntry::Flag_Exact :
                TranspositionTableEntry::Flag_UpperBound;
        }

        TranspositionTableEntry entry;
        entry.positionHash = position.GetHash();
        entry.score = bestValue;
        entry.move = bestMove;
        entry.depth = (uint8_t)inversedDepth;
        entry.flag = flag;

        mTranspositionTable.Write(entry);
    }

    return bestValue;
}
