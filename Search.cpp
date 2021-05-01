#include "Search.hpp"
#include "Move.hpp"
#include "Evaluate.hpp"

#include <iostream>
#include <string>
#include <chrono>
#include <intrin.h>

Search::Search()
{
}

void Search::RecordBoardPosition(const Position& position)
{
    GameHistoryPositionEntry& entry = historyGamePositions[position.GetHash()];

    for (GameHistoryPosition& historyPosition : entry)
    {
        if (historyPosition.pos == position)
        {
            historyPosition.count++;
            return;
        }
    }

    entry.push_back({ position, 1u });
}

bool Search::IsPositionRepeated(const Position& position, uint32_t repetitionCount) const
{
    const auto iter = historyGamePositions.find(position.GetHash());
    if (iter == historyGamePositions.end())
    {
        return false;
    }

    const GameHistoryPositionEntry& entry = iter->second;

    for (const GameHistoryPosition& historyPosition : entry)
    {
        if (historyPosition.pos == position)
        {
            return historyPosition.count >= repetitionCount;
        }
    }

    return false;
}

Search::ScoreType Search::DoSearch(const Position& position, Move& outBestMove, const SearchParam& searchParam)
{
    ScoreType score = 0;

    prevPvArrayLength = 0;

    transpositionTable.clear();
    transpositionTable.resize(searchParam.transpositionTableSize);

    int32_t aspirationWindow = 400;
    const int32_t minAspirationWindow = 40;
    const uint32_t aspirationSearchStartDepth = 20;

    int32_t alpha = -InfValue;
    int32_t beta  =  InfValue;

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t depth = 1; depth <= searchParam.maxDepth; ++depth)
    {
        memset(pvArray, 0, sizeof(pvArray));
        memset(pvLengths, 0, sizeof(pvLengths));
        memset(searchHistory, 0, sizeof(searchHistory));
        memset(killerMoves, 0, sizeof(killerMoves));

        NodeInfo rootNode;
        rootNode.position = &position;
        rootNode.depth = 0u;
        rootNode.isPvNode = true;
        rootNode.maxDepth = (uint8_t)depth;
        rootNode.alpha = alpha;
        rootNode.beta = beta;
        rootNode.color = position.GetSideToMove();

        SearchContext context;

        score = NegaMax(rootNode, context);

        // out of aspiration window, redo the search in full score range
        if (score <= alpha || score >= beta)
        {
            if (searchParam.debugLog)
            {
                //std::cout << "out of the aspiration window: alpha=" << alpha << " beta=" << beta << " score=" << score << std::endl;
            }
            aspirationWindow *= 2;
            alpha -= aspirationWindow;
            beta += aspirationWindow;
            depth--;
            continue;
        }

        const bool isMate = (score > CheckmateValue - 1000) || (score < -CheckmateValue + 1000);

        if (depth >= aspirationSearchStartDepth)
        {
            alpha = score - aspirationWindow;
            beta = score + aspirationWindow;
            aspirationWindow = (aspirationWindow + minAspirationWindow + 1) / 2; // narrow aspiration window
            ASSERT(aspirationWindow >= minAspirationWindow);
        }

        uint16_t pvLength = pvLengths[0];

        if (pvLength > 0)
        {
            outBestMove = position.MoveFromPacked(pvArray[0][0]);
            ASSERT(outBestMove.IsValid());
        }

        if (searchParam.debugLog)
        {
            std::cout << "info";
            std::cout << " depth " << (uint32_t)depth;
            if (isMate)
            {
                std::cout << " score mate " << (pvLength+1)/2;
            }
            else
            {
                std::cout << " score cp " << score;
            }
            std::cout << " nodes " << context.nodes;
            //std::cout << " ordering " << (context.fh > 0 ? 100.0f * (float)context.fhf / (float)context.fh : 0.0f) << "%";
            //std::cout << " branching " << ((float)context.pseudoMovesPerNode / (float)context.nodes);
            //std::cout << " ttHit " << context.ttHits;

            std::cout << " pv ";
            {
                prevPvArrayLength = pvLength;

                // reconstruct moves path
                Position iteratedPosition = position;
                for (uint32_t i = 0; i < pvLength; ++i)
                {
                    const Move move = iteratedPosition.MoveFromPacked(pvArray[0][i]);
                    ASSERT(move.IsValid());

                    prevPvArray[i] = { iteratedPosition.GetHash(), move };
                    std::cout << move.ToString() << " ";
                    const bool moveLegal = iteratedPosition.DoMove(move);
                    ASSERT(moveLegal);
                }
            }

            std::cout << std::endl;
        }
    }

    /*
    if (searchParam.debugLog)
    {
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Elapsed time: " << (std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000000.0) << std::endl;

        if (outBestMove.IsValid())
        {
            std::cout << "Best move:    " << outBestMove.ToString() << " (" << position.MoveToString(outBestMove) << ")" << std::endl;
        }
    }
    */

    return score;
}

static INLINE int32_t ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}

const Move Search::FindPvMove(uint32_t depth, const uint64_t positionHash, MoveList& moves) const
{
    ASSERT(depth < MaxSearchDepth);

    Move pvMove;
    if (depth < prevPvArrayLength && positionHash == prevPvArray[depth].positionHash)
    {
        pvMove = prevPvArray[depth].move;
    }

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        if (pvMove.IsValid() && moves.moves[i].move == pvMove)
        {
            moves.moves[i].score = INT32_MAX;
            break;
        }
    }

    return pvMove;
}

void Search::FindHistoryMoves(Color color, MoveList& moves) const
{
    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        const Move move = moves.moves[i].move;
        ASSERT(move.IsValid());

        const uint32_t pieceIndex = (uint32_t)(move.piece) - 1;
        ASSERT(pieceIndex < 6u);

        const uint32_t score = searchHistory[(uint32_t)color][pieceIndex][move.toSquare.Index()];
        const int64_t finalScore = (int64_t)moves.moves[i].score + score;
        moves.moves[i].score = (int32_t)std::min<uint64_t>(finalScore, INT32_MAX);
    }
}

void Search::FindKillerMoves(uint32_t depth, MoveList& moves) const
{
    ASSERT(depth < MaxSearchDepth);

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        for (uint32_t j = 0; j < NumKillerMoves; ++j)
        {
            if (moves.moves[i].move == killerMoves[depth][j])
            {
                moves.moves[i].score += 100000 - j;
            }
        }
    }
}

void Search::UpdatePvArray(uint32_t depth, const Move move)
{
    const uint16_t childPvLength = pvLengths[depth + 1];
    pvArray[depth][depth] = move;
    for (uint32_t j = depth + 1; j < childPvLength; ++j)
    {
        pvArray[depth][j] = pvArray[depth + 1][j];
    }
    pvLengths[depth] = childPvLength;
}

void Search::UpdateSearchHistory(const NodeInfo& node, const Move move)
{
    if (move.isCapture)
    {
        return;
    }

    const uint32_t pieceIndex = (uint32_t)(move.piece) - 1;
    ASSERT(pieceIndex < 6u);

    uint32_t& historyCounter = searchHistory[(uint32_t)node.color][pieceIndex][move.toSquare.Index()];

    const uint32_t historyBonus = node.maxDepth - node.depth;
    ASSERT(historyBonus > 0u);

    const uint64_t newValue = std::min<uint64_t>(UINT32_MAX, (uint64_t)historyCounter + (uint64_t)historyBonus * (uint64_t)historyBonus);
    historyCounter = (uint32_t)newValue;
}

bool Search::IsRepetition(const NodeInfo& node) const
{
    const NodeInfo* parentNode = node.parentNode;
    while (parentNode)
    {
        ASSERT(parentNode->position);
        if (parentNode->position->GetHash() == node.position->GetHash())
        {
            return parentNode->position == node.position;
        }

        if (parentNode->parentNode)
        {
            parentNode = parentNode->parentNode->parentNode;
        }
        else
        {
            parentNode = nullptr;
        }
    }

    return IsPositionRepeated(*node.position);
}

Search::ScoreType Search::QuiescenceNegaMax(const NodeInfo& node, SearchContext& ctx)
{
    ctx.quiescenceNodes++;

    if (IsRepetition(node))
    {
        return 0;
    }

    if (CheckInsufficientMaterial(*node.position))
    {
        return 0;
    }

    ScoreType score = ColorMultiplier(node.color) * Evaluate(*node.position);

    if (score >= node.beta)
    {
        return node.beta;
    }

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.depth = 0;
    childNodeParam.maxDepth = 0;
    childNodeParam.color = GetOppositeColor(node.color);

    uint32_t moveGenFlags = 0;
    if (!node.position->IsInCheck(node.color))
    {
        moveGenFlags |= MOVE_GEN_ONLY_CAPTURES;
    }

    MoveList moves;
    node.position->GenerateMoveList(moves, moveGenFlags);

    if (moves.numMoves > 1u)
    {
        FindPvMove(node.depth, node.position->GetHash(), moves);
    }

    Move bestMove;
    ScoreType alpha = std::max(score, node.alpha);
    ScoreType beta = node.beta;
    uint32_t numLegalMoves = 0;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.PickBestMove(i);

        Position childPosition = *node.position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        numLegalMoves++;

        childNodeParam.position = &childPosition;
        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;
        score = -QuiescenceNegaMax(childNodeParam, ctx);

        if (score > alpha)
        {
            alpha = score;
            bestMove = move;
        }

        if (score >= beta)
        {
            // for move ordering stats
            ctx.fh++;
            if (numLegalMoves == 1u) ctx.fhf++;
            return beta;
        }
    }

    return alpha;
}

void Search::PrefetchTranspositionTableEntry(const Position& position) const
{
    const size_t hashmapMask = transpositionTable.size() - 1;

    const TranspositionTableEntry& ttEntry = transpositionTable[position.GetHash() & hashmapMask];
    _mm_prefetch(reinterpret_cast<const char*>(&ttEntry), _MM_HINT_T0);
}

TranspositionTableEntry* Search::ReadTranspositionTable(const Position& position)
{
    const size_t hashmapMask = transpositionTable.size() - 1;

    TranspositionTableEntry& ttEntry = transpositionTable[position.GetHash() & hashmapMask];
    if (ttEntry.positionHash == position.GetHash() && ttEntry.flag != TranspositionTableEntry::Flag_Invalid)
    {
        return &ttEntry;
    }

    return nullptr;
}

void Search::WriteTranspositionTable(const TranspositionTableEntry& entry)
{
    const size_t hashmapMask = transpositionTable.size() - 1;

    transpositionTable[entry.positionHash & hashmapMask] = entry;
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

Search::ScoreType Search::NegaMax(const NodeInfo& node, SearchContext& ctx)
{
    pvLengths[node.depth] = node.depth;

    if (IsRepetition(node))
    {
        return 0;
    }

    if (CheckInsufficientMaterial(*node.position))
    {
        return 0;
    }

    const bool isInCheck = node.position->IsInCheck(node.color);
    const uint16_t inversedDepth = node.maxDepth - node.depth;

    const ScoreType oldAlpha = node.alpha;
    ScoreType alpha = node.alpha;
    ScoreType beta = node.beta;

    // transposition table lookup
    PackedMove ttMove;
    TranspositionTableEntry* ttEntry = ReadTranspositionTable(*node.position);

    if (ttEntry)
    {
        if (ttEntry->depth >= inversedDepth)
        {
            ctx.ttHits++;

            if (ttEntry->flag == TranspositionTableEntry::Flag_Exact)
            {
                return ttEntry->score;
            }
            else if (ttEntry->flag == TranspositionTableEntry::Flag_LowerBound)
            {
                alpha = std::max(alpha, ttEntry->score);
            }
            else if (ttEntry->flag == TranspositionTableEntry::Flag_UpperBound)
            {
                beta = std::min(beta, ttEntry->score);
            }

            if (alpha >= beta)
            {
                return alpha;
            }
        }

        ttMove = ttEntry->move;
    }

    if (node.depth >= node.maxDepth)
    {
        return QuiescenceNegaMax(node, ctx);
    }

    ctx.nodes++;

    NodeInfo childNodeParam;
    childNodeParam.parentNode = &node;
    childNodeParam.depth = node.depth + 1;
    childNodeParam.color = GetOppositeColor(node.color);

    uint8_t childNodeMaxDepth = node.maxDepth;

    // check extension
    if (isInCheck)
    {
        if (childNodeMaxDepth < UINT8_MAX)
        {
            childNodeMaxDepth++;
        }
    }

    MoveList moves;
    node.position->GenerateMoveList(moves);
    ctx.pseudoMovesPerNode += moves.numMoves;

    const Move pvMove = FindPvMove(node.depth, node.position->GetHash(), moves);

    if (moves.numMoves > 1u)
    {
        FindHistoryMoves(node.color, moves);
        FindKillerMoves(node.depth, moves);

        if (ttMove.IsValid())
        {
            for (uint32_t i = 0; i < moves.numMoves; ++i)
            {
                if (moves.moves[i].move == ttMove)
                {
                    moves.moves[i].score = INT32_MAX - 1;
                    break;
                }
            }
        }
    }

    // mate distance prunning
    {
        int32_t mateDistanceScore = PruneByMateDistance(node, alpha, beta);
        if (mateDistanceScore != 0)
        {
            return mateDistanceScore;
        }
    }

    if (node.isPvNode)
    {
        //std::cout << "Moves at PV node, depth " << (uint32_t)node.depth << std::endl;
        //moves.Print();
    }

    Move bestMove;
    uint32_t numLegalMoves = 0;
    uint32_t numQuietMoves = 0;
    bool betaCutoff = false;

    // count (pseudo) quiet moves
    uint32_t totalQuietMoves = 0;
    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.moves[i].move;
        if (!move.isCapture && move.promoteTo == Piece::None)
        {
            totalQuietMoves++;
        }
    }

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.PickBestMove(i);
        ASSERT(move.IsValid());

        Position childPosition = *node.position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        PrefetchTranspositionTableEntry(childPosition);

        // store any best move, in case of we never improve alpha in this loop,
        // so we can write anything into transposition table
        if (numLegalMoves == 0)
        {
            bestMove = move;
        }

        numLegalMoves++;

        childNodeParam.isPvNode = pvMove == move;
        childNodeParam.maxDepth = childNodeMaxDepth;

        if (!move.isCapture && move.promoteTo == Piece::None)
        {
            numQuietMoves++;

            // Late Move Reduction
            // don't reduce PV moves, while in check
            if (!isInCheck && totalQuietMoves > 0 && numLegalMoves > 1u && node.depth >= 5)
            {
                // 0% reduction for first quiet move
                // 50% reduction for last quiet move
                //int32_t depthReduction = node.maxDepth * numQuietMoves / totalQuietMoves / 2;
                //childNodeParam.maxDepth = (uint8_t)std::max(1, (int32_t)childNodeMaxDepth - depthReduction);

                int32_t depthReduction = numQuietMoves > (totalQuietMoves / 2) ? 1 : 0;
                childNodeParam.maxDepth = (uint8_t)std::max(1, (int32_t)childNodeMaxDepth - depthReduction);
            }
        }

        ScoreType score;
        {
            childNodeParam.position = &childPosition;

            if (numLegalMoves == 1)
            {
                childNodeParam.alpha = -beta;
                childNodeParam.beta = -alpha;
                score = -NegaMax(childNodeParam, ctx);
            }
            else // Principal Variation Search
            {
                childNodeParam.alpha = -alpha - 1;
                childNodeParam.beta = -alpha;
                score = -NegaMax(childNodeParam, ctx);

                if (score > alpha && score < beta)
                {
                    childNodeParam.alpha = -beta;
                    childNodeParam.beta = -alpha;
                    score = -NegaMax(childNodeParam, ctx);
                }
            }
        }

        if (score > alpha) // new best move found
        {
            bestMove = move;
            alpha = score;

            UpdatePvArray(node.depth, move);
            UpdateSearchHistory(node, move);
        }

        if (score >= beta) // beta cutoff
        {
            // for move ordering stats
            ctx.fh++;
            if (numLegalMoves == 1u) ctx.fhf++;

            if (!move.isCapture)
            {
                for (uint32_t j = NumKillerMoves; j-- > 1u; )
                {
                    killerMoves[node.depth][j] = killerMoves[node.depth][j - 1];
                }
                killerMoves[node.depth][0] = move;
            }

            betaCutoff = true;
            break;
        }
    }

    if (numLegalMoves == 0u)
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

    ASSERT(bestMove.IsValid());

    // update transposition table
    {
        TranspositionTableEntry::Flags flag = TranspositionTableEntry::Flag_Exact;
        if (alpha <= oldAlpha)
        {
            flag = TranspositionTableEntry::Flag_UpperBound;
        }
        else if (betaCutoff)
        {
            flag = TranspositionTableEntry::Flag_LowerBound;
        }

        const TranspositionTableEntry entry{ node.position->GetHash(), alpha, bestMove, (uint8_t)inversedDepth, flag };

        WriteTranspositionTable(entry);
    }

    ASSERT(alpha > -CheckmateValue && alpha < CheckmateValue);

    return alpha;
}