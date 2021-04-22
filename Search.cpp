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
    transpositionTable.resize(TranspositionTableSize);

    int32_t aspirationWindow = 400;
    const int32_t minAspirationWindow = 40;
    const uint32_t aspirationSearchStartDepth = 20;

    int32_t alpha = -InfValue;
    int32_t beta  =  InfValue;

    auto start = std::chrono::high_resolution_clock::now();

    for (uint8_t depth = 1; depth <= searchParam.maxDepth; ++depth)
    {
        memset(pvArray, 0, sizeof(pvArray));
        memset(pvLengths, 0, sizeof(pvLengths));
        memset(searchHistory, 0, sizeof(searchHistory));
        memset(killerMoves, 0, sizeof(killerMoves));

        NegaMaxParam negaMaxParam;
        negaMaxParam.position = &position;
        negaMaxParam.depth = 0u;
        negaMaxParam.maxDepth = depth;
        negaMaxParam.alpha = alpha;
        negaMaxParam.beta = beta;
        negaMaxParam.color = position.GetSideToMove();

        SearchContext context;

        score = NegaMax(negaMaxParam, context);

        if (searchParam.debugLog)
        {
            std::cout << "depth " << (uint32_t)depth << " ";
            std::cout << "window " << aspirationWindow << " ";
        }

        // out of aspiration window, redo the search in full score range
        if (score <= alpha || score >= beta)
        {
            if (searchParam.debugLog)
            {
                std::cout << "out of the aspiration window: alpha=" << alpha << " beta=" << beta << " score=" << score << std::endl;
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
            if (isMate)
            {
                std::cout << "mate " << pvLength;
            }
            else
            {
                std::cout << "val " << (float)score / 100.0f;
            }
            std::cout << " nodes " << context.nodes << " (" << context.quiescenceNodes << "q)";
            std::cout << " (ordering " << (context.fh > 0 ? 100.0f * (float)context.fhf / (float)context.fh : 0.0f) << "%)";
            std::cout << " ttHit " << context.ttHits;

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
                    std::cout << iteratedPosition.MoveToString(move) << " ";
                    const bool moveLegal = iteratedPosition.DoMove(move);
                    ASSERT(moveLegal);
                }
            }

            std::cout << std::endl;
        }
    }

    if (searchParam.debugLog)
    {
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Elapsed time: " << (std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000000.0) << std::endl;

        if (outBestMove.IsValid())
        {
            std::cout << "Best move:    " << outBestMove.ToString() << " (" << position.MoveToString(outBestMove) << ")" << std::endl;
        }
    }

    return score;
}

static INLINE int32_t ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}

void Search::FindPvMove(uint32_t depth, const uint64_t positionHash, MoveList& moves) const
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
}

void Search::FindHistoryMoves(Color color, MoveList& moves) const
{
    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        const Move move = moves.moves[i].move;
        ASSERT(move.IsValid());

        const uint32_t pieceIndex = (uint32_t)(move.piece) - 1;
        ASSERT(pieceIndex < 6u);

        const uint64_t score = searchHistory[(uint32_t)color][pieceIndex][move.toSquare.Index()];
        const int64_t finalScore = moves.moves[i].score + score;
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

bool Search::IsRepetition(const NegaMaxParam& param) const
{
    const NegaMaxParam* parentParam = param.parentParam;
    while (parentParam)
    {
        ASSERT(parentParam->position);
        if (parentParam->position->GetHash() == param.position->GetHash())
        {
            // TODO double check (in case of hash collision)
            return true;
        }

        // TODO skip 2
        parentParam = parentParam->parentParam;
    }

    return IsPositionRepeated(*param.position);
}

Search::ScoreType Search::QuiescenceNegaMax(const NegaMaxParam& param, SearchContext& ctx)
{
    if (IsRepetition(param))
    {
        return 0;
    }

    ScoreType score = ColorMultiplier(param.color) * Evaluate(*param.position);

    if (score >= param.beta)
    {
        return param.beta;
    }

    NegaMaxParam childNodeParam;
    childNodeParam.parentParam = &param;
    childNodeParam.depth = 0;
    childNodeParam.maxDepth = 0;
    childNodeParam.color = GetOppositeColor(param.color);

    MoveList moves;
    param.position->GenerateMoveList(moves, MOVE_GEN_ONLY_CAPTURES);

    if (moves.numMoves > 1u)
    {
        FindPvMove(param.depth, param.position->GetHash(), moves);
    }

    Move bestMove;
    ScoreType alpha = std::max(score, param.alpha);
    ScoreType beta = param.beta;
    uint32_t numLegalMoves = 0;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.PickBestMove(i);
        ASSERT(move.isCapture);

        Position childPosition = *param.position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        ctx.quiescenceNodes++;
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
    const TranspositionTableEntry& ttEntry = transpositionTable[position.GetHash() % TranspositionTableSize];
    _mm_prefetch(reinterpret_cast<const char*>(&ttEntry), _MM_HINT_T0);
}

Search::ScoreType Search::NegaMax(const NegaMaxParam& param, SearchContext& ctx)
{
    pvLengths[param.depth] = param.depth;

    if (IsRepetition(param))
    {
        return 0;
    }

    const uint16_t inversedDepth = param.maxDepth - param.depth;

    const ScoreType oldAlpha = param.alpha;
    ScoreType alpha = param.alpha;
    ScoreType beta = param.beta;

    // transposition table lookup
    PackedMove ttMove;
    TranspositionTableEntry& ttEntry = transpositionTable[param.position->GetHash() % TranspositionTableSize];
    {
        if (ttEntry.positionHash == param.position->GetHash() && ttEntry.flag != TranspositionTableEntry::Flag_Invalid)
        {
            if (ttEntry.depth >= inversedDepth)
            {
                ctx.ttHits++;

                if (ttEntry.flag == TranspositionTableEntry::Flag_Exact)
                {
                    return ttEntry.score;
                }
                else if (ttEntry.flag == TranspositionTableEntry::Flag_LowerBound)
                {
                    alpha = std::max(alpha, ttEntry.score);
                }
                else if (ttEntry.flag == TranspositionTableEntry::Flag_UpperBound)
                {
                    beta = std::min(beta, ttEntry.score);
                }

                if (alpha >= beta)
                {
                    return alpha;
                }
            }

            ttMove = ttEntry.move;
        }
    }

    if (param.depth >= param.maxDepth)
    {
        return QuiescenceNegaMax(param, ctx);
    }

    NegaMaxParam childNodeParam;
    childNodeParam.parentParam = &param;
    childNodeParam.depth = param.depth + 1;
    childNodeParam.maxDepth = param.maxDepth;
    childNodeParam.color = GetOppositeColor(param.color);

    MoveList moves;
    param.position->GenerateMoveList(moves);

    if (moves.numMoves > 1u)
    {
        FindHistoryMoves(param.color, moves);
        FindKillerMoves(param.depth, moves);

        FindPvMove(param.depth, param.position->GetHash(), moves);

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
        int32_t matingValue = CheckmateValue - param.depth;
        if (matingValue < beta)
        {
            beta = matingValue;
            if (alpha >= matingValue)
            {
                return matingValue;
            }
        }

        matingValue = -CheckmateValue + param.depth;
        if (matingValue > alpha)
        {
            alpha = matingValue;
            if (beta <= matingValue)
            {
                return matingValue;
            }
        }
    }

    //if (param.depth == 0)
    //{
    //    moves.Print();
    //}

    Move bestMove;
    uint32_t numLegalMoves = 0;
    bool betaCutoff = false;

    const uint32_t lateMovePrunningDepthTreshold = 5;
    uint32_t numMovesToSearch = moves.Size();
    //if (param.depth >= lateMovePrunningDepthTreshold && numMovesToSearch > 1)
    //{
    //    numMovesToSearch = std::max(1u, numMovesToSearch / (2 + param.depth - lateMovePrunningDepthTreshold));
    //}

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move move = moves.PickBestMove(i);
        ASSERT(move.IsValid());

        Position childPosition = *param.position;
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
        ctx.nodes++;

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

            UpdatePvArray(param.depth, move);

            if (!move.isCapture && !(move.piece == Piece::Pawn && move.isEnPassant))
            {
                const uint32_t pieceIndex = (uint32_t)(move.piece) - 1;
                ASSERT(pieceIndex < 6u);

                const uint32_t historyBonus = param.maxDepth - param.depth;
                searchHistory[(uint32_t)param.color][pieceIndex][move.toSquare.Index()] += historyBonus * historyBonus;
            }
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
                    killerMoves[param.depth][j] = killerMoves[param.depth][j - 1];
                }
                killerMoves[param.depth][0] = move;
            }

            betaCutoff = true;
            break;
        }

        if (numLegalMoves >= numMovesToSearch)
        {
            break;
        }
    }

    if (numLegalMoves == 0u)
    {
        if (param.position->IsInCheck(param.color)) // checkmate
        {
            return -CheckmateValue + param.depth;
        }
        else // stalemate
        {
            return 0;
        }
    }

    ASSERT(bestMove.IsValid());

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

        ttEntry = { param.position->GetHash(), alpha, bestMove, (uint8_t)inversedDepth, flag };
    }

    ASSERT(alpha > -CheckmateValue && alpha < CheckmateValue);

    return alpha;
}