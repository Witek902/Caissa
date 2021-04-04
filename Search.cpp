#include "Search.hpp"
#include "Move.hpp"
#include "Evaluate.hpp"

#include <iostream>
#include <string>
#include <chrono>

Search::Search()
{
    pvTable.resize(PvTableSize);
}

Search::ScoreType Search::DoSearch(const Position& position, Move& outBestMove)
{
    uint16_t maxDepth = 8u;

    ScoreType score = 0;

    auto start = std::chrono::high_resolution_clock::now();

    pvTable.clear();
    memset(searchHistory, 0, sizeof(searchHistory));
    memset(killerMoves, 0, sizeof(killerMoves));

    for (uint16_t depth = 1; depth <= maxDepth; ++depth)
    {
        NegaMaxParam param;
        param.position = &position;
        param.positionHash = position.GetHash();
        param.depth = 0u;
        param.maxDepth = depth;
        param.alpha = -InfValue;
        param.beta = InfValue;
        param.color = position.GetSideToMove();

        SearchContext context;

        score = NegaMax(param, context, &outBestMove);

        std::cout << "depth " << depth << ", ";
        std::cout << "best " << position.MoveToString(outBestMove) << ", ";

        uint32_t pvLength = depth;
        if (score > -CheckmateValue - 1000)
        {
            pvLength = -CheckmateValue - score;
            std::cout << "val: Blacks Mate in " << pvLength;
        }
        else if (score < CheckmateValue + 1000)
        {
            pvLength = score - CheckmateValue;
            std::cout << "val: Whites Mate in " << pvLength;
        }
        else
        {
            std::cout << "val: " << (float)score / 100.0f;
        }
        std::cout << ", nodes: " << context.nodes << " (" << context.quiescenceNodes << "q)";
        std::cout << ", ordering: " << (context.fh > 0 ? 100.0f * (float)context.fhf / (float)context.fh : 0.0f) << "%";

        /*
        std::cout << ", PV: ";
        {
            // reconstruct moves path
            Position iteratedPosition = position;
            for (uint32_t i = 0; i < pvLength; ++i)
            {
                const auto iter = pvTable.find(iteratedPosition.GetHash());
                ASSERT(iter != pvTable.end());
                const Move move = iter->second;
                std::cout << iteratedPosition.MoveToString(move) << " ";
                const bool moveLegal = iteratedPosition.DoMove(move);
                ASSERT(moveLegal);
            }
        }
        */

        std::cout << std::endl;
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000000.0 << " s\n";
    std::cout << "Best move:    " << position.MoveToString(outBestMove) << std::endl;

    return score;
}

static INLINE int32_t ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}

void Search::FindPvMove(const uint64_t positionHash, MoveList& moves) const
{
    const PvTableEntry& entry = pvTable[positionHash % PvTableSize];
    if (entry.positionHash != positionHash)
    {
        return;
    }

    const Move pvMove = entry.move;
    if (!entry.move.IsValid())
    {
        return;
    }

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        if (moves.moves[i].move == pvMove)
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

        if (moves.moves[i].score < INT32_MAX)
        {
            const uint32_t pieceIndex = (uint32_t)(move.piece) - 1;
            ASSERT(pieceIndex < 6u);

            const uint64_t score = searchHistory[(uint32_t)color][pieceIndex][move.toSquare.Index()];
            const int64_t finalScore = moves.moves[i].score + score;
            moves.moves[i].score = (int32_t)std::min<uint64_t>(finalScore, INT32_MAX);
        }
    }
}

void Search::FindKillerMoves(uint32_t depth, MoveList& moves) const
{
    ASSERT(depth < MaxSearchDepth);

    for (uint32_t i = 0; i < NumKillerMoves; ++i)
    {
        if (moves.moves[i].score < INT32_MAX)
        {
            if (moves.moves[i].move == killerMoves[depth][i])
            {
                moves.moves[i].score += 100000 - i;
            }
        }
    }
}

void Search::UpdatePvEntry(uint64_t positionHash, const Move move)
{
    PvTableEntry& entry = pvTable[positionHash % PvTableSize];

    entry = { positionHash, move };
}

bool Search::IsRepetition(const NegaMaxParam& param)
{
    const NegaMaxParam* parentParam = param.parentParam;
    while (parentParam)
    {
        ASSERT(parentParam->position);
        if (parentParam->positionHash == param.positionHash)
        {
            // TODO double check (in case of hash collision)
            return true;
        }

        // TODO skip 2
        parentParam = parentParam->parentParam;
    }

    return false;
}

Search::ScoreType Search::QuiescenceNegaMax(const NegaMaxParam& param, SearchContext& ctx)
{
    ASSERT(param.positionHash);

    if (IsRepetition(param))
    {
        return 0;
    }

    // TODO max quiescence depth?

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
        FindPvMove(param.positionHash, moves);
    }

    Move bestMove;
    ScoreType alpha = std::max(score, param.alpha);
    ScoreType oldAlpha = alpha;
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
        childNodeParam.positionHash = childPosition.GetHash();
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

    if (alpha != oldAlpha)
    {
        ASSERT(bestMove.IsValid());
        UpdatePvEntry(param.positionHash, bestMove);
    }

    return alpha;
}

Search::ScoreType Search::NegaMax(const NegaMaxParam& param, SearchContext& ctx, Move* outBestMove)
{
    ASSERT(param.positionHash);

    if (param.depth >= param.maxDepth)
    {
        return QuiescenceNegaMax(param, ctx);
    }

    if (IsRepetition(param))
    {
        return 0;
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
        FindPvMove(param.positionHash, moves);
        FindHistoryMoves(param.color, moves);
        FindKillerMoves(param.depth, moves);
    }

    Move bestMove;
    ScoreType oldAlpha = param.alpha;
    ScoreType alpha = param.alpha;
    ScoreType beta = param.beta;
    uint32_t numLegalMoves = 0;

    //if (param.depth == 0)
    //{
    //    moves.Print();
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

        numLegalMoves++;
        ctx.nodes++;

        childNodeParam.position = &childPosition;
        childNodeParam.positionHash = childPosition.GetHash();
        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;

        const ScoreType score = -NegaMax(childNodeParam, ctx);

        if (score > alpha)
        {
            bestMove = move;
            alpha = score;

            if (!move.isCapture && !(move.piece==Piece::Pawn && move.isEnPassant))
            {
                const uint32_t pieceIndex = (uint32_t)(move.piece) - 1;
                ASSERT(pieceIndex < 6u);

                const uint32_t historyBonus = param.maxDepth - param.depth;
                searchHistory[(uint32_t)param.color][pieceIndex][move.toSquare.Index()] += historyBonus;
            }
        }

        if (score >= beta)
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

            break;
        }
    }

    if (numLegalMoves == 0u)
    {
        if (param.position->IsInCheck(param.color)) // checkmate
        {
            return CheckmateValue + param.depth;
        }
        else // stalemate
        {
            return 0;
        }
    }

    if (alpha != oldAlpha)
    {
        ASSERT(bestMove.IsValid());
        UpdatePvEntry(param.positionHash, bestMove);

        if (outBestMove)
        {
            *outBestMove = bestMove;
        }
    }

    ASSERT(alpha > CheckmateValue && alpha < -CheckmateValue);

    return alpha;
}