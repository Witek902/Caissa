#include "Search.hpp"
#include "Move.hpp"

#include <iostream>
#include <string>

static constexpr int32_t c_kingValue        = 1000;
static constexpr int32_t c_queenValue       = 900;
static constexpr int32_t c_rookValue        = 500;
static constexpr int32_t c_bishopValue      = 330;
static constexpr int32_t c_knightValue      = 320;
static constexpr int32_t c_pawnValue        = 100;

static constexpr int32_t c_mobilityBonus    = 2;

Search::ScoreType Search::Evaluate(const Position& position)
{
    ScoreType value = 0;

    value += c_queenValue * ((int32_t)position.mWhites.queens.Count() - (int32_t)position.mBlacks.queens.Count());
    value += c_rookValue * ((int32_t)position.mWhites.rooks.Count() - (int32_t)position.mBlacks.rooks.Count());
    value += c_bishopValue * ((int32_t)position.mWhites.bishops.Count() - (int32_t)position.mBlacks.bishops.Count());
    value += c_knightValue * ((int32_t)position.mWhites.knights.Count() - (int32_t)position.mBlacks.knights.Count());
    value += c_pawnValue * ((int32_t)position.mWhites.pawns.Count() - (int32_t)position.mBlacks.pawns.Count());

    const Bitboard whiteAttackedSquares = position.GetAttackedSquares(Color::White);
    const Bitboard blackAttackedSquares = position.GetAttackedSquares(Color::Black);
    const Bitboard whiteOccupiedSquares = position.mWhites.Occupied();
    const Bitboard blackOccupiedSquares = position.mBlacks.Occupied();
    const Bitboard whitesMobility = whiteAttackedSquares & whiteOccupiedSquares;
    const Bitboard blacksMobility = blackAttackedSquares & blackOccupiedSquares;

    value += c_mobilityBonus * ((int32_t)whitesMobility.Count() - (int32_t)blacksMobility.Count());

    return value;
}

Search::ScoreType Search::DoSearch(const Position& position, Move& outBestMove)
{
    uint16_t maxDepth = 6u;

    ScoreType score;

    for (uint16_t depth = 1; depth <= maxDepth; ++depth)
    {
        NegaMaxParam param;
        param.depth = 0u;
        param.maxDepth = depth;
        param.alpha = -InfValue;
        param.beta = InfValue;
        param.color = position.GetSideToMove();

        SearchContext context;

        score = NegaMax(position, param, context, outBestMove);

        std::cout << "depth: " << depth << ", ";
        std::cout << "best: " << position.MoveToString(outBestMove) << ", ";

        /*
        std::cout << "PV:        ";
        {
            // reconstruct moves path
            Position iteratedPosition = position;
            for (uint32_t i = 0; i < depth; ++i)
            {
                const Move move = context.moves[i];
                std::cout << iteratedPosition.MoveToString(move) << " ";
                const bool moveLegal = iteratedPosition.DoMove(move);
                ASSERT(moveLegal);
            }
            std::cout << std::endl;
        }
        */

        if (score > -CheckmateValue - 1000)
        {
            std::cout << "val: Blacks Mate in " << (-CheckmateValue - score) << "!, ";
        }
        else if (score < CheckmateValue + 1000)
        {
            std::cout << "val: Whites Mate in " << (score - CheckmateValue) << "!, ";
        }
        else
        {
            std::cout << "val: " << (float)score / 100.0f << " ";
        }
        std::cout << "nodes: " << context.nodes; // << " (" << context.quiescenceNodes << "q)" << "\t";
        std::cout << std::endl;
    }

    return score;
}

static INLINE int32_t ColorMultiplier(Color color)
{
    return color == Color::White ? 1 : -1;
}

Search::ScoreType Search::QuiescenceNegaMax(const Position& position, const NegaMaxParam& param, SearchContext& ctx)
{
    // TODO check for repetition
    // TODO max quiescence depth?

    ScoreType score = ColorMultiplier(param.color) * Evaluate(position);

    return score;

    if (score >= param.beta)
    {
        return param.beta;
    }

    NegaMaxParam childNodeParam;
    childNodeParam.depth = 0;
    childNodeParam.maxDepth = 0;
    childNodeParam.color = GetOppositeColor(param.color);

    MoveList moves;
    position.GenerateMoveList(moves, MOVE_GEN_ONLY_CAPTURES);
    moves.Sort();

    ScoreType alpha = std::max(score, param.alpha);
    ScoreType beta = param.beta;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move& move = moves.GetMove(i);
        ASSERT(move.isCapture);

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        ctx.quiescenceNodes++;

        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;
        score = -QuiescenceNegaMax(childPosition, childNodeParam, ctx);

        if (score > alpha)
        {
            alpha = score;
        }

        if (score >= beta)
        {
            break;
        }
    }

    // TODO transposition tables
    // TODO iterative deepening

    return alpha;
}

Search::ScoreType Search::NegaMax(const Position& position, const NegaMaxParam& param, SearchContext& ctx, Move& outBestMove)
{
    if (param.depth >= param.maxDepth)
    {
        return QuiescenceNegaMax(position, param, ctx);
    }

    NegaMaxParam childNodeParam;
    childNodeParam.depth = param.depth + 1;
    childNodeParam.maxDepth = param.maxDepth;
    childNodeParam.color = GetOppositeColor(param.color);

    MoveList moves;
    position.GenerateMoveList(moves);
    moves.Sort();

    ScoreType alpha = param.alpha;
    ScoreType beta = param.beta;
    bool hasLegalMoves = false;

    for (uint32_t i = 0; i < moves.Size(); ++i)
    {
        const Move& move = moves.GetMove(i);

        Position childPosition = position;
        if (!childPosition.DoMove(move))
        {
            continue;
        }

        hasLegalMoves = true;
        ctx.nodes++;

        childNodeParam.alpha = -beta;
        childNodeParam.beta = -alpha;

        Move childBestMove;
        const ScoreType score = -NegaMax(childPosition, childNodeParam, ctx, childBestMove);

        if (score > alpha)
        {
            ctx.moves[param.depth] = move;
            outBestMove = move;
            alpha = score;
        }

        if (score >= beta)
        {
            break;
        }
    }

    if (!hasLegalMoves)
    {
        if (position.IsInCheck(param.color)) // checkmate
        {
            return CheckmateValue + param.depth;
        }
        else // stalemate
        {
            return 0;
        }
    }

    // TODO transposition tables
    // TODO iterative deepening

    ASSERT(alpha > CheckmateValue && alpha < -CheckmateValue);

    return alpha;
}