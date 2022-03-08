#include "MoveOrderer.hpp"
#include "Search.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"

#include <algorithm>
#include <limits>

static constexpr int32_t RecaptureBonus = 1000;

static const int32_t c_PromotionScores[] =
{
    0,          // none
    0,          // pawn
    1000,       // knight
    0,          // bishop
    0,          // rook
    10000000,   // queen
};

static int32_t ComputeMvvLvaScore(const Piece capturedPiece, const Piece attackingPiece)
{
    return 8192 * (int32_t)capturedPiece - (int32_t)attackingPiece;
}

void MoveOrderer::DebugPrint() const
{
#ifndef CONFIGURATION_FINAL
    std::cout << "=== QUIET MOVES HISTORY HEURISTICS ===" << std::endl;

    for (uint32_t fromIndex = 0; fromIndex < 64; ++fromIndex)
    {
        for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
        {
            for (uint32_t color = 0; color < 2; ++color)
            {
                const CounterType count = quietMoveHistory[color][fromIndex][toIndex];

                if (count)
                {
                    std::cout
                        << Square(fromIndex).ToString() << Square(toIndex).ToString()
                        << (color > 0 ? " (black)" : " (white)")
                        << " ==> " << count << '\n';
                }
            }
        }
    }

    std::cout << "=== CAPTURES HISTORY HEURISTICS ===" << std::endl;

    for (uint32_t fromIndex = 0; fromIndex < 64; ++fromIndex)
    {
        for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
        {
            for (uint32_t color = 0; color < 2; ++color)
            {
                const CounterType count = capturesHistory[color][fromIndex][toIndex];

                if (count)
                {
                    std::cout
                        << Square(fromIndex).ToString() << Square(toIndex).ToString()
                        << (color > 0 ? " (black)" : " (white)")
                        << " ==> " << count << '\n';
                }
            }
        }
    }

    std::cout << "=== QUIET MOVES CONTINUATION HISTORY HEURISTICS ===" << std::endl;

    for (uint32_t prevPiece = 0; prevPiece < 6; ++prevPiece)
    {
        for (uint32_t prevToIndex = 0; prevToIndex < 64; ++prevToIndex)
        {
            for (uint32_t piece = 0; piece < 6; ++piece)
            {
                for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
                {
                    const CounterType count = quietMoveContinuationHistory[prevPiece][prevToIndex][piece][toIndex];

                    if (count)
                    {
                        std::cout
                            << PieceToChar(Piece(prevPiece + 1)) << Square(prevToIndex).ToString()
                            << ", "
                            << PieceToChar(Piece(piece + 1)) << Square(toIndex).ToString()
                            << " ==> " << count << '\n';
                    }
                }
            }
        }
    }

    std::cout << "=== QUIET MOVES FOLLOWUP HISTORY HEURISTICS ===" << std::endl;

    for (uint32_t prevPiece = 0; prevPiece < 6; ++prevPiece)
    {
        for (uint32_t prevToIndex = 0; prevToIndex < 64; ++prevToIndex)
        {
            for (uint32_t piece = 0; piece < 6; ++piece)
            {
                for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
                {
                    const CounterType count = quietMoveFollowupHistory[prevPiece][prevToIndex][piece][toIndex];

                    if (count)
                    {
                        std::cout
                            << PieceToChar(Piece(prevPiece + 1)) << Square(prevToIndex).ToString()
                            << ", "
                            << PieceToChar(Piece(piece + 1)) << Square(toIndex).ToString()
                            << " ==> " << count << '\n';
                    }
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << "=== KILLER MOVE HEURISTICS ===" << std::endl;

    for (uint32_t d = 0; d < MaxSearchDepth; ++d)
    {
        std::cout << d;

        bool hasAnyValid = false;

        for (uint32_t i = 0; i < NumKillerMoves; ++i)
        {
            hasAnyValid |= killerMoves[d][i].IsValid();
            std::cout << "\t" << killerMoves[d][i].ToString() << " ";
        }

        if (!hasAnyValid)
        {
            break;
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
#endif // CONFIGURATION_FINAL
}

void MoveOrderer::Clear()
{
    //const CounterType scaleDownFactor = 16;
    //for (uint32_t i = 0; i < sizeof(searchHistory) / sizeof(CounterType); ++i)
    //{
    //    reinterpret_cast<CounterType*>(searchHistory)[i] /= scaleDownFactor;
    //}
    //for (uint32_t i = 0; i < sizeof(continuationHistory) / sizeof(CounterType); ++i)
    //{
    //    reinterpret_cast<CounterType*>(continuationHistory)[i] /= scaleDownFactor;
    //}
    //for (uint32_t i = 0; i < sizeof(followupHistory) / sizeof(CounterType); ++i)
    //{
    //    reinterpret_cast<CounterType*>(followupHistory)[i] /= scaleDownFactor;
    //}

    //memset(capturesHistory, 0, sizeof(capturesHistory));
    memset(quietMoveHistory, 0, sizeof(quietMoveHistory));
    memset(quietMoveContinuationHistory, 0, sizeof(quietMoveContinuationHistory));
    memset(quietMoveFollowupHistory, 0, sizeof(quietMoveFollowupHistory));
    memset(killerMoves, 0, sizeof(killerMoves));
}

void MoveOrderer::UpdateHistoryCounter(CounterType& counter, int32_t delta)
{
    int32_t newValue = counter;
    newValue += 16 * delta;
    newValue -= counter * std::abs(delta) / 1024;

    // there should be no saturation
    ASSERT(newValue > std::numeric_limits<CounterType>::min());
    ASSERT(newValue < std::numeric_limits<CounterType>::max());

    counter = (CounterType)newValue;
}

void MoveOrderer::UpdateQuietMovesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth)
{
    ASSERT(depth >= 0);

    // don't update uncertain moves
    if (numMoves <= 1 && depth <= 2)
    {
        return;
    }

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    const Move prevMove = !node.isNullMove ? node.previousMove : Move::Invalid();
    const Move followupMove = node.parentNode && !node.parentNode->isNullMove ? node.parentNode->previousMove : Move::Invalid();

    const int32_t bonus = std::min(depth * depth, 256);

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];
        const int32_t delta = move == bestMove ? bonus : -bonus;

        const uint32_t piece = (uint32_t)move.GetPiece() - 1;
        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();

        UpdateHistoryCounter(quietMoveHistory[color][from][to], delta);

        if (prevMove.IsValid())
        {
            const uint32_t prevPiece = (uint32_t)prevMove.GetPiece() - 1;
            const uint32_t prevTo = prevMove.ToSquare().Index();
            UpdateHistoryCounter(quietMoveContinuationHistory[prevPiece][prevTo][piece][to], delta);
        }

        if (followupMove.IsValid())
        {
            const uint32_t prevPiece = (uint32_t)followupMove.GetPiece() - 1;
            const uint32_t prevTo = followupMove.ToSquare().Index();
            UpdateHistoryCounter(quietMoveFollowupHistory[prevPiece][prevTo][piece][to], delta);
        }
    }
}

void MoveOrderer::UpdateCapturesHistory(const NodeInfo& node, const Move* moves, uint32_t numMoves, const Move bestMove, int32_t depth)
{
    ASSERT(depth >= 0);

    // don't update uncertain moves
    if (numMoves <= 1 && depth <= 2)
    {
        return;
    }

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    const int32_t bonus = std::min(depth * depth, 256);

    for (uint32_t i = 0; i < numMoves; ++i)
    {
        const Move move = moves[i];
        const int32_t delta = move == bestMove ? bonus : -bonus;

        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();

        UpdateHistoryCounter(capturesHistory[color][from][to], delta);
    }
}

void MoveOrderer::UpdateKillerMove(const NodeInfo& node, const Move move)
{
    if (node.height < MaxSearchDepth)
    {
        for (uint32_t j = 0; j < NumKillerMoves; ++j)
        {
            if (move == killerMoves[node.height][j])
            {
                // move to the front
                std::swap(killerMoves[node.height][0], killerMoves[node.height][j]);

                return;
            }
        }

        for (uint32_t j = NumKillerMoves; j-- > 1u; )
        {
            killerMoves[node.height][j] = killerMoves[node.height][j - 1];
        }
        killerMoves[node.height][0] = move;
    }
}

void MoveOrderer::ScoreMoves(const NodeInfo& node, MoveList& moves) const
{
    const Position& pos = node.position;

    const uint32_t color = (uint32_t)node.position.GetSideToMove();

    const Move prevMove = !node.isNullMove ? node.previousMove : Move::Invalid();
    const Move followupMove = node.parentNode && !node.parentNode->isNullMove ? node.parentNode->previousMove : Move::Invalid();

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        const Move move = moves[i].move;
        ASSERT(move.IsValid());

        const uint32_t piece = (uint32_t)move.GetPiece() - 1;
        const uint32_t from = move.FromSquare().Index();
        const uint32_t to = move.ToSquare().Index();

        // skip PV & TT moves
        if (moves[i].score >= TTMoveValue - 64)
        {
            continue;
        }

        int64_t score = 0;

        if (move.IsCapture())
        {
            const Piece attackingPiece = move.GetPiece();
            const Piece capturedPiece = pos.GetOpponentSide().GetPieceAtSquare(move.ToSquare());
            const int32_t mvvLva = ComputeMvvLvaScore(capturedPiece, attackingPiece);

            if ((uint32_t)attackingPiece < (uint32_t)capturedPiece)
            {
                score = WinningCaptureValue + mvvLva;
            }
            else
            {
                if (pos.StaticExchangeEvaluation(move, 100))
                {
                    score = WinningCaptureValue + mvvLva;
                }
                else if (pos.StaticExchangeEvaluation(move, 0))
                {
                    score = GoodCaptureValue + mvvLva;
                }
                else
                {
                    score = LosingCaptureValue + mvvLva;
                }
            }

            // history heuristics
            //score += capturesHistory[color][from][to];

            // bonus for capturing previously moved piece
            if (prevMove.IsValid() && move.ToSquare() == prevMove.ToSquare())
            {
                score += RecaptureBonus;
            }
        }
        else // non-capture
        {
            // killer moves heuristics
            bool isKiller = false;
            if (node.height < MaxSearchDepth)
            {
                for (uint32_t j = 0; j < NumKillerMoves; ++j)
                {
                    if (move == killerMoves[node.height][j])
                    {
                        score = KillerMoveBonus - j;
                        isKiller = true;
                        break;
                    }
                }
            }

            if (!isKiller)
            {
                // history heuristics
                score += quietMoveHistory[color][from][to];

                // counter move history
                if (prevMove.IsValid())
                {
                    const uint32_t prevPiece = (uint32_t)prevMove.GetPiece() - 1;
                    const uint32_t prevTo = prevMove.ToSquare().Index();
                    score += quietMoveContinuationHistory[prevPiece][prevTo][piece][to] / 2;
                }

                // followup move history
                if (followupMove.IsValid())
                {
                    const uint32_t prevPiece = (uint32_t)followupMove.GetPiece() - 1;
                    const uint32_t prevTo = followupMove.ToSquare().Index();
                    score += quietMoveFollowupHistory[prevPiece][prevTo][piece][to] / 2;
                }
            }
        }

        if (move.IsPromotion())
        {
            const uint32_t pieceIndex = (uint32_t)move.GetPromoteTo();
            ASSERT(pieceIndex > 1 && pieceIndex < 6);
            score += c_PromotionScores[pieceIndex];
        }

        moves[i].score = (int32_t)std::min<int64_t>(score, INT32_MAX);
    }
}
