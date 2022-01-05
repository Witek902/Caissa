#include "MoveOrderer.hpp"
#include "Search.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"

#include <algorithm>
#include <limits>

static constexpr int32_t RecaptureBonus = 10000;
static constexpr int32_t PieceSquareTableScale = 32;

static const int32_t c_PromotionScores[] =
{
    0,          // none
    0,          // pawn
    10000000,   // knight
    0,          // bishop
    0,          // rook
    20000000,   // queen
};

static int32_t ComputeMvvLvaScore(const Piece capturedPiece, const Piece attackingPiece)
{
    return 100 * (int32_t)capturedPiece - (int32_t)attackingPiece;
}

void MoveOrderer::DebugPrint() const
{
    std::cout << "=== HISTORY HEURISTICS ===" << std::endl;

    for (uint32_t fromIndex = 0; fromIndex < 64; ++fromIndex)
    {
        for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
        {
            for (uint32_t color = 0; color < 2; ++color)
            {
                const CounterType count = searchHistory[color][fromIndex][toIndex];

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

    std::cout << std::endl;
    std::cout << "=== KILLER MOVE HEURISTICS ===" << std::endl;

    for (uint32_t d = 0; d < MaxSearchDepth; ++d)
    {
        std::cout << d;

        for (uint32_t i = 0; i < NumKillerMoves; ++i)
        {
            std::cout << " " << killerMoves[d][i].ToString() << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void MoveOrderer::Clear()
{
    memset(searchHistory, 0, sizeof(searchHistory));
    memset(continuationHistory, 0, sizeof(continuationHistory));
    memset(followupHistory, 0, sizeof(followupHistory));
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
    if (numMoves == 1 && depth <= 2)
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

        UpdateHistoryCounter(searchHistory[color][from][to], delta);

        if (prevMove.IsValid())
        {
            const uint32_t prevPiece = (uint32_t)prevMove.GetPiece() - 1;
            const uint32_t prevTo = prevMove.ToSquare().Index();
            UpdateHistoryCounter(continuationHistory[prevPiece][prevTo][piece][to], delta);
        }

        if (followupMove.IsValid())
        {
            const uint32_t prevPiece = (uint32_t)followupMove.GetPiece() - 1;
            const uint32_t prevTo = followupMove.ToSquare().Index();
            UpdateHistoryCounter(followupHistory[prevPiece][prevTo][piece][to], delta);
        }
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

        if (move.IsEnPassant())
        {
            score += MoveOrderer::GoodCaptureValue;
        }
        else if (move.IsCapture())
        {
            const Piece attackingPiece = move.GetPiece();
            const Piece capturedPiece = pos.GetOpponentSide().GetPieceAtSquare(move.ToSquare());
            const int32_t mvvLva = ComputeMvvLvaScore(capturedPiece, attackingPiece);

            if ((uint32_t)attackingPiece < (uint32_t)capturedPiece)
            {
                score += WinningCaptureValue + mvvLva;
            }
            else
            {
                if (pos.StaticExchangeEvaluation(move, 100))
                {
                    score += WinningCaptureValue + mvvLva;
                }
                else if (pos.StaticExchangeEvaluation(move, 0))
                {
                    score += GoodCaptureValue + mvvLva;
                }
                else
                {
                    score += LosingCaptureValue + mvvLva;
                }
            }

            // bonus for capturing previously moved piece
            if (move.ToSquare() == prevMove.ToSquare())
            {
                score += RecaptureBonus;
            }
        }
        else
        {
            score += PieceSquareTableScale * ScoreQuietMove(pos, move);
        }

        if (move.GetPiece() == Piece::Pawn && move.GetPromoteTo() != Piece::None)
        {
            const uint32_t pieceIndex = (uint32_t)move.GetPromoteTo();
            ASSERT(pieceIndex > 1 && pieceIndex < 6);
            score += c_PromotionScores[pieceIndex];
        }

        if (move.IsQuiet())
        {
            // history heuristics
            {
                score += searchHistory[color][from][to];
            }

            // killer moves heuristics
            if (node.height < MaxSearchDepth)
            {
                for (uint32_t j = 0; j < NumKillerMoves; ++j)
                {
                    if (move == killerMoves[node.height][j])
                    {
                        score += KillerMoveBonus - j;
                    }
                }
            }

            // counter move history
            if (prevMove.IsValid())
            {
                const uint32_t prevPiece = (uint32_t)prevMove.GetPiece() - 1;
                const uint32_t prevTo = prevMove.ToSquare().Index();
                score += continuationHistory[prevPiece][prevTo][piece][to] / 2;
            }

            // followup move history
            if (followupMove.IsValid())
            {
                const uint32_t prevPiece = (uint32_t)followupMove.GetPiece() - 1;
                const uint32_t prevTo = followupMove.ToSquare().Index();
                score += followupHistory[prevPiece][prevTo][piece][to] / 2;
            }
        }

        moves[i].score = (int32_t)std::min<int64_t>(score, INT32_MAX);
    }
}
