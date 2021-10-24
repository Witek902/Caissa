#include "MoveOrderer.hpp"
#include "Search.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"

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
                const uint32_t count = searchHistory[color][fromIndex][toIndex];

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
    std::cout << "=== COUNTER MOVE HEURISTICS ===" << std::endl;

    for (uint32_t fromIndex = 0; fromIndex < 64; ++fromIndex)
    {
        for (uint32_t toIndex = 0; toIndex < 64; ++toIndex)
        {
            for (uint32_t color = 0; color < 2; ++color)
            {
                const PackedMove counterMove = counterMoveHistory[color][fromIndex][toIndex];

                if (counterMove.IsValid())
                {
                    std::cout
                        << Square(fromIndex).ToString()
                        << Square(toIndex).ToString()
                        << (color > 0 ? " (black)" : " (white)")
                        << " ==> " << counterMove.ToString() << '\n';
                }
            }
        }
    }

    std::cout << std::endl;
}

void MoveOrderer::Clear()
{
    memset(searchHistory, 0, sizeof(searchHistory));
    memset(killerMoves, 0, sizeof(killerMoves));
    memset(counterMoveHistory, 0, sizeof(counterMoveHistory));
}

void MoveOrderer::OnBetaCutoff(const NodeInfo& node, const Move move)
{
    if (move.IsCapture())
    {
        return;
    }

    const uint32_t color = (uint32_t)node.position->GetSideToMove();

    // update history heuristics
    if (node.depth > 0)
    {
        uint32_t& historyCounter = searchHistory[color][move.FromSquare().Index()][move.ToSquare().Index()];

        const uint64_t historyBonus = node.depth * node.depth;

        const uint64_t newValue = std::min<uint64_t>(UINT32_MAX, (uint64_t)historyCounter + historyBonus);
        historyCounter = (uint32_t)newValue;
    }

    // update killer heuristics
    if (node.height < MaxSearchDepth && killerMoves[node.height][0] != PackedMove(move))
    {
        for (uint32_t j = NumKillerMoves; j-- > 1u; )
        {
            killerMoves[node.height][j] = killerMoves[node.height][j - 1];
        }
        killerMoves[node.height][0] = move;
    }

    if (node.previousMove.IsValid())
    {
        const uint8_t fromIndex = node.previousMove.FromSquare().mIndex;
        const uint8_t toIndex = node.previousMove.ToSquare().mIndex;
        counterMoveHistory[color][fromIndex][toIndex] = move;
    }
}

void MoveOrderer::ScoreMoves(const NodeInfo& node, MoveList& moves) const
{
    const Position& pos = *node.position;

    const uint32_t color = (uint32_t)node.position->GetSideToMove();

    const Move previousMove = node.previousMove;

    for (uint32_t i = 0; i < moves.numMoves; ++i)
    {
        const Move move = moves[i].move;
        ASSERT(move.IsValid());

        // skip PV move
        if (moves[i].score >= TTMoveValue)
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
        }
        else
        {
            score += ScoreQuietMove(pos, move);
        }

        if (move.GetPiece() == Piece::Pawn && move.GetPromoteTo() != Piece::None)
        {
            const uint32_t pieceIndex = (uint32_t)move.GetPromoteTo();
            ASSERT(pieceIndex > 1 && pieceIndex < 6);
            score += c_PromotionScores[pieceIndex];
        }

        // history heuristics
        {
            score += searchHistory[color][move.FromSquare().Index()][move.ToSquare().Index()];
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

        // counter move heuristics
        if (node.previousMove.IsValid() && !node.isNullMove)
        {
            const uint8_t fromIndex = node.previousMove.FromSquare().mIndex;
            const uint8_t toIndex = node.previousMove.ToSquare().mIndex;

            if (move == counterMoveHistory[color][fromIndex][toIndex])
            {
                score += CounterMoveBonus;
            }
        }

        //score += rand() % 2;

        moves[i].score = (int32_t)std::min<uint64_t>(score, INT32_MAX);
    }
}
