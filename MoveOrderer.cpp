#include "MoveOrderer.hpp"
#include "Search.hpp"
#include "MoveList.hpp"
#include "Evaluate.hpp"

static const int32_t c_MvvLvaScoreBaseValue = 10000000;

static const int32_t c_PromotionScores[] =
{
    0,          // none
    0,          // pawn
    9000000,    // knight
    0,          // bishop
    0,          // rook
    9000001,    // queen
};

static const int16_t c_PieceValues[] =
{
    0,      // none
    100,    // pawn
    320,    // knight
    330,    // bishop
    500,    // rook
    900,    // queen
};

static int32_t ComputeMvvLvaScore(const Piece attackingPiece, const Piece capturedPiece)
{
    return c_MvvLvaScoreBaseValue + 100 * (int32_t)capturedPiece - (int32_t)attackingPiece;
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
    if (move.isCapture)
    {
        return;
    }

    const uint32_t color = (uint32_t)node.color;

    // update history heuristics
    if (node.depth > 0)
    {
        uint32_t& historyCounter = searchHistory[color][move.fromSquare.Index()][move.toSquare.Index()];

        const uint64_t historyBonus = std::min(1024, node.depth * node.depth);

        const uint64_t newValue = std::min<uint64_t>(UINT32_MAX, (uint64_t)historyCounter + historyBonus);
        historyCounter = (uint32_t)newValue;
    }

    // update killer heuristics
    if (node.height < MaxSearchDepth && killerMoves[node.height][0] != move)
    {
        for (uint32_t j = NumKillerMoves; j-- > 1u; )
        {
            killerMoves[node.height][j] = killerMoves[node.height][j - 1];
        }
        killerMoves[node.height][0] = move;
    }

    if (node.previousMove.IsValid())
    {
        const uint8_t fromIndex = node.previousMove.fromSquare.mIndex;
        const uint8_t toIndex = node.previousMove.toSquare.mIndex;
        counterMoveHistory[color][fromIndex][toIndex] = move;
    }
}

void MoveOrderer::ScoreMoves(const NodeInfo& node, MoveList& moves) const
{
    const Position& pos = *node.position;

    const uint32_t KillerMoveBonus  = 100000;
    const uint32_t CounterMoveBonus = 0;

    const uint32_t color = (uint32_t)node.color;

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

        if (move.isEnPassant)
        {
            score += c_MvvLvaScoreBaseValue;
        }
        else if (move.isCapture)
        {
            const Piece attackingPiece = move.piece;
            const Piece capturedPiece = pos.GetOpponentSide().GetPieceAtSquare(move.toSquare);
            score += ComputeMvvLvaScore(attackingPiece, capturedPiece);
        }
        else
        {
            score += ScoreQuietMove(pos, move);
        }

        if (move.piece == Piece::Pawn && move.promoteTo != Piece::None)
        {
            const uint32_t pieceIndex = (uint32_t)move.promoteTo;
            ASSERT(pieceIndex > 1 && pieceIndex < 6);
            score += c_PromotionScores[pieceIndex];
        }

        // history heuristics
        {
            score += searchHistory[color][move.fromSquare.Index()][move.toSquare.Index()];
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
            const uint8_t fromIndex = node.previousMove.fromSquare.mIndex;
            const uint8_t toIndex = node.previousMove.toSquare.mIndex;

            if (move == counterMoveHistory[color][fromIndex][toIndex])
            {
                score += CounterMoveBonus;
            }
        }

        moves[i].score = (int32_t)std::min<uint64_t>(score, INT32_MAX);
    }
}
