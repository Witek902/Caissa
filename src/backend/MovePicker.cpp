#include "MovePicker.hpp"
#include "MoveOrderer.hpp"
#include "Position.hpp"
#include "TranspositionTable.hpp"
#include "Search.hpp"

bool MovePicker::PickMove(const NodeInfo& node, const Game& game, Move& outMove, int32_t& outScore)
{
    const bool generateQuiets = m_moveGenFlags & MOVE_GEN_MASK_QUIET;

    switch (m_stage)
    {
        case Stage::TTMove:
        {
            m_stage = Stage::GenerateCaptures;
            const Move move = m_position.MoveFromPacked(m_ttMove);
            if (move.IsValid() && (!move.IsQuiet() || generateQuiets))
            {
                outMove = move;
                outScore = MoveOrderer::TTMoveValue;
                return true;
            }
            [[fallthrough]];
        }

        case Stage::GenerateCaptures:
        {
            m_stage = Stage::Captures;
            m_moveIndex = 0;
            m_position.GenerateMoveList(m_moves, m_moveGenFlags & (MOVE_GEN_MASK_CAPTURES | MOVE_GEN_MASK_PROMOTIONS));

            // remove PV and TT moves from generated list
            m_moves.RemoveMove(m_ttMove);

            m_moveOrderer.ScoreMoves(node, game, m_moves, false);

            [[fallthrough]];
        }

        case Stage::Captures:
        {
            if (m_moves.Size() > 0)
            {
                const uint32_t index = m_moves.BestMoveIndex();
                outMove = m_moves.GetMove(index);
                outScore = m_moves.GetScore(index);

                ASSERT(outMove.IsValid());
                ASSERT(outScore > INT32_MIN);

                if (outScore >= MoveOrderer::PromotionValue)
                {
                    m_moves.RemoveByIndex(index);
                    return true;
                }
            }

            if (!generateQuiets)
            {
                m_stage = Stage::End;
                return false;
            }

            m_stage = Stage::Killer1;
            [[fallthrough]];
        }

        case Stage::Killer1:
        {
            m_stage = Stage::Killer2;
            Move move = m_moveOrderer.GetKillerMoves(node.height).moves[0];
            if (move.IsValid() && move != m_ttMove)
            {
                move = m_position.MoveFromPacked(move);
                if (move.IsValid() && !move.IsCapture())
                {
                    m_killerMoves[0] = move;
                    outMove = move;
                    outScore = MoveOrderer::KillerMoveBonus;
                    return true;
                }
            }
            [[fallthrough]];
        }

        case Stage::Killer2:
        {
            m_stage = Stage::GenerateQuiets;
            Move move = m_moveOrderer.GetKillerMoves(node.height).moves[1];
            if (move.IsValid() && move != m_ttMove)
            {
                move = m_position.MoveFromPacked(move);
                if (move.IsValid() && !move.IsCapture())
                {
                    m_killerMoves[1] = move;
                    outMove = move;
                    outScore = MoveOrderer::KillerMoveBonus - 1;
                    return true;
                }
            }
            [[fallthrough]];
        }

        case Stage::Counter:
        {
            m_stage = Stage::GenerateQuiets;
            PackedMove move = m_moveOrderer.GetCounterMove(node.position.GetSideToMove(), node.previousMove);
            if (move.IsValid() && move != m_ttMove)
            {
                const auto& killers = m_moveOrderer.GetKillerMoves(node.height);
                if (move != killers.moves[0] && move != killers.moves[1])
                {
                    const Move counterMove = m_position.MoveFromPacked(move);
                    if (counterMove.IsValid() && !counterMove.IsCapture())
                    {
                        m_counterMove = counterMove;
                        outMove = counterMove;
                        outScore = MoveOrderer::CounterMoveBonus;
                        return true;
                    }
                }
            }
            [[fallthrough]];
        }

        case Stage::GenerateQuiets:
        {
            m_stage = Stage::PickQuiets;
            if (m_moveGenFlags & MOVE_GEN_MASK_QUIET)
            {
                m_position.GenerateMoveList(m_moves, MOVE_GEN_MASK_QUIET);

                // remove PV and TT moves from generated list
                m_moves.RemoveMove(m_ttMove);
                m_moves.RemoveMove(m_killerMoves[0]);
                m_moves.RemoveMove(m_killerMoves[1]);
                m_moves.RemoveMove(m_counterMove);

                m_moveOrderer.ScoreMoves(node, game, m_moves, true, m_nodeCacheEntry);
            }
            [[fallthrough]];
        }

        case Stage::PickQuiets:
        {
            if (m_moves.Size() > 0)
            {
                const uint32_t index = m_moves.BestMoveIndex();
                outMove = m_moves.GetMove(index);
                outScore = m_moves.GetScore(index);

                ASSERT(outMove.IsValid());
                ASSERT(outScore > INT32_MIN);

                m_moves.RemoveByIndex(index);

                return true;
            }

            m_stage = Stage::End;
            break;
        }
    }

    return false;
}
