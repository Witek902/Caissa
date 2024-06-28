#include "MovePicker.hpp"
#include "MoveGen.hpp"
#include "Search.hpp"


bool MovePicker::PickMove(const NodeInfo& node, Move& outMove, int32_t& outScore)
{
    switch (m_stage)
    {
        case Stage::TTMove:
        {
            m_stage = Stage::GenerateCaptures;
            const Move move = m_position.MoveFromPacked(m_ttMove);
            if (move.IsValid() && (!move.IsQuiet() || m_generateQuiets))
            {
                outMove = move;
                outScore = MoveOrderer::TTMoveValue;
                return true;
            }
            [[fallthrough]];
        }

        case Stage::GenerateCaptures:
        {
            m_moveIndex = 0;
            m_stage = Stage::Captures;
            m_killerMove = Move::Invalid();
            m_counterMove = Move::Invalid();
            GenerateMoveList<MoveGenerationMode::Captures>(m_position, node.threats.allThreats, m_moves);

            // remove PV and TT moves from generated list
            m_moves.RemoveMove(m_ttMove);

            m_moveOrderer.ScoreMoves(node, m_moves, false);

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

            if (!m_generateQuiets)
            {
                m_stage = Stage::End;
                return false;
            }

            m_stage = Stage::Killer;
            [[fallthrough]];
        }

        case Stage::Killer:
        {
            m_stage = Stage::Counter;
            Move move = m_moveOrderer.GetKillerMove(node.ply);
            if (move.IsValid() && move != m_ttMove)
            {
                move = m_position.MoveFromPacked(move);
                if (move.IsValid() && !move.IsCapture())
                {
                    m_killerMove = move;
                    outMove = move;
                    outScore = MoveOrderer::KillerMoveBonus;
                    return true;
                }
            }
            [[fallthrough]];
        }

        case Stage::Counter:
        {
            m_stage = Stage::GenerateQuiets;
            Move move = m_moveOrderer.GetCounterMove(node);
            if (move.IsValid() && move != m_ttMove && move != m_killerMove)
            {
                move = m_position.MoveFromPacked(move);
                if (move.IsValid() && !move.IsCapture())
                {
                    m_counterMove = move;
                    outMove = move;
                    outScore = MoveOrderer::CounterMoveBonus;
                    return true;
                }
            }
            [[fallthrough]];
        }

        case Stage::GenerateQuiets:
        {
            m_stage = Stage::PickQuiets;
            if (m_generateQuiets)
            {
                GenerateMoveList<MoveGenerationMode::Quiets>(m_position, node.threats.allThreats, m_moves);

                // remove played moves from generated list
                m_moves.RemoveMove(m_ttMove);
                m_moves.RemoveMove(m_killerMove);
                m_moves.RemoveMove(m_counterMove);

                m_moveOrderer.ScoreMoves(node, m_moves, true, m_nodeCacheEntry);
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
